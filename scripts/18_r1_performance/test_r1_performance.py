import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
ARTIFACT_ROOT = HERE.parents[1]
_official_env = os.environ.get("SNM_OFFICIAL_REPO")
_official_candidates = (
    ARTIFACT_ROOT / "third_party" / "Newton-Muon-official-r0",
    ARTIFACT_ROOT / "third_party" / "Newton-Muon-official",
)
OFFICIAL = (
    Path(_official_env).expanduser().resolve()
    if _official_env
    else next((path for path in _official_candidates if path.is_dir()), _official_candidates[0])
)


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


builder = load_module("r1_perf_source_builder_test", "r1_perf_source_builder.py")
runner = load_module("run_r1_performance_test", "run_r1_performance.py")


@unittest.skipUnless(OFFICIAL.is_dir(), f"pinned upstream repo unavailable: {OFFICIAL}")
class SourceBuilderTests(unittest.TestCase):
    def test_all_sources_compile(self):
        observed = {}
        for method in builder.METHODS:
            built = builder.build_perf_source(OFFICIAL, method)
            compile(built.source, f"<{method}>", "exec")
            observed[method] = built
        self.assertEqual(set(observed), set(builder.METHODS))
        self.assertEqual(observed["block4"].derived_sha256, observed["diag"].derived_sha256)
        self.assertEqual(observed["diag"].derived_sha256, observed["none"].derived_sha256)

    def test_adamw_is_fused_hidden_matrix_control(self):
        source = builder.build_perf_source(OFFICIAL, "adamw").source
        self.assertIn('R1_METHOD != "adamw"', source)
        self.assertIn("raw_model.transformer.h.parameters(), lr=0.000576", source)
        self.assertIn("fused=True", source)

    def test_dense_full_has_full_statistics_inverse_and_apply(self):
        source = builder.build_perf_source(OFFICIAL, "dense_full").source
        self.assertIn('R1_CPROJ_K_MODE == "dense_full"', source)
        self.assertIn('"kind": "c_proj_full"', source)
        self.assertIn('torch.ops.nanogpt.accum_xtx(z2d', source)
        self.assertIn('"inv_proj_full"', source)
        self.assertIn("torch.linalg.cholesky_ex(work", source)
        self.assertIn('torch.bmm(G, plan["inv_proj_full"]', source)


class RunnerTests(unittest.TestCase):
    def test_generated_source_environment_exposes_official_repo(self):
        environment = runner.environment_for(
            "block4", OFFICIAL / "data" / "fineweb10B", OFFICIAL, 2026, 34, True
        )
        self.assertEqual(environment["PYTHONPATH"].split(runner.os.pathsep)[0], str(OFFICIAL))
        self.assertEqual(environment["R1_METHOD"], "block4")

    def test_rotated_order(self):
        methods = ["diag", "none", "block4"]
        self.assertEqual(runner.rotated_order(methods, 0), methods)
        self.assertEqual(runner.rotated_order(methods, 1), ["none", "block4", "diag"])
        self.assertEqual(runner.rotated_order(methods, 2), ["block4", "diag", "none"])

    def test_parse_log(self):
        content = """R1_METADATA method=diag cproj_k_mode=diag seed=2026 init_sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
step:544/544 val_loss:3.9 train_time:620000ms step_avg:1210.94ms
R1_K_MEMORY k_cov_bytes=1 k_inv_bytes=2 k_state_bytes=3
R1_FINAL_MEMORY optimizer_state_bytes=4 model_parameter_bytes=5
peak memory consumption: 38000 MiB
"""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "terminal.log"
            path.write_text(content, encoding="utf-8")
            parsed = runner.parse_log(path)
        self.assertEqual(parsed["method"], "diag")
        self.assertEqual(parsed["final_step"], 544)
        self.assertEqual(parsed["official_train_time_s"], 620.0)
        self.assertEqual(parsed["k_memory"]["k_state_bytes"], 3)
        self.assertEqual(parsed["peak_memory_mib"], 38000)

    @unittest.skipUnless(OFFICIAL.is_dir(), f"pinned upstream repo unavailable: {OFFICIAL}")
    def test_smoke_manifest_rejects_runtime_change(self):
        built = {"diag": builder.build_perf_source(OFFICIAL, "diag")}
        payload = {
            "status": "complete",
            "methods": ["diag"],
            "runtime": {"torch": "x"},
            "source_sha256": {"diag": built["diag"].derived_sha256},
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "runtime fingerprint"):
                runner.validate_smoke_manifest(path, {"torch": "y"}, built)


if __name__ == "__main__":
    unittest.main()

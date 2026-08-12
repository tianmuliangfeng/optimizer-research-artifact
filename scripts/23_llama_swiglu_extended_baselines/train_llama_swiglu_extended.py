"""Thin, provenance-preserving adapter for LLaMA/SwiGLU extended optimizers.

The validated 124M LLaMA trainer remains unchanged.  This entry point expands
its method vocabulary and replaces only optimizer construction with the
audited NorMuon and Moonlight-Muon implementations used by the R1 extended
baseline experiment.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
BASE_TRAINER_PATH = SCRIPTS_DIR / "17_llama_swiglu_validation" / "train_llama_swiglu.py"
EXTENDED_OPTIMIZERS_DIR = SCRIPTS_DIR / "19_r1_extended_baselines"
METHODS = ("normuon", "moonlight_muon")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def extract_extended_args(argv: list[str]) -> tuple[list[str], float]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--extended-weight-decay", type=float, required=True)
    known, remaining = parser.parse_known_args(argv[1:])
    if known.extended_weight_decay < 0:
        parser.error("--extended-weight-decay must be non-negative")
    return [argv[0], *remaining], float(known.extended_weight_decay)


def main() -> None:
    import torch

    filtered_argv, weight_decay = extract_extended_args(sys.argv)
    sys.argv = filtered_argv
    sys.path.insert(0, str(EXTENDED_OPTIMIZERS_DIR))
    base = load_module("llama_swiglu_base_train_extended", BASE_TRAINER_PATH)
    from extended_optimizers import R1MoonlightMuon, R1NorMuon

    base.METHODS = METHODS
    base.NEWTON_METHODS = ()

    def make_optimizers(model, args):
        backup = [parameter for _, parameter in model.backup_named_parameters()]
        matrix = [parameter for _, parameter in model.matrix_named_parameters()]
        eps = 1e-10 if args.method == "normuon" else 1e-8
        backup_optimizer = torch.optim.AdamW(
            backup,
            lr=args.backup_lr,
            betas=(0.9, 0.95),
            eps=eps,
            weight_decay=weight_decay,
            fused=True,
        )
        if args.method == "normuon":
            matrix_optimizer = R1NorMuon(
                matrix,
                lr=args.matrix_lr,
                weight_decay=weight_decay,
                momentum=0.95,
                beta2=0.95,
                ns_steps=5,
            )
        elif args.method == "moonlight_muon":
            matrix_optimizer = R1MoonlightMuon(
                matrix,
                lr=args.matrix_lr,
                weight_decay=weight_decay,
                momentum=0.95,
                nesterov=True,
                ns_steps=5,
            )
        else:  # guarded by the base parser, retained for defense in depth
            raise ValueError(f"unsupported extended method {args.method!r}")
        return [backup_optimizer, matrix_optimizer], matrix_optimizer

    base.make_optimizers = make_optimizers
    base.main()

    if "--init-only" in sys.argv:
        return

    # The base trainer intentionally knows nothing about extended recipes.
    # Enrich its durable local summary after successful completion.
    args_parser = argparse.ArgumentParser(add_help=False)
    args_parser.add_argument("--output-dir", type=Path, required=True)
    args_parser.add_argument("--method", choices=METHODS, required=True)
    args_parser.add_argument("--backup-lr", type=float, required=True)
    args_parser.add_argument("--matrix-lr", type=float, required=True)
    parsed, _ = args_parser.parse_known_args(sys.argv[1:])
    summary_path = parsed.output_dir.resolve() / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["family"] = "llama_swiglu_124m_extended_baselines"
    payload["extended_optimizer"] = {
        "method": parsed.method,
        "auxiliary_lr": parsed.backup_lr,
        "matrix_lr": parsed.matrix_lr,
        "weight_decay": weight_decay,
        "matrix_parameter_routing": "all_2d_except_tied_token_embedding",
        "auxiliary_parameter_routing": "tied_embedding_plus_rmsnorm_gains",
        "separate_qkv": True,
        "packed_qkv_split_applied": False,
    }
    base.atomic_write_json(summary_path, payload)
    print("LLAMAX_FINAL_SUMMARY " + json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

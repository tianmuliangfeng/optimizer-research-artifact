import copy
import math
import tempfile
import unittest
from pathlib import Path

import torch

from model import GPT, GPTConfig
from optimizer_factory import register_input_cache_hooks, use_muon_family_param
from optimizers import (
    CProjKModeNewtonMuon,
    InputCovState,
    matrix_sign_ns5,
    matrix_sign_svd,
    muon_momentum_direction,
    muon_orthogonalize,
)
from quadratic_probe import (
    _apply_low_rank_cov_inverse,
    _preconditioned_gradient,
    expanded_probe_modes,
    run_cproj_quadratic_probe,
    run_cproj_quadratic_probe_repeated,
    write_quadratic_probe_artifacts,
)
from temporal_quadratic_probe import (
    run_cproj_temporal_quadratic_probe,
    temporal_candidate_names,
    write_temporal_quadratic_probe_artifacts,
)


class QuadraticProbeTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)

    def test_woodbury_matches_dense_inverse(self):
        g = torch.randn(3, 6)
        x = torch.randn(4, 6)
        damping = torch.tensor(0.3)
        k = x.T @ x / x.shape[0]
        expected = g @ torch.linalg.inv(k + damping * torch.eye(k.shape[0]))
        actual = _apply_low_rank_cov_inverse(g, x, damping)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)

    def test_scalar_and_none_have_same_ns_direction(self):
        g = torch.randn(4, 12)
        x = torch.randn(8, 12)
        none_r = _preconditioned_gradient(g, x, mode="none", ridge=0.2, blocks=4)
        scalar_r = _preconditioned_gradient(
            g, x, mode="scalar", ridge=0.2, blocks=4
        )
        none_q = matrix_sign_ns5(none_r, steps=5)
        scalar_q = matrix_sign_ns5(scalar_r, steps=5)
        torch.testing.assert_close(none_q, scalar_q, atol=2e-6, rtol=2e-6)

    def test_blog_muon_momentum_matches_public_lerp_form(self):
        grad = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        initial = torch.tensor([[0.5, -0.5], [1.0, -1.0]])
        buffer = initial.clone()
        beta = 0.95

        direction = muon_momentum_direction(
            grad,
            buffer,
            beta=beta,
            nesterov=True,
            momentum_ema=True,
        )
        expected_buffer = torch.lerp(initial, grad, 1.0 - beta)
        expected_direction = torch.lerp(grad, expected_buffer, beta)
        torch.testing.assert_close(buffer, expected_buffer)
        torch.testing.assert_close(direction, expected_direction)

    def test_blog_muon_qkv_split_and_shape_scaling(self):
        packed_qkv = torch.randn(6, 2)
        actual = muon_orthogonalize(
            packed_qkv,
            name="transformer.h.0.attn.c_attn.weight",
            ns_steps=2,
            eps=1e-8,
            split_qkv=True,
            adjust_lr_for_shape=True,
            ns_compute_dtype="float32",
        )
        expected = torch.cat(
            [
                matrix_sign_ns5(piece, steps=2, eps=1e-8)
                for piece in packed_qkv.split(2, dim=0)
            ],
            dim=0,
        )
        torch.testing.assert_close(actual, expected)

        expansion = torch.randn(8, 2)
        unscaled = muon_orthogonalize(
            expansion,
            name="transformer.h.0.mlp.c_fc.weight",
            ns_steps=2,
            eps=1e-8,
            split_qkv=True,
            adjust_lr_for_shape=False,
            ns_compute_dtype="float32",
        )
        scaled = muon_orthogonalize(
            expansion,
            name="transformer.h.0.mlp.c_fc.weight",
            ns_steps=2,
            eps=1e-8,
            split_qkv=True,
            adjust_lr_for_shape=True,
            ns_compute_dtype="float32",
        )
        torch.testing.assert_close(scaled, 2.0 * unscaled)

    def test_reference_k_initialization_and_refresh_offset(self):
        state = InputCovState(
            n=4,
            device=torch.device("cpu"),
            beta=0.95,
            ridge=0.2,
            refresh_interval=32,
            max_samples=None,
            init_scale=0.001,
            init_inverse_scale=1.0,
            first_refresh_step=31,
        )
        torch.testing.assert_close(state.K, 0.001 * torch.eye(4))
        torch.testing.assert_close(state.K_inv, torch.eye(4))

        activations = torch.randn(16, 4)
        state.maybe_refresh(activations, step=0)
        self.assertEqual(state.num_updates, 0)
        torch.testing.assert_close(state.K_inv, torch.eye(4))

        state.maybe_refresh(activations, step=31)
        self.assertEqual(state.num_updates, 1)
        self.assertFalse(torch.equal(state.K_inv, torch.eye(4)))

    def test_paper_block4_reference_reports_zero_release(self):
        config = GPTConfig(
            block_size=8,
            vocab_size=64,
            n_layer=1,
            n_head=1,
            n_embd=8,
            dropout=0.0,
            bias=False,
        )
        model = GPT(config)
        param_to_name = {
            param: name
            for name, param in model.named_parameters()
            if use_muon_family_param(name, param)
        }
        matrix_params = list(param_to_name)
        param_to_module, handles = register_input_cache_hooks(model, matrix_params)
        optimizer = CProjKModeNewtonMuon(
            matrix_params,
            param_to_module=param_to_module,
            param_to_name=param_to_name,
            lr=0.01,
            momentum=0.9,
            ns_steps=2,
            input_beta=0.5,
            input_ridge=0.2,
            input_refresh=1,
            input_max_samples=None,
            cproj_k_mode="block4",
            cproj_k_blocks=4,
            cproj_k_reference_mode="block4",
        )

        try:
            x = torch.randint(0, config.vocab_size, (1, 6))
            y = torch.randint(0, config.vocab_size, (1, 6))
            _, loss = model(x, y)
            loss.backward()
            optimizer.step()

            stats = optimizer.last_stats
            self.assertEqual(stats["cproj_reference_mode_id"], 1)
            self.assertEqual(stats["k_state_bytes"], stats["full_k_state_bytes"])
            self.assertEqual(stats["k_state_released_bytes"], 0)
            self.assertEqual(stats["k_state_released_fraction"], 0.0)
            self.assertEqual(stats["cproj_k_state_released_bytes"], 0)
            self.assertEqual(stats["cproj_k_state_released_fraction"], 0.0)
        finally:
            for handle in handles:
                handle.remove()

    def test_fp64_svd_returns_fp32_orthogonal_direction(self):
        left, _ = torch.linalg.qr(torch.randn(8, 8, dtype=torch.float64))
        right, _ = torch.linalg.qr(torch.randn(32, 8, dtype=torch.float64))
        singular = torch.logspace(
            0,
            -10,
            8,
            dtype=torch.float64,
        )
        matrix = (left @ torch.diag(singular) @ right.T).float()
        direction = matrix_sign_svd(
            matrix,
            compute_dtype=torch.float64,
        )
        self.assertEqual(direction.dtype, torch.float32)
        identity = torch.eye(direction.shape[0])
        residual = (
            torch.linalg.vector_norm(direction @ direction.T - identity)
            / torch.linalg.vector_norm(identity)
        )
        self.assertLess(float(residual), 1e-5)

    def test_end_to_end_probe_restores_parameters(self):
        config = GPTConfig(
            block_size=8,
            vocab_size=64,
            n_layer=1,
            n_head=1,
            n_embd=8,
            dropout=0.0,
            bias=False,
        )
        model = GPT(config)
        batch = (
            torch.randint(0, config.vocab_size, (1, 6)),
            torch.randint(0, config.vocab_size, (1, 6)),
        )
        heldout = (
            torch.randint(0, config.vocab_size, (1, 6)),
            torch.randint(0, config.vocab_size, (1, 6)),
        )
        param = model.transformer.h[0].mlp.c_proj.weight
        before = param.detach().clone()
        modes = ["none", "scalar", "diag", "block4", "full"]
        rows, line_rows, metadata = run_cproj_quadratic_probe(
            model,
            batch,
            heldout,
            step=0,
            layers=[0],
            modes=modes,
            ridge=0.2,
            blocks=4,
            ns_steps=2,
            matrix_eps=1e-8,
            matrix_learning_rate=0.01,
            line_search_multipliers=[0.0, 0.5],
            exact_hvp=True,
            exact_svd=False,
            line_search=True,
            device_type="cpu",
            autocast_dtype=torch.float32,
        )
        self.assertEqual(len(rows), len(modes))
        self.assertEqual(len(line_rows), len(modes) * 2 * 2)
        self.assertEqual(metadata["direction_rows"], len(modes))
        torch.testing.assert_close(param, before, atol=0.0, rtol=0.0)
        scalar_row = next(row for row in rows if row["mode"] == "scalar")
        self.assertGreaterEqual(scalar_row["direction_cos_vs_none"], 0.9999)
        for row in rows:
            self.assertTrue(math.isfinite(row["alignment_raw"]))
            self.assertTrue(math.isfinite(row["curvature_exact"]))

    def test_repeated_float32_controls_and_quality_gates(self):
        config = GPTConfig(
            block_size=8,
            vocab_size=64,
            n_layer=1,
            n_head=1,
            n_embd=8,
            dropout=0.0,
            bias=False,
        )
        model = GPT(config)

        def batch():
            return (
                torch.randint(0, config.vocab_size, (1, 6)),
                torch.randint(0, config.vocab_size, (1, 6)),
            )

        build_batches = [batch(), batch()]
        heldout_batches = [batch(), batch()]
        param = model.transformer.h[0].mlp.c_proj.weight
        before = param.detach().clone()
        base_modes = ["none", "scalar", "diag", "block4", "full"]
        normmatch_modes = ["diag", "block4", "full"]
        modes = expanded_probe_modes(
            base_modes,
            include_none_repeat=True,
            normmatch_modes=normmatch_modes,
        )
        rows, line_rows, metadata = run_cproj_quadratic_probe_repeated(
            model,
            build_batches,
            heldout_batches,
            step=0,
            layers=[0],
            modes=base_modes,
            include_none_repeat=True,
            normmatch_modes=normmatch_modes,
            ridge=0.2,
            blocks=4,
            ns_steps=2,
            matrix_eps=1e-8,
            matrix_learning_rate=0.01,
            line_search_multipliers=[0.0, 1.0],
            exact_hvp=True,
            exact_svd=True,
            exact_svd_repeats=1,
            line_search=True,
            device_type="cpu",
            autocast_dtype=torch.float32,
        )
        self.assertEqual(len(modes), 9)
        self.assertEqual(len(rows), 2 * len(modes))
        self.assertEqual(len(line_rows), 2 * len(modes) * 3 * 2)
        self.assertEqual(metadata["build_repeats"], 2)
        self.assertEqual(metadata["heldout_batches"], 2)
        torch.testing.assert_close(param, before, atol=0.0, rtol=0.0)

        none_index = {
            (
                row["build_repeat"],
                row["layer"],
                row["eval_split"],
                row["lr_multiplier"],
            ): row["loss_delta"]
            for row in line_rows
            if row["mode"] == "none"
        }
        for row in line_rows:
            if row["mode"] != "none_repeat":
                continue
            key = (
                row["build_repeat"],
                row["layer"],
                row["eval_split"],
                row["lr_multiplier"],
            )
            self.assertEqual(row["loss_delta"], none_index[key])

        normmatched = [
            row for row in rows if row["direction_variant"] == "normmatch"
        ]
        self.assertEqual(len(normmatched), 2 * len(normmatch_modes))
        for row in normmatched:
            self.assertAlmostEqual(
                row["direction_norm_ratio_vs_none"], 1.0, places=6
            )
        self.assertEqual(
            sum(math.isfinite(float(row["ns_svd_cos"])) for row in rows),
            len(modes),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            paths = write_quadratic_probe_artifacts(
                temp_dir,
                rows,
                line_rows,
                config={"probe_precision": "float32"},
                metadata=[metadata],
                expected_steps=[0],
                expected_layers=[0],
                expected_modes=modes,
                expected_build_repeats=2,
                expected_heldout_batches=2,
                line_search_multipliers=[0.0, 1.0],
                exact_hvp=True,
                exact_svd_repeats=1,
                include_none_repeat=True,
                normmatch_modes=normmatch_modes,
                probe_precision="float32",
                line_search=True,
            )
            checks = Path(paths["probe_data_quality_checks"]).read_text(
                encoding="utf-8"
            )
            self.assertNotIn(",fail,", checks)

    def test_temporal_probe_uses_shadow_ema_k_and_momentum(self):
        config = GPTConfig(
            block_size=8,
            vocab_size=64,
            n_layer=1,
            n_head=1,
            n_embd=8,
            dropout=0.0,
            bias=False,
        )
        model = GPT(config)
        param_to_name = {
            param: name
            for name, param in model.named_parameters()
            if use_muon_family_param(name, param)
        }
        matrix_params = list(param_to_name)
        param_to_module, handles = register_input_cache_hooks(
            model,
            matrix_params,
        )
        optimizer = CProjKModeNewtonMuon(
            matrix_params,
            param_to_module=param_to_module,
            param_to_name=param_to_name,
            lr=0.01,
            momentum=0.9,
            ns_steps=2,
            input_beta=0.5,
            input_ridge=0.2,
            input_refresh=1,
            input_max_samples=None,
            cproj_k_mode="none",
            cproj_shadow_k_modes=("diag", "full"),
        )

        def batch():
            return (
                torch.randint(0, config.vocab_size, (1, 6)),
                torch.randint(0, config.vocab_size, (1, 6)),
            )

        try:
            for _ in range(2):
                x, y = batch()
                _, loss = model(x, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            cproj = model.transformer.h[0].mlp.c_proj.weight
            state = optimizer.get_cproj_temporal_probe_state(cproj)
            self.assertEqual(state["actual_mode"], "none")
            self.assertEqual(set(state["shadows"]), {"diag", "full"})
            self.assertGreater(state["shadows"]["diag"]["input_cov"].num_updates, 0)
            self.assertGreater(state["shadows"]["full"]["input_cov"].num_updates, 0)
            self.assertGreater(
                float(state["shadows"]["diag"]["momentum"].norm()),
                0.0,
            )
            self.assertGreater(
                float(state["shadows"]["full"]["momentum"].norm()),
                0.0,
            )

            build_batches = [batch()]
            heldout_batches = [batch()]
            before = cproj.detach().clone()
            rows, line_rows, metadata = run_cproj_temporal_quadratic_probe(
                model,
                optimizer,
                build_batches,
                heldout_batches,
                step=2,
                layers=[0],
                modes=["none", "diag", "full"],
                ridge=0.2,
                blocks=4,
                ns_steps=2,
                matrix_eps=1e-8,
                matrix_learning_rate=0.01,
                line_search_multipliers=[0.0, 1.0],
                exact_hvp=True,
                exact_svd_repeats=1,
                line_search=True,
                device_type="cpu",
                autocast_dtype=torch.float32,
                svd_compute_dtype=torch.float64,
            )
            names = temporal_candidate_names(
                ["none", "diag", "full"],
                include_exact_svd=True,
            )
            self.assertEqual(len(names), 12)
            self.assertEqual(len(rows), 12)
            self.assertEqual(len(line_rows), 12 * 2 * 2)
            self.assertEqual(metadata["direction_rows"], 12)
            torch.testing.assert_close(cproj, before, atol=0.0, rtol=0.0)

            fresh_none = next(
                row
                for row in rows
                if row["candidate"] == "fresh_gradient_none_ns5"
            )
            ema_none = next(
                row
                for row in rows
                if row["candidate"] == "ema_gradient_none_ns5"
            )
            self.assertGreaterEqual(
                ema_none["direction_cos_vs_global_reference"],
                0.999999,
            )
            self.assertEqual(
                fresh_none["direction_fro_norm"],
                ema_none["direction_fro_norm"],
            )
            svd_rows = [row for row in rows if row["projection"] == "svd"]
            self.assertEqual(len(svd_rows), 3)
            self.assertEqual(
                {row["projection_compute_dtype"] for row in svd_rows},
                {"float64"},
            )
            self.assertEqual(metadata["svd_compute_dtype"], "float64")
            self.assertLess(
                max(row["row_orthogonality_residual"] for row in svd_rows),
                1e-4,
            )

            with tempfile.TemporaryDirectory() as temp_dir:
                paths = write_temporal_quadratic_probe_artifacts(
                    temp_dir,
                    rows,
                    line_rows,
                    config={
                        "probe_precision": "float32",
                        "probe_variant": "temporal",
                    },
                    metadata=[metadata],
                    expected_steps=[2],
                    expected_layers=[0],
                    modes=["none", "diag", "full"],
                    expected_build_repeats=1,
                    expected_heldout_batches=1,
                    line_search_multipliers=[0.0, 1.0],
                    exact_hvp=True,
                    exact_svd_repeats=1,
                    probe_precision="float32",
                    line_search=True,
                    svd_compute_dtype="float64",
                )
                checks = Path(paths["probe_data_quality_checks"]).read_text(
                    encoding="utf-8"
                )
                self.assertNotIn(",fail,", checks)
        finally:
            for handle in handles:
                handle.remove()

    def test_shadow_states_do_not_change_none_trajectory(self):
        config = GPTConfig(
            block_size=8,
            vocab_size=64,
            n_layer=1,
            n_head=1,
            n_embd=8,
            dropout=0.0,
            bias=False,
        )
        reference_model = GPT(config)
        shadow_model = copy.deepcopy(reference_model)

        def make_optimizer(model, shadow_modes):
            param_to_name = {
                param: name
                for name, param in model.named_parameters()
                if use_muon_family_param(name, param)
            }
            params = list(param_to_name)
            param_to_module, handles = register_input_cache_hooks(model, params)
            optimizer = CProjKModeNewtonMuon(
                params,
                param_to_module=param_to_module,
                param_to_name=param_to_name,
                lr=0.01,
                momentum=0.9,
                ns_steps=2,
                input_beta=0.5,
                input_ridge=0.2,
                input_refresh=1,
                input_max_samples=None,
                cproj_k_mode="none",
                cproj_shadow_k_modes=shadow_modes,
            )
            return optimizer, handles

        reference_optimizer, reference_handles = make_optimizer(
            reference_model,
            (),
        )
        shadow_optimizer, shadow_handles = make_optimizer(
            shadow_model,
            ("diag", "full"),
        )
        batches = [
            (
                torch.randint(0, config.vocab_size, (1, 6)),
                torch.randint(0, config.vocab_size, (1, 6)),
            )
            for _ in range(2)
        ]
        try:
            for x, y in batches:
                _, reference_loss = reference_model(x, y)
                reference_loss.backward()
                reference_optimizer.step()
                reference_optimizer.zero_grad(set_to_none=True)

                _, shadow_loss = shadow_model(x, y)
                shadow_loss.backward()
                shadow_optimizer.step()
                shadow_optimizer.zero_grad(set_to_none=True)

            reference_params = dict(reference_model.named_parameters())
            shadow_params = dict(shadow_model.named_parameters())
            for name, reference_param in reference_params.items():
                torch.testing.assert_close(
                    shadow_params[name],
                    reference_param,
                    atol=0.0,
                    rtol=0.0,
                )
        finally:
            for handle in reference_handles + shadow_handles:
                handle.remove()


if __name__ == "__main__":
    unittest.main()

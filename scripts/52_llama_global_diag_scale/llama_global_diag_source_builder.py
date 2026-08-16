#!/usr/bin/env python3
"""Derive EX52 trainers/controllers from accepted Experiments 17 and 20."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


PARENT_SOURCE_SHA256 = {
    "scripts/17_llama_swiglu_validation/train_llama_swiglu.py":
        "b72eb0d2a1dfa91b61cd49b4784b3e0739ecebc2fd3228b8f719cec125706f2a",
    "scripts/17_llama_swiglu_validation/run_llama_swiglu_validation.py":
        "9804398a680cb9d2b3553dcba92e104f7c105b0c6a513932d104df1bb2809c1d",
    "scripts/20_llama_swiglu_1b/run_llama_swiglu_1b.py":
        "f55601c5e1d5d898b3f7b51100d02eb552c7a615963828dd26688766b652f7d0",
    "scripts/20_llama_swiglu_1b/train_llama_swiglu_1b.py":
        "043c758f3d5eb5d1abc9e1f9029a8d085a238cf169ef69ba86580014699dc401",
}

# The anonymous release rewrites only portable path defaults in two parent
# controllers.  Both accepted scientific hashes and public package hashes are
# pinned so source derivation remains fail-closed in either tree.
PUBLIC_PARENT_SOURCE_SHA256 = {
    "scripts/17_llama_swiglu_validation/run_llama_swiglu_validation.py":
        "bc9fd326017c2bf0efb747a9bef4f4f8d7d14886fda4ea285f3d9447a969f7b3",
    "scripts/20_llama_swiglu_1b/run_llama_swiglu_1b.py":
        "8ecb634b751017397bc9bd60f032fdd890b2d5c1a4462de7b697e1bdf97ef8c4",
}


@dataclass(frozen=True)
class Sources:
    trainer: str
    runner124: str
    runner1b: str
    wrapper1b: str


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_pinned(repo: Path, relative: str) -> str:
    path = repo / relative
    source = path.read_text(encoding="utf-8")
    observed = sha256_text(source)
    accepted = {PARENT_SOURCE_SHA256[relative]}
    public = PUBLIC_PARENT_SOURCE_SHA256.get(relative)
    if public is not None:
        accepted.add(public)
    if observed not in accepted:
        raise RuntimeError(
            f"accepted parent source drift for {relative}: {observed} not in {sorted(accepted)}"
        )
    return source


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, observed {count}")
    return text.replace(old, new, 1)


def derive_trainer(source: str) -> str:
    source = replace_once(source, 'METHODS = ("adamw", "muon", "newton_full", "down_none", "down_diag")\nNEWTON_METHODS = ("newton_full", "down_none", "down_diag")', 'METHODS = ("adamw", "muon", "newton_full", "down_none", "down_diag", "global_diag")\nNEWTON_METHODS = ("newton_full", "down_none", "down_diag", "global_diag")', "method registration")
    source = replace_once(source, 'class LlamaAttention(nn.Module):\n    def __init__(self, config: ModelConfig, enable_stats: bool) -> None:', 'class LlamaAttention(nn.Module):\n    def __init__(self, config: ModelConfig, enable_stats: bool, global_diag: bool) -> None:', "attention signature")
    source = replace_once(source, '        self.enable_stats = bool(enable_stats)\n        if self.enable_stats:\n            d = config.n_embd\n            self.register_buffer("attn_in_accum", torch.zeros(d, d), persistent=True)\n            self.register_buffer("attn_in_count", torch.zeros(()), persistent=True)\n            self.register_buffer("o_accum", torch.zeros(d, d), persistent=True)\n            self.register_buffer("o_count", torch.zeros(()), persistent=True)', '        self.enable_stats = bool(enable_stats)\n        self.global_diag = bool(global_diag)\n        if self.enable_stats:\n            d = config.n_embd\n            shape = (d,) if self.global_diag else (d, d)\n            self.register_buffer("attn_in_accum", torch.zeros(shape), persistent=True)\n            self.register_buffer("attn_in_count", torch.zeros(()), persistent=True)\n            self.register_buffer("o_accum", torch.zeros(shape), persistent=True)\n            self.register_buffer("o_count", torch.zeros(()), persistent=True)', "attention stats")
    source = replace_once(source, '            torch.ops.llama_swiglu.accum_xtx(\n                x.flatten(0, -2), self.attn_in_accum, self.attn_in_count, xtx_tmp_d\n            )', '            flat = x.flatten(0, -2)\n            if self.global_diag:\n                torch.ops.llama_swiglu.accum_diag(flat, self.attn_in_accum, self.attn_in_count)\n            else:\n                torch.ops.llama_swiglu.accum_xtx(flat, self.attn_in_accum, self.attn_in_count, xtx_tmp_d)', "attention input accumulation")
    source = replace_once(source, '            torch.ops.llama_swiglu.accum_xtx(\n                y.flatten(0, -2), self.o_accum, self.o_count, xtx_tmp_d\n            )', '            flat = y.flatten(0, -2)\n            if self.global_diag:\n                torch.ops.llama_swiglu.accum_diag(flat, self.o_accum, self.o_count)\n            else:\n                torch.ops.llama_swiglu.accum_xtx(flat, self.o_accum, self.o_count, xtx_tmp_d)', "attention output accumulation")
    source = replace_once(source, 'class SwiGLU(nn.Module):\n    def __init__(self, config: ModelConfig, down_mode: str, enable_stats: bool) -> None:', 'class SwiGLU(nn.Module):\n    def __init__(self, config: ModelConfig, down_mode: str, enable_stats: bool, global_diag: bool) -> None:', "MLP signature")
    source = replace_once(source, '        self.enable_stats = bool(enable_stats)\n        self.down_mode = down_mode\n        if self.enable_stats:\n            d = config.n_embd\n            ff = config.intermediate_size\n            self.register_buffer("mlp_in_accum", torch.zeros(d, d), persistent=True)', '        self.enable_stats = bool(enable_stats)\n        self.down_mode = down_mode\n        self.global_diag = bool(global_diag)\n        if self.enable_stats:\n            d = config.n_embd\n            ff = config.intermediate_size\n            shape = (d,) if self.global_diag else (d, d)\n            self.register_buffer("mlp_in_accum", torch.zeros(shape), persistent=True)', "MLP stats")
    source = replace_once(source, '            torch.ops.llama_swiglu.accum_xtx(\n                x.flatten(0, -2), self.mlp_in_accum, self.mlp_in_count, xtx_tmp_d\n            )', '            flat = x.flatten(0, -2)\n            if self.global_diag:\n                torch.ops.llama_swiglu.accum_diag(flat, self.mlp_in_accum, self.mlp_in_count)\n            else:\n                torch.ops.llama_swiglu.accum_xtx(flat, self.mlp_in_accum, self.mlp_in_count, xtx_tmp_d)', "MLP input accumulation")
    source = replace_once(source, 'class LlamaBlock(nn.Module):\n    def __init__(self, config: ModelConfig, down_mode: str, enable_stats: bool) -> None:', 'class LlamaBlock(nn.Module):\n    def __init__(self, config: ModelConfig, down_mode: str, enable_stats: bool, global_diag: bool) -> None:', "block signature")
    source = replace_once(source, '        self.attn = LlamaAttention(config, enable_stats)\n        self.mlp_norm = RMSNorm(config.n_embd, config.rms_norm_eps)\n        self.mlp = SwiGLU(config, down_mode, enable_stats)', '        self.attn = LlamaAttention(config, enable_stats, global_diag)\n        self.mlp_norm = RMSNorm(config.n_embd, config.rms_norm_eps)\n        self.mlp = SwiGLU(config, down_mode, enable_stats, global_diag)', "block children")
    source = replace_once(source, '            "down_diag": "diag",\n            "down_none": "none",', '            "down_diag": "diag",\n            "global_diag": "diag",\n            "down_none": "none",', "down route")
    source = replace_once(source, '        self.layers = nn.ModuleList(\n            [LlamaBlock(config, down_mode, enable_stats) for _ in range(config.n_layer)]\n        )', '        global_diag = method == "global_diag"\n        self.layers = nn.ModuleList(\n            [LlamaBlock(config, down_mode, enable_stats, global_diag) for _ in range(config.n_layer)]\n        )', "block creation")
    source = replace_once(source, '            "xtx_tmp_d", torch.empty(config.n_embd, config.n_embd), persistent=False\n        )', '            "xtx_tmp_d", torch.empty(1, 1) if global_diag else torch.empty(config.n_embd, config.n_embd), persistent=False\n        )', "dense workspace removal")
    source = source.replace('"kind": "dense",', '"kind": "diag" if self.method == "global_diag" else "dense",', 3)
    if source.count('"kind": "diag" if self.method == "global_diag" else "dense",') != 3:
        raise RuntimeError("expected exactly three converted shared-input group kinds")
    source = replace_once(source, '    return {\n        "architecture": "llama_swiglu_parameter_matched_r1",', '    return {\n        "global_diag_route": model.method == "global_diag",\n        "dense_k_workspace_allowed": model.method != "global_diag",\n        "architecture": "llama_swiglu_parameter_matched_r1",', "architecture route certificate")
    compile(source, "<ex52_global_diag_trainer>", "exec")
    return source


def derive_runner124(source: str) -> str:
    source = replace_once(source, 'METHOD_ORDER = ("down_diag", "down_none", "newton_full", "muon", "adamw")', 'METHOD_ORDER = ("global_diag",)', "124M method order")
    source = replace_once(
        source,
        '    data_dir = (args.official_repo / "data" / "fineweb10B").resolve()',
        '    data_dir = Path(os.environ.get("EX52_DATA_DIR", str(args.official_repo / "data" / "fineweb10B"))).expanduser().resolve()',
        "explicit EX52 data directory",
    )
    source = replace_once(source, '            "newton_full": 48,\n            "down_diag": 48,', '            "newton_full": 48,\n            "global_diag": 48,\n            "down_diag": 48,', "124M expected groups")
    compile(source, "<ex52_124_runner>", "exec")
    return source


def derive_runner1b(source: str, trainer_sha: str) -> str:
    old_pin = 'PINNED_BASE_TRAINER_SHA256 = "b72eb0d2a1dfa91b61cd49b4784b3e0739ecebc2fd3228b8f719cec125706f2a"'
    source = replace_once(source, old_pin, f'PINNED_BASE_TRAINER_SHA256 = "{trainer_sha}"', "1B trainer pin")
    source = replace_once(source, '        "newton_full": 72,\n        "down_diag": 72,', '        "newton_full": 72,\n        "global_diag": 72,\n        "down_diag": 72,', "1B expected groups")
    source = replace_once(source, 'METHOD_ORDER = ("down_diag", "down_none", "newton_full", "muon", "adamw")', 'METHOD_ORDER = ("global_diag",)', "1B method order")
    compile(source, "<ex52_1b_runner>", "exec")
    return source


def build(repo: Path) -> Sources:
    trainer = derive_trainer(
        read_pinned(repo, "scripts/17_llama_swiglu_validation/train_llama_swiglu.py")
    )
    trainer_sha = sha256_text(trainer)
    return Sources(
        trainer=trainer,
        runner124=derive_runner124(
            read_pinned(repo, "scripts/17_llama_swiglu_validation/run_llama_swiglu_validation.py")
        ),
        runner1b=derive_runner1b(
            read_pinned(repo, "scripts/20_llama_swiglu_1b/run_llama_swiglu_1b.py"),
            trainer_sha,
        ),
        wrapper1b=read_pinned(
            repo, "scripts/20_llama_swiglu_1b/train_llama_swiglu_1b.py"
        ),
    )


def expected_k_state_bytes(scale: str) -> int:
    if scale == "124m":
        return 12 * (3 * 768 + 2048) * 8
    if scale == "1b":
        return 18 * (3 * 2048 + 5504) * 8
    raise ValueError(scale)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    sources = build(args.repo.resolve())
    targets = {
        "17_llama_swiglu_validation/train_llama_swiglu.py": sources.trainer,
        "17_llama_swiglu_validation/run_llama_swiglu_validation.py": sources.runner124,
        "20_llama_swiglu_1b/run_llama_swiglu_1b.py": sources.runner1b,
        "20_llama_swiglu_1b/train_llama_swiglu_1b.py": sources.wrapper1b,
    }
    for relative, text in targets.items():
        path = args.output_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        print(f"{relative} sha256={sha256_text(text)}")

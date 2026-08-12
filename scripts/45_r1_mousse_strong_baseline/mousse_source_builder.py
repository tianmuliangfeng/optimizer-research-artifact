"""Derive the controlled Mousse-R1 trainer from the pinned official R1 source."""

from __future__ import annotations

import difflib
import hashlib
from dataclasses import dataclass
from pathlib import Path


OFFICIAL_R1_SCRIPT = "train_gpt_muon_1.py"
OFFICIAL_R1_CANONICAL_SHA256 = "8e3e990a9a010a9f8ddee0e6d111ac7b83acedc7f41d2d3370de5f404c9aab59"


@dataclass(frozen=True)
class DerivedSource:
    base_script: str
    base_canonical_sha256: str
    derived_sha256: str
    source: str
    unified_diff: str


def canonical_bytes(raw: bytes) -> bytes:
    return raw.replace(b"\r\n", b"\n")


def canonical_sha256(raw: bytes) -> str:
    return hashlib.sha256(canonical_bytes(raw)).hexdigest()


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Mousse source derivation expected one {label!r} anchor, observed {count}")
    return source.replace(old, new, 1)


CONTROL = '''args = Hyperparameters()

# Experiment-45 controlled R1 overlay.  Only the hidden-matrix optimizer is
# replaced; model, loader, batch, precision, schedule, validation, and the
# tied embedding/head AdamW route remain the pinned R1 contract.
R1M_SEED = int(os.environ["R1M_SEED"])
R1M_DATA_DIR = os.environ["R1M_DATA_DIR"]
R1M_TOTAL_STEPS = int(os.environ["R1M_TOTAL_STEPS"])
R1M_WARMDOWN_STEPS = int(os.environ["R1M_WARMDOWN_STEPS"])
R1M_VAL_EVERY = int(os.environ["R1M_VAL_EVERY"])
R1M_VAL_TOKENS = int(os.environ["R1M_VAL_TOKENS"])
R1M_AUX_LR = float(os.environ["R1M_AUX_LR"])
R1M_MATRIX_LR = float(os.environ["R1M_MATRIX_LR"])
R1M_MATRIX_WEIGHT_DECAY = float(os.environ["R1M_MATRIX_WEIGHT_DECAY"])
R1M_INIT_ONLY = os.environ.get("R1M_INIT_ONLY", "0") == "1"
R1M_DISABLE_CHECKPOINT = os.environ.get("R1M_DISABLE_CHECKPOINT", "1") == "1"
if R1M_TOTAL_STEPS < 2:
    raise ValueError("R1M_TOTAL_STEPS must be at least 2")
if not 1 <= R1M_WARMDOWN_STEPS <= R1M_TOTAL_STEPS:
    raise ValueError("R1M_WARMDOWN_STEPS must be in [1, total_steps]")
args.input_bin = os.path.join(R1M_DATA_DIR, "fineweb_train_*.bin")
args.input_val_bin = os.path.join(R1M_DATA_DIR, "fineweb_val_*.bin")
args.num_iterations = R1M_TOTAL_STEPS
args.warmdown_iters = R1M_WARMDOWN_STEPS
args.val_loss_every = R1M_VAL_EVERY
args.val_tokens = R1M_VAL_TOKENS
args.save_every = 0
'''


SEED_HELPERS = '''torch.cuda.set_device(0)

random.seed(R1M_SEED)
np.random.seed(R1M_SEED)
torch.manual_seed(R1M_SEED)
torch.cuda.manual_seed_all(R1M_SEED)

def _r1m_model_init_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            tensor = parameter.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(memoryview(tensor.numpy()).cast("B"))
    return digest.hexdigest()
'''


MODEL_AUDIT = '''model = GPT(GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=12, n_embd=768))
r1m_init_sha256 = _r1m_model_init_sha256(model)
r1m_routing = parameter_routing_audit(model)
print("R1M_METADATA " + json.dumps({"method": "mousse", "seed": R1M_SEED, "init_sha256": r1m_init_sha256}, sort_keys=True))
print("R1M_ROUTING " + json.dumps(r1m_routing, sort_keys=True))
if R1M_INIT_ONLY:
    raise SystemExit(0)
model = model.cuda()
'''


OPTIMIZERS = '''hidden_parameters = list(raw_model.transformer.h.parameters())
auxiliary_parameters = list(raw_model.lm_head.parameters())
optimizer1 = torch.optim.AdamW(
    auxiliary_parameters,
    lr=R1M_AUX_LR,
    betas=(0.9, 0.95),
    weight_decay=0.0,
    fused=True,
)
optimizer2 = R1Mousse(
    hidden_parameters,
    lr=R1M_MATRIX_LR,
    weight_decay=R1M_MATRIX_WEIGHT_DECAY,
    momentum=0.95,
    nesterov=False,
    factor_beta=0.95,
    factor_epsilon=1e-5,
    factor_alpha=0.125,
    refresh_interval=10,
    bias_correction=True,
    grafting=True,
    adjust_lr="spectral_norm",
    ns_epsilon=1e-8,
)
optimizers = [optimizer1, optimizer2]
print(
    "R1M_HYPERPARAMS "
    + json.dumps(
        {
            "method": "mousse",
            "aux_lr": R1M_AUX_LR,
            "matrix_lr": R1M_MATRIX_LR,
            "matrix_weight_decay": R1M_MATRIX_WEIGHT_DECAY,
            "momentum": 0.95,
            "nesterov": False,
            "factor_beta": 0.95,
            "factor_epsilon": 1e-5,
            "factor_alpha": 0.125,
            "refresh_interval": 10,
            "bias_correction": True,
            "grafting": True,
            "adjust_lr": "spectral_norm",
            "ns_epsilon": 1e-8,
            "total_steps": R1M_TOTAL_STEPS,
            "warmdown_steps": R1M_WARMDOWN_STEPS,
            "val_every": R1M_VAL_EVERY,
            "val_tokens": R1M_VAL_TOKENS,
        },
        sort_keys=True,
    )
)
'''


FINAL_AUDIT = '''if master_process:
    r1m_state = optimizer_state_breakdown(optimizers)
    r1m_state["model_parameter_bytes"] = sum(
        parameter.numel() * parameter.element_size() for parameter in raw_model.parameters()
    )
    r1m_state["state_schema"] = state_schema(optimizer2)
    print("R1M_FINAL_MEMORY " + json.dumps(r1m_state, sort_keys=True))
    print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
'''


def build_source(official_repo: Path) -> DerivedSource:
    base_path = official_repo / OFFICIAL_R1_SCRIPT
    raw = base_path.read_bytes()
    observed_hash = canonical_sha256(raw)
    if observed_hash != OFFICIAL_R1_CANONICAL_SHA256:
        raise RuntimeError(
            f"official R1 base hash mismatch: {observed_hash} != {OFFICIAL_R1_CANONICAL_SHA256}"
        )
    base = canonical_bytes(raw).decode("utf-8")
    source = base
    source = _replace_once(
        source,
        "import time\nfrom dataclasses import dataclass\n",
        "import time\nimport random\nimport hashlib\nimport json\nfrom dataclasses import dataclass\n",
        "common imports",
    )
    source = _replace_once(
        source,
        "import torch.nn.functional as F\n",
        '''import torch.nn.functional as F
from mousse_optimizer import (
    R1Mousse,
    optimizer_state_breakdown,
    parameter_routing_audit,
    state_schema,
)
''',
        "Mousse optimizer import",
    )
    source = _replace_once(source, "args = Hyperparameters()\n", CONTROL, "args")
    source = _replace_once(source, "torch.cuda.set_device(0)\n", SEED_HELPERS, "seed setup")
    source = _replace_once(
        source,
        '''model = GPT(GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=12, n_embd=768))
model = model.cuda()
''',
        MODEL_AUDIT,
        "model initialization",
    )
    source = _replace_once(
        source,
        '''optimizer1 = torch.optim.AdamW(raw_model.lm_head.parameters(), lr=args.learning_rate, betas=(0.9, 0.95),
                               weight_decay=args.weight_decay, fused=True)
optimizer2 = Muon(raw_model.transformer.h.parameters(), lr=0.1*args.learning_rate, momentum=0.95)
optimizers = [optimizer1, optimizer2]
''',
        OPTIMIZERS,
        "optimizer block",
    )
    source = _replace_once(
        source,
        "if master_process and (last_step or (args.save_every > 0 and step % args.save_every == 0)):\n",
        "if master_process and (not R1M_DISABLE_CHECKPOINT) and (last_step or (args.save_every > 0 and step % args.save_every == 0)):\n",
        "checkpoint guard",
    )
    source = _replace_once(
        source,
        '''if master_process:
    print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
''',
        FINAL_AUDIT,
        "final audit",
    )
    compile(source, "<Mousse-R1-adaptation>", "exec")
    diff = "".join(
        difflib.unified_diff(
            base.splitlines(keepends=True),
            source.splitlines(keepends=True),
            fromfile=f"official/{OFFICIAL_R1_SCRIPT}",
            tofile="experiment45/train_r1_mousse.py",
        )
    )
    return DerivedSource(
        base_script=OFFICIAL_R1_SCRIPT,
        base_canonical_sha256=observed_hash,
        derived_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        source=source,
        unified_diff=diff,
    )

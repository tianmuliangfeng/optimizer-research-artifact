"""Derive controlled R1 extended-baseline sources from official AdamW code."""

from __future__ import annotations

import difflib
import hashlib
from dataclasses import dataclass
from pathlib import Path


ALLOWED_METHODS = ("adamw", "normuon", "moonlight_muon")
OFFICIAL_ADAM_SCRIPT = "train_gpt_adam_1.py"
OFFICIAL_ADAM_CANONICAL_SHA256 = (
    "2c0a9e119b2529502ca6659376daf8f628b68abf8e8a0579a5813cb06b718f75"
)


@dataclass(frozen=True)
class DerivedSource:
    method: str
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
        raise RuntimeError(
            f"extended source derivation expected one {label!r} anchor, observed {count}"
        )
    return source.replace(old, new, 1)


COMMON_CONTROL = '''args = Hyperparameters()

# Controlled R1 extended-baseline overlay. The model, data order, global batch,
# and precision remain those of the pinned official Newton-Muon-1 experiment.
R1X_METHOD = os.environ["R1X_METHOD"]
R1X_SEED = int(os.environ["R1X_SEED"])
R1X_DATA_DIR = os.environ["R1X_DATA_DIR"]
R1X_TOTAL_STEPS = int(os.environ["R1X_TOTAL_STEPS"])
R1X_WARMDOWN_STEPS = int(os.environ["R1X_WARMDOWN_STEPS"])
R1X_VAL_EVERY = int(os.environ["R1X_VAL_EVERY"])
R1X_VAL_TOKENS = int(os.environ["R1X_VAL_TOKENS"])
R1X_AUX_LR = float(os.environ["R1X_AUX_LR"])
R1X_MATRIX_LR = float(os.environ["R1X_MATRIX_LR"])
R1X_WEIGHT_DECAY = float(os.environ["R1X_WEIGHT_DECAY"])
R1X_INIT_ONLY = os.environ.get("R1X_INIT_ONLY", "0") == "1"
R1X_DISABLE_CHECKPOINT = os.environ.get("R1X_DISABLE_CHECKPOINT", "1") == "1"
if R1X_METHOD not in ("adamw", "normuon", "moonlight_muon"):
    raise ValueError(f"unsupported R1X_METHOD={R1X_METHOD!r}")
if R1X_TOTAL_STEPS < 2:
    raise ValueError("R1X_TOTAL_STEPS must be at least 2")
if not 1 <= R1X_WARMDOWN_STEPS <= R1X_TOTAL_STEPS:
    raise ValueError("R1X_WARMDOWN_STEPS must be in [1, total_steps]")
args.input_bin = os.path.join(R1X_DATA_DIR, "fineweb_train_*.bin")
args.input_val_bin = os.path.join(R1X_DATA_DIR, "fineweb_val_*.bin")
args.num_iterations = R1X_TOTAL_STEPS
args.warmdown_iters = R1X_WARMDOWN_STEPS
args.val_loss_every = R1X_VAL_EVERY
args.val_tokens = R1X_VAL_TOKENS
args.save_every = 0
'''


SEED_AND_AUDIT_HELPERS = '''torch.cuda.set_device(0)

random.seed(R1X_SEED)
np.random.seed(R1X_SEED)
torch.manual_seed(R1X_SEED)
torch.cuda.manual_seed_all(R1X_SEED)

def _r1x_model_init_sha256(module: nn.Module) -> str:
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
r1x_init_sha256 = _r1x_model_init_sha256(model)
r1x_routing = parameter_routing_audit(model)
print(
    "R1X_METADATA "
    + json.dumps(
        {
            "method": R1X_METHOD,
            "seed": R1X_SEED,
            "init_sha256": r1x_init_sha256,
        },
        sort_keys=True,
    )
)
print("R1X_ROUTING " + json.dumps(r1x_routing, sort_keys=True))
if R1X_INIT_ONLY:
    raise SystemExit(0)
model = model.cuda()
'''


OPTIMIZER_BLOCKS = {
    "adamw": '''if R1X_METHOD != "adamw":
    raise ValueError("AdamW derived source received a different method")
hidden_parameters = list(raw_model.transformer.h.parameters())
auxiliary_parameters = list(raw_model.lm_head.parameters())
optimizer1 = torch.optim.AdamW(
    auxiliary_parameters,
    lr=R1X_AUX_LR,
    betas=(0.9, 0.95),
    weight_decay=R1X_WEIGHT_DECAY,
    fused=True,
)
optimizer2 = torch.optim.AdamW(
    hidden_parameters,
    lr=R1X_MATRIX_LR,
    betas=(0.9, 0.95),
    weight_decay=R1X_WEIGHT_DECAY,
    fused=True,
)
optimizers = [optimizer1, optimizer2]
''',
    "normuon": '''if R1X_METHOD != "normuon":
    raise ValueError("NorMuon derived source received a different method")
hidden_parameters = list(raw_model.transformer.h.parameters())
auxiliary_parameters = list(raw_model.lm_head.parameters())
optimizer1 = torch.optim.AdamW(
    auxiliary_parameters,
    lr=R1X_AUX_LR,
    betas=(0.9, 0.95),
    eps=1e-10,
    weight_decay=R1X_WEIGHT_DECAY,
    fused=True,
)
optimizer2 = R1NorMuon(
    hidden_parameters,
    lr=R1X_MATRIX_LR,
    weight_decay=R1X_WEIGHT_DECAY,
    momentum=0.95,
    beta2=0.95,
    ns_steps=5,
)
optimizers = [optimizer1, optimizer2]
''',
    "moonlight_muon": '''if R1X_METHOD != "moonlight_muon":
    raise ValueError("Moonlight Muon derived source received a different method")
hidden_parameters = list(raw_model.transformer.h.parameters())
auxiliary_parameters = list(raw_model.lm_head.parameters())
optimizer1 = torch.optim.AdamW(
    auxiliary_parameters,
    lr=R1X_AUX_LR,
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=R1X_WEIGHT_DECAY,
    fused=True,
)
optimizer2 = R1MoonlightMuon(
    hidden_parameters,
    lr=R1X_MATRIX_LR,
    weight_decay=R1X_WEIGHT_DECAY,
    momentum=0.95,
    nesterov=True,
    ns_steps=5,
)
optimizers = [optimizer1, optimizer2]
''',
}


HYPERPARAMETER_REPORT = '''print(
    "R1X_HYPERPARAMS "
    + json.dumps(
        {
            "method": R1X_METHOD,
            "aux_lr": R1X_AUX_LR,
            "matrix_lr": R1X_MATRIX_LR,
            "weight_decay": R1X_WEIGHT_DECAY,
            "total_steps": R1X_TOTAL_STEPS,
            "warmdown_steps": R1X_WARMDOWN_STEPS,
            "val_every": R1X_VAL_EVERY,
            "val_tokens": R1X_VAL_TOKENS,
        },
        sort_keys=True,
    )
)
'''


FINAL_MEMORY = '''if master_process:
    r1x_state = optimizer_state_breakdown(optimizers)
    r1x_state["model_parameter_bytes"] = sum(
        parameter.numel() * parameter.element_size()
        for parameter in raw_model.parameters()
    )
    print("R1X_FINAL_MEMORY " + json.dumps(r1x_state, sort_keys=True))
    print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
'''


def build_source(official_repo: Path, method: str) -> DerivedSource:
    if method not in ALLOWED_METHODS:
        raise ValueError(f"unknown extended-baseline method: {method}")
    base_path = official_repo / OFFICIAL_ADAM_SCRIPT
    raw = base_path.read_bytes()
    observed_hash = canonical_sha256(raw)
    if observed_hash != OFFICIAL_ADAM_CANONICAL_SHA256:
        raise RuntimeError(
            "official AdamW base hash mismatch: "
            f"{observed_hash} != {OFFICIAL_ADAM_CANONICAL_SHA256}"
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
from extended_optimizers import (
    R1MoonlightMuon,
    R1NorMuon,
    optimizer_state_breakdown,
    parameter_routing_audit,
)
''',
        "extended optimizer import",
    )
    source = _replace_once(source, "args = Hyperparameters()\n", COMMON_CONTROL, "args")
    source = _replace_once(
        source,
        "torch.cuda.set_device(0)\n",
        SEED_AND_AUDIT_HELPERS,
        "seed setup",
    )
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
optimizer2 = torch.optim.AdamW(raw_model.transformer.h.parameters(), lr=0.16 * args.learning_rate, betas=(0.9, 0.95),
                               weight_decay=args.weight_decay, fused=True)
optimizers = [optimizer1, optimizer2]
''',
        OPTIMIZER_BLOCKS[method] + HYPERPARAMETER_REPORT,
        "optimizer block",
    )
    source = _replace_once(
        source,
        "if master_process and (last_step or (args.save_every > 0 and step % args.save_every == 0)):\n",
        "if master_process and (not R1X_DISABLE_CHECKPOINT) and (last_step or (args.save_every > 0 and step % args.save_every == 0)):\n",
        "checkpoint guard",
    )
    source = _replace_once(
        source,
        '''if master_process:
    print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
''',
        FINAL_MEMORY,
        "final memory",
    )
    compile(source, f"<R1-extended-{method}>", "exec")
    diff = "".join(
        difflib.unified_diff(
            base.splitlines(keepends=True),
            source.splitlines(keepends=True),
            fromfile=f"official/{OFFICIAL_ADAM_SCRIPT}",
            tofile=f"r1_extended/train_{method}.py",
        )
    )
    return DerivedSource(
        method=method,
        base_script=OFFICIAL_ADAM_SCRIPT,
        base_canonical_sha256=observed_hash,
        derived_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        source=source,
        unified_diff=diff,
    )

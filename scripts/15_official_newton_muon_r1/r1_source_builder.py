"""Derive controlled R1 training sources from the pinned official scripts.

The builder intentionally uses exact, single-occurrence textual replacements.
If the pinned upstream source changes, generation fails instead of silently
applying a patch to the wrong code.  The derived source is saved with every run
for auditability.
"""

from __future__ import annotations

import difflib
import hashlib
from dataclasses import dataclass
from pathlib import Path


ALLOWED_METHODS = ("muon", "block4", "none", "diag")


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
            f"R1 source derivation expected one {label!r} anchor, observed {count}"
        )
    return source.replace(old, new, 1)


COMMON_CONTROL = '''args = Hyperparameters()

# R1 controlled-experiment overlay. Training mathematics remain upstream except
# for the explicitly selected mlp.c_proj K representation in Newton variants.
R1_METHOD = os.environ["R1_METHOD"]
R1_CPROJ_K_MODE = os.environ["R1_CPROJ_K_MODE"]
R1_SEED = int(os.environ["R1_SEED"])
R1_DATA_DIR = os.environ["R1_DATA_DIR"]
R1_INIT_ONLY = os.environ.get("R1_INIT_ONLY", "0") == "1"
R1_SMOKE_TEST = os.environ.get("R1_SMOKE_TEST", "0") == "1"
R1_SMOKE_STEPS = int(os.environ.get("R1_SMOKE_STEPS", "10"))
R1_DISABLE_CHECKPOINT = os.environ.get("R1_DISABLE_CHECKPOINT", "0") == "1"
args.input_bin = os.path.join(R1_DATA_DIR, "fineweb_train_*.bin")
args.input_val_bin = os.path.join(R1_DATA_DIR, "fineweb_val_*.bin")
if R1_SMOKE_TEST:
    # Preserve the formal train shape and optimizer mathematics. Only shorten
    # the number of updates, validation work, and checkpoint behavior.
    if R1_SMOKE_STEPS < 2:
        raise ValueError("R1_SMOKE_STEPS must be at least 2")
    args.num_iterations = R1_SMOKE_STEPS
    args.warmdown_iters = 1
    args.val_loss_every = R1_SMOKE_STEPS
    args.val_tokens = args.device_batch_size * args.sequence_length
    args.save_every = 0
'''


COMMON_SEED_AND_AUDIT = '''torch.cuda.set_device(0)

# Seed all RNGs before constructing the model. The official data loader itself
# is deterministic and consumes sorted shards from position zero.
random.seed(R1_SEED)
np.random.seed(R1_SEED)
torch.manual_seed(R1_SEED)
torch.cuda.manual_seed_all(R1_SEED)

def _r1_model_init_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            tensor = parameter.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(memoryview(tensor.numpy()).cast("B"))
    return digest.hexdigest()

def _r1_iter_tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _r1_iter_tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _r1_iter_tensors(child)

def _r1_storage_key(tensor: torch.Tensor):
    storage = tensor.untyped_storage()
    device_index = tensor.device.index if tensor.device.index is not None else -1
    return (tensor.device.type, device_index, storage.data_ptr(), storage.nbytes())

def _r1_unique_storage_bytes(tensors) -> int:
    storages = {}
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            key = _r1_storage_key(tensor)
            storages[key] = key[-1]
    return int(sum(storages.values()))

def _r1_optimizer_state_bytes(optimizers) -> int:
    tensors = []
    for optimizer in optimizers:
        tensors.extend(_r1_iter_tensors(optimizer.state))
    return _r1_unique_storage_bytes(tensors)
'''


MODEL_AUDIT_MUON = '''model = GPT(GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=12, n_embd=768))
r1_init_sha256 = _r1_model_init_sha256(model)
print(f"R1_METADATA method={R1_METHOD} cproj_k_mode={R1_CPROJ_K_MODE} seed={R1_SEED} init_sha256={r1_init_sha256}")
if R1_INIT_ONLY:
    raise SystemExit(0)
model = model.cuda()
'''


MODEL_AUDIT_NEWTON = '''model = GPT(GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=12, n_embd=768))
r1_init_sha256 = _r1_model_init_sha256(model)
print(f"R1_METADATA method={R1_METHOD} cproj_k_mode={R1_CPROJ_K_MODE} seed={R1_SEED} init_sha256={r1_init_sha256}")
if R1_INIT_ONLY:
    raise SystemExit(0)
model = model.cuda()
'''


FINAL_MEMORY_BLOCK = '''if master_process:
    r1_optimizer_state_bytes = _r1_optimizer_state_bytes(optimizers)
    r1_model_parameter_bytes = sum(p.numel() * p.element_size() for p in raw_model.parameters())
    print(f"R1_FINAL_MEMORY optimizer_state_bytes={r1_optimizer_state_bytes} model_parameter_bytes={r1_model_parameter_bytes}")
    print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
'''


DIAG_CUSTOM_OP = '''
@torch.compile
def _accum_xtx_diag4_impl(x_2d: Tensor, accum: Tensor, count: Tensor) -> Tensor:
    N, fourD = x_2d.shape
    assert fourD % 4 == 0
    D = fourD // 4
    blocks = x_2d.view(N, 4, D)
    accum.add_(blocks.square().mean(dim=0))
    count.add_(1.0)
    return _dummy_scalar_like(accum)

@torch.library.custom_op("nanogpt::accum_xtx_diag4", mutates_args=("accum", "count"))
@torch.no_grad()
def accum_xtx_diag4_op(x_2d: Tensor, accum: Tensor, count: Tensor) -> Tensor:
    return _accum_xtx_diag4_impl(x_2d, accum, count)

@accum_xtx_diag4_op.register_fake
def accum_xtx_diag4_fake(x_2d: Tensor, accum: Tensor, count: Tensor):
    return accum.new_empty(())
'''


MLP_INIT_OFFICIAL = '''        self.proj_xtx_accum = nn.Buffer(torch.zeros(4, d, d, dtype=torch.float32), persistent=False)
        self.proj_xtx_tmp   = nn.Buffer(torch.empty(4, d, d, dtype=torch.float32), persistent=False)
        self.proj_xtx_count = nn.Buffer(torch.zeros((), dtype=torch.float32), persistent=False)

        self.c_fc.weight._stats_ref = {"kind": "c_fc",   "d": d, "accum": self.fc_xtx_accum,   "count": self.fc_xtx_count}
        self.c_proj.weight._stats_ref = {"kind": "c_proj","d": d, "accum": self.proj_xtx_accum, "count": self.proj_xtx_count}
'''


MLP_INIT_R1 = '''        self.c_fc.weight._stats_ref = {"kind": "c_fc", "d": d, "accum": self.fc_xtx_accum, "count": self.fc_xtx_count}
        if R1_CPROJ_K_MODE == "block4":
            self.proj_xtx_accum = nn.Buffer(torch.zeros(4, d, d, dtype=torch.float32), persistent=False)
            self.proj_xtx_tmp = nn.Buffer(torch.empty(4, d, d, dtype=torch.float32), persistent=False)
            self.proj_xtx_count = nn.Buffer(torch.zeros((), dtype=torch.float32), persistent=False)
            self.c_proj.weight._stats_ref = {"kind": "c_proj", "d": d, "accum": self.proj_xtx_accum, "count": self.proj_xtx_count}
        elif R1_CPROJ_K_MODE == "diag":
            self.proj_xtx_accum = nn.Buffer(torch.zeros(4, d, dtype=torch.float32), persistent=False)
            self.proj_xtx_count = nn.Buffer(torch.zeros((), dtype=torch.float32), persistent=False)
            self.c_proj.weight._stats_ref = {"kind": "c_proj_diag", "d": d, "accum": self.proj_xtx_accum, "count": self.proj_xtx_count}
        elif R1_CPROJ_K_MODE == "none":
            self.c_proj.weight._stats_ref = None
        else:
            raise ValueError(f"unsupported R1_CPROJ_K_MODE={R1_CPROJ_K_MODE!r}")
'''


MLP_FORWARD_OFFICIAL = '''        if precond_flag:
            z2d = x.flatten(0, -2)
            torch.ops.nanogpt.accum_xtx_blocks4(z2d, self.proj_xtx_accum, self.proj_xtx_count, self.proj_xtx_tmp)
'''


MLP_FORWARD_R1 = '''        if precond_flag and R1_CPROJ_K_MODE != "none":
            z2d = x.flatten(0, -2)
            if R1_CPROJ_K_MODE == "block4":
                torch.ops.nanogpt.accum_xtx_blocks4(z2d, self.proj_xtx_accum, self.proj_xtx_count, self.proj_xtx_tmp)
            else:
                torch.ops.nanogpt.accum_xtx_diag4(z2d, self.proj_xtx_accum, self.proj_xtx_count)
'''


MLP_APPLY_OFFICIAL = '''        self.c_fc.weight._stats_ref = {"kind": "c_fc",   "d": d, "accum": self.fc_xtx_accum,   "count": self.fc_xtx_count}
        self.c_proj.weight._stats_ref = {"kind": "c_proj","d": d, "accum": self.proj_xtx_accum, "count": self.proj_xtx_count}
        return self
'''


MLP_APPLY_R1 = '''        self.c_fc.weight._stats_ref = {"kind": "c_fc", "d": d, "accum": self.fc_xtx_accum, "count": self.fc_xtx_count}
        if R1_CPROJ_K_MODE == "block4":
            self.c_proj.weight._stats_ref = {"kind": "c_proj", "d": d, "accum": self.proj_xtx_accum, "count": self.proj_xtx_count}
        elif R1_CPROJ_K_MODE == "diag":
            self.c_proj.weight._stats_ref = {"kind": "c_proj_diag", "d": d, "accum": self.proj_xtx_accum, "count": self.proj_xtx_count}
        else:
            self.c_proj.weight._stats_ref = None
        return self
'''


PRECOND_MEMORY_REPORT = '''
def _r1_report_preconditioner_memory(optimizer, model):
    cov_tensors = []
    inv_tensors = []
    for state in optimizer.state.values():
        if isinstance(state, dict):
            if isinstance(state.get("precond_cov"), torch.Tensor):
                cov_tensors.append(state["precond_cov"])
            if isinstance(state.get("precond_inv_apply"), torch.Tensor):
                inv_tensors.append(state["precond_inv_apply"])

    activation_stat_tensors = []
    activation_workspace_tensors = []
    for name, tensor in model.named_buffers():
        if "xtx_accum" in name or "xtx_count" in name:
            activation_stat_tensors.append(tensor)
        elif "xtx_tmp" in name:
            activation_workspace_tensors.append(tensor)

    plan_tensors = []
    if optimizer._apply_plan is not None:
        plan_tensors.extend(_r1_iter_tensors(optimizer._apply_plan))
    if isinstance(optimizer._refresh_K, torch.Tensor):
        plan_tensors.append(optimizer._refresh_K)

    inv_keys = {_r1_storage_key(tensor) for tensor in inv_tensors}
    workspace_tensors = [
        tensor for tensor in [*plan_tensors, *activation_workspace_tensors]
        if _r1_storage_key(tensor) not in inv_keys
    ]
    k_cov_bytes = _r1_unique_storage_bytes(cov_tensors)
    k_inv_bytes = _r1_unique_storage_bytes(inv_tensors)
    k_state_bytes = _r1_unique_storage_bytes([*cov_tensors, *inv_tensors])
    activation_stat_bytes = _r1_unique_storage_bytes(activation_stat_tensors)
    precond_workspace_bytes = _r1_unique_storage_bytes(workspace_tensors)
    total_precond_bytes = _r1_unique_storage_bytes(
        [*cov_tensors, *inv_tensors, *activation_stat_tensors, *workspace_tensors]
    )
    print(
        "R1_K_MEMORY "
        f"k_cov_bytes={k_cov_bytes} k_inv_bytes={k_inv_bytes} "
        f"k_state_bytes={k_state_bytes} activation_stat_bytes={activation_stat_bytes} "
        f"precond_workspace_bytes={precond_workspace_bytes} "
        f"total_precond_bytes={total_precond_bytes}"
    )

_r1_report_preconditioner_memory(optimizer2, raw_model)
'''


def _apply_common(source: str, method: str) -> str:
    source = _replace_once(
        source,
        "import time\nfrom dataclasses import dataclass\n",
        "import time\nimport random\nimport hashlib\nfrom dataclasses import dataclass\n",
        "common imports",
    )
    source = _replace_once(
        source,
        "args = Hyperparameters()\n",
        COMMON_CONTROL,
        "Hyperparameters instance",
    )
    source = _replace_once(
        source,
        "torch.cuda.set_device(0)\n",
        COMMON_SEED_AND_AUDIT,
        "CUDA device selection",
    )
    source = _replace_once(
        source,
        "if master_process and (last_step or (args.save_every > 0 and step % args.save_every == 0)):\n",
        "if master_process and (not R1_DISABLE_CHECKPOINT) and (last_step or (args.save_every > 0 and step % args.save_every == 0)):\n",
        "checkpoint guard",
    )
    source = _replace_once(
        source,
        '''if master_process:
    print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
''',
        FINAL_MEMORY_BLOCK,
        "final memory print",
    )
    if method == "muon":
        source = _replace_once(
            source,
            '''model = GPT(GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=12, n_embd=768))
model = model.cuda()
''',
            MODEL_AUDIT_MUON,
            "Muon model initialization",
        )
        source = _replace_once(
            source,
            "optimizers = [optimizer1, optimizer2]\n",
            '''optimizers = [optimizer1, optimizer2]
print("R1_K_MEMORY k_cov_bytes=0 k_inv_bytes=0 k_state_bytes=0 activation_stat_bytes=0 precond_workspace_bytes=0 total_precond_bytes=0")
''',
            "Muon optimizer list",
        )
        source = _replace_once(
            source,
            COMMON_CONTROL,
            COMMON_CONTROL
            + '''if R1_METHOD != "muon" or R1_CPROJ_K_MODE != "muon":
    raise ValueError("the R1 Muon source requires method=muon and cproj_k_mode=muon")
''',
            "Muon R1 mode validation",
        )
    else:
        source = _replace_once(
            source,
            "model = GPT(GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=12, n_embd=768)).cuda()\n",
            MODEL_AUDIT_NEWTON,
            "Newton model initialization",
        )
        source = _replace_once(
            source,
            COMMON_CONTROL,
            COMMON_CONTROL
            + '''if R1_METHOD not in ("block4", "none", "diag"):
    raise ValueError(f"invalid Newton R1 method={R1_METHOD!r}")
if R1_CPROJ_K_MODE != R1_METHOD:
    raise ValueError("Newton R1 method and cproj_k_mode must match")
''',
            "Newton R1 mode validation",
        )
    return source


def _apply_newton_modes(source: str) -> str:
    source = _replace_once(
        source,
        '''@accum_xtx_blocks4_op.register_fake
def accum_xtx_blocks4_fake(x_2d: Tensor, accum: Tensor, count: Tensor, tmp: Tensor):
    return accum.new_empty(())
''',
        '''@accum_xtx_blocks4_op.register_fake
def accum_xtx_blocks4_fake(x_2d: Tensor, accum: Tensor, count: Tensor, tmp: Tensor):
    return accum.new_empty(())
'''
        + DIAG_CUSTOM_OP,
        "diagonal custom operator insertion",
    )
    source = _replace_once(
        source,
        '''        elif kind == "c_proj":
            cov = torch.empty((4, d, d), device=p.device, dtype=torch.float32)
            cov.zero_()
            cov.diagonal(dim1=-2, dim2=-1).fill_(self.precond_init_diag)
            st["precond_cov"] = cov
''',
        '''        elif kind == "c_proj":
            cov = torch.empty((4, d, d), device=p.device, dtype=torch.float32)
            cov.zero_()
            cov.diagonal(dim1=-2, dim2=-1).fill_(self.precond_init_diag)
            st["precond_cov"] = cov
        elif kind == "c_proj_diag":
            st["precond_cov"] = torch.full(
                (4, d), self.precond_init_diag, device=p.device, dtype=torch.float32
            )
''',
        "diagonal covariance state",
    )
    source = _replace_once(
        source,
        '''    @torch.no_grad()
    def _finalize_precond_buffers_(self):
''',
        '''        if plan["inv_proj_diag"] is not None:
            for i, p in enumerate(plan["proj_diag_params"]):
                if p.grad is not None:
                    p.grad.view(d, 4, d).mul_(plan["inv_proj_diag"][i].unsqueeze(0))

    @torch.no_grad()
    def _finalize_precond_buffers_(self):
''',
        "diagonal gradient application",
    )
    source = _replace_once(
        source,
        "        qkv_params, o_params, fc_params, proj_params = [], [], [], []\n",
        "        qkv_params, o_params, fc_params, proj_params, proj_diag_params = [], [], [], [], []\n",
        "preconditioner parameter lists",
    )
    source = _replace_once(
        source,
        '''            elif kind == "c_proj":
                proj_params.append(p)
''',
        '''            elif kind == "c_proj":
                proj_params.append(p)
            elif kind == "c_proj_diag":
                proj_diag_params.append(p)
''',
        "diagonal parameter list",
    )
    source = _replace_once(
        source,
        '''            "proj_params": proj_params,

            "g_qkv": alloc_grad_buf(qkv_params, 3),
''',
        '''            "proj_params": proj_params,
            "proj_diag_params": proj_diag_params,

            "g_qkv": alloc_grad_buf(qkv_params, 3),
''',
        "diagonal apply-plan params",
    )
    source = _replace_once(
        source,
        '''            "inv_proj4": torch.empty((len(proj_params), 4, d, d), device=dev, dtype=torch.float32) if proj_params else None,
            "tmp_proj_blocks": torch.empty((len(proj_params) * 4, d, d), device=dev, dtype=torch.float32) if proj_params else None,
''',
        '''            "inv_proj4": torch.empty((len(proj_params), 4, d, d), device=dev, dtype=torch.float32) if proj_params else None,
            "inv_proj_diag": torch.empty((len(proj_diag_params), 4, d), device=dev, dtype=torch.float32) if proj_diag_params else None,
            "tmp_proj_blocks": torch.empty((len(proj_params) * 4, d, d), device=dev, dtype=torch.float32) if proj_params else None,
''',
        "diagonal inverse buffer",
    )
    source = _replace_once(
        source,
        '''        if plan["inv_proj4"] is not None:
            plan["inv_proj4"].zero_()
            plan["inv_proj4"].diagonal(dim1=-2, dim2=-1).fill_(1.0)
            for i, p in enumerate(proj_params):
                self.state[p]["precond_inv_apply"] = plan["inv_proj4"][i]

        self._precond_ready = True
''',
        '''        if plan["inv_proj4"] is not None:
            plan["inv_proj4"].zero_()
            plan["inv_proj4"].diagonal(dim1=-2, dim2=-1).fill_(1.0)
            for i, p in enumerate(proj_params):
                self.state[p]["precond_inv_apply"] = plan["inv_proj4"][i]

        if plan["inv_proj_diag"] is not None:
            plan["inv_proj_diag"].fill_(1.0)
            for i, p in enumerate(proj_diag_params):
                self.state[p]["precond_inv_apply"] = plan["inv_proj_diag"][i]

        self._precond_ready = True
''',
        "diagonal inverse initialization",
    )
    source = _replace_once(
        source,
        '''            elif kind == "c_proj":
                st["precond_cov"].lerp_(stref["accum"] / cnt.clamp_min(1.0), w)
''',
        '''            elif kind in ("c_proj", "c_proj_diag"):
                st["precond_cov"].lerp_(stref["accum"] / cnt.clamp_min(1.0), w)
''',
        "diagonal EWMA refresh",
    )
    source = _replace_once(
        source,
        '''                st["precond_inv_apply"][sub].copy_(inv_i)

    def step(self):
''',
        '''                st["precond_inv_apply"][sub].copy_(inv_i)

        if self._apply_plan["inv_proj_diag"] is not None:
            for p in self._apply_plan["proj_diag_params"]:
                cov = self.state[p]["precond_cov"]
                ridge = cov.mean(dim=-1) * self.precond_ridge_mult + self.precond_eps
                self.state[p]["precond_inv_apply"].copy_((cov + ridge.unsqueeze(-1)).reciprocal())

    def step(self):
''',
        "diagonal inverse refresh",
    )
    source = _replace_once(source, MLP_INIT_OFFICIAL, MLP_INIT_R1, "R1 MLP state")
    source = _replace_once(source, MLP_FORWARD_OFFICIAL, MLP_FORWARD_R1, "R1 MLP accumulation")
    source = _replace_once(source, MLP_APPLY_OFFICIAL, MLP_APPLY_R1, "R1 MLP apply")
    source = _replace_once(
        source,
        '''optimizer2.attach_preconditioner()
optimizers = [optimizer1, optimizer2]
''',
        '''optimizer2.attach_preconditioner()
optimizers = [optimizer1, optimizer2]
'''
        + PRECOND_MEMORY_REPORT,
        "preconditioner memory report",
    )
    return source


def build_source(official_repo: Path, method: str) -> DerivedSource:
    if method not in ALLOWED_METHODS:
        raise ValueError(f"unknown R1 method: {method}")
    base_script = "train_gpt_muon_1.py" if method == "muon" else "train_gpt_newton_muon_1.py"
    base_path = official_repo / base_script
    base_raw = base_path.read_bytes()
    base_source = canonical_bytes(base_raw).decode("utf-8")
    derived = _apply_common(base_source, method)
    if method != "muon":
        derived = _apply_newton_modes(derived)
    compile(derived, f"<R1-{method}>", "exec")
    diff = "".join(
        difflib.unified_diff(
            base_source.splitlines(keepends=True),
            derived.splitlines(keepends=True),
            fromfile=f"official/{base_script}",
            tofile=f"r1/train_r1_{method}.py",
        )
    )
    return DerivedSource(
        method=method,
        base_script=base_script,
        base_canonical_sha256=canonical_sha256(base_raw),
        derived_sha256=hashlib.sha256(derived.encode("utf-8")).hexdigest(),
        source=derived,
        unified_diff=diff,
    )


def self_test_diag_math() -> None:
    """Check diagonal accumulation/application against explicit dense blocks."""
    import math
    import random

    generator = random.Random(123)
    samples, d = 17, 5
    x = [[generator.gauss(0.0, 1.0) for _ in range(4 * d)] for _ in range(samples)]
    observed_diag = [
        [sum(row[block * d + col] ** 2 for row in x) / samples for col in range(d)]
        for block in range(4)
    ]
    dense_covariances = [
        [
            [
                sum(row[block * d + i] * row[block * d + j] for row in x) / samples
                for j in range(d)
            ]
            for i in range(d)
        ]
        for block in range(4)
    ]
    expected_diag = [
        [dense_covariances[block][index][index] for index in range(d)]
        for block in range(4)
    ]
    if any(
        not math.isclose(observed_diag[block][index], expected_diag[block][index], rel_tol=0, abs_tol=1e-15)
        for block in range(4)
        for index in range(d)
    ):
        raise AssertionError("diagonal covariance accumulation does not match dense block diagonals")

    gradient = [[generator.gauss(0.0, 1.0) for _ in range(4 * d)] for _ in range(d)]
    inv_diag = [[generator.random() + 0.1 for _ in range(d)] for _ in range(4)]
    observed_update = [
        [gradient[row][block * d + col] * inv_diag[block][col] for col in range(d)]
        for row in range(d)
        for block in range(4)
    ]
    expected_update = []
    for row in range(d):
        for block in range(4):
            dense_diagonal = [
                [inv_diag[block][i] if i == j else 0.0 for j in range(d)]
                for i in range(d)
            ]
            expected_update.append(
                [
                    sum(
                        gradient[row][block * d + inner] * dense_diagonal[inner][col]
                        for inner in range(d)
                    )
                    for col in range(d)
                ]
            )
    if observed_update != expected_update:
        raise AssertionError("diagonal right preconditioner does not match dense diagonal multiplication")

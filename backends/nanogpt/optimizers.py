import math
from dataclasses import dataclass

import torch


@torch.no_grad()
def matrix_sign_svd(
    x: torch.Tensor,
    *,
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Exact polar factor with an explicit SVD compute dtype.

    The returned update keeps the input dtype so it can be applied by the
    optimizer or line search without changing model precision.  Mechanism
    probes may request float64 SVD to avoid attributing FP32 orthogonality
    loss on ill-conditioned preconditioned gradients to the polar geometry.
    """
    if x.ndim != 2:
        raise ValueError("matrix_sign_svd expects a 2D tensor")
    if compute_dtype not in (torch.float32, torch.float64):
        raise ValueError(
            "matrix_sign_svd compute_dtype must be torch.float32 or "
            f"torch.float64, got {compute_dtype}"
        )
    x_compute = x.to(dtype=compute_dtype)
    u, _, vh = torch.linalg.svd(x_compute, full_matrices=False)
    return (u @ vh).to(dtype=x.dtype)


def _resolve_ns_compute_dtype(value: str | torch.dtype) -> torch.dtype:
    if value in ("float32", torch.float32):
        return torch.float32
    if value in ("bfloat16", torch.bfloat16):
        return torch.bfloat16
    raise ValueError(
        "NS compute dtype must be 'float32' or 'bfloat16'; "
        f"got {value!r}"
    )


@torch.no_grad()
def matrix_sign_ns5(
    x: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    compute_dtype: str | torch.dtype = torch.float32,
) -> torch.Tensor:
    """Newton-Schulz approximation of msgn(X) = U @ V.T."""
    if x.ndim != 2:
        raise ValueError("matrix_sign_ns5 expects a 2D tensor")

    orig_dtype = x.dtype
    y = x.to(dtype=_resolve_ns_compute_dtype(compute_dtype))
    transposed = False
    if y.shape[0] > y.shape[1]:
        y = y.T
        transposed = True

    y = y / y.norm().clamp_min(eps)

    # Coefficients commonly used by the Muon polar/Newton-Schulz update.
    a = 3.4445
    b = -4.7750
    c = 2.0315
    for _ in range(steps):
        yy_t = y @ y.T
        y = a * y + (b * yy_t + c * (yy_t @ yy_t)) @ y

    if transposed:
        y = y.T
    return y.to(dtype=orig_dtype)


@torch.no_grad()
def muon_momentum_direction(
    grad: torch.Tensor,
    momentum_buffer: torch.Tensor,
    *,
    beta: float,
    nesterov: bool,
    momentum_ema: bool,
) -> torch.Tensor:
    """Return the pre-orthogonalization Muon direction.

    ``momentum_ema=True`` reproduces the public Muon implementation's
    ``momentum.lerp_(grad, 1 - beta)`` convention.  The older local convention
    stores an unnormalized SGD-momentum sum; the two are direction-equivalent
    without Nesterov but must not be mixed when Nesterov is enabled.
    """
    if momentum_ema:
        momentum_buffer.lerp_(grad, 1.0 - beta)
        if nesterov:
            return torch.lerp(grad, momentum_buffer, beta)
        return momentum_buffer

    momentum_buffer.mul_(beta).add_(grad)
    if nesterov:
        return grad + beta * momentum_buffer
    return momentum_buffer


@torch.no_grad()
def muon_orthogonalize(
    update: torch.Tensor,
    *,
    name: str,
    ns_steps: int,
    eps: float,
    split_qkv: bool,
    adjust_lr_for_shape: bool,
    ns_compute_dtype: str | torch.dtype,
) -> torch.Tensor:
    """Apply the public Muon matrix post-processing conventions."""

    def orthogonalize_piece(piece: torch.Tensor) -> torch.Tensor:
        result = matrix_sign_ns5(
            piece,
            steps=ns_steps,
            eps=eps,
            compute_dtype=ns_compute_dtype,
        )
        if adjust_lr_for_shape:
            rows, cols = piece.shape
            result = result * math.sqrt(max(1.0, float(rows) / float(cols)))
        return result

    is_packed_qkv = (
        split_qkv
        and ".attn.c_attn.weight" in name
        and update.shape[0] == 3 * update.shape[1]
    )
    if is_packed_qkv:
        return torch.cat(
            [orthogonalize_piece(piece) for piece in update.split(update.shape[1], dim=0)],
            dim=0,
        )
    return orthogonalize_piece(update)


def matrix_cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    a = a.float()
    b = b.float()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(eps))


def fro_norm_match(update: torch.Tensor, reference: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return update * (reference.float().norm() / update.float().norm().clamp_min(eps))


def _condition_number_sym(x: torch.Tensor, eps: float = 1e-12) -> float:
    eigvals = torch.linalg.eigvalsh(0.5 * (x.float() + x.float().T))
    return float(eigvals.max() / eigvals.min().clamp_min(eps))


class PureMuon(torch.optim.Optimizer):
    """Plain Muon over 2D hidden-layer weights."""

    def __init__(
        self,
        params,
        lr=0.02,
        momentum=0.95,
        ns_steps=5,
        eps=1e-8,
        weight_decay=0.0,
        param_to_name: dict | None = None,
        cheap_muon_probe_enabled=False,
        nesterov=False,
        momentum_ema=False,
        split_qkv=False,
        adjust_lr_for_shape=False,
        ns_compute_dtype="float32",
    ):
        _resolve_ns_compute_dtype(ns_compute_dtype)
        defaults = dict(
            lr=lr,
            momentum=momentum,
            ns_steps=ns_steps,
            eps=eps,
            weight_decay=weight_decay,
            nesterov=bool(nesterov),
            momentum_ema=bool(momentum_ema),
            split_qkv=bool(split_qkv),
            adjust_lr_for_shape=bool(adjust_lr_for_shape),
            ns_compute_dtype=ns_compute_dtype,
        )
        super().__init__(params, defaults)
        self.last_stats = {}
        self.param_to_name = param_to_name or {}
        self.cheap_muon_probe_enabled = bool(cheap_muon_probe_enabled)
        self._step = 0

    @staticmethod
    def _rms(x: torch.Tensor) -> float:
        return float(torch.sqrt(torch.mean(x.float() * x.float())).item())

    @staticmethod
    def _potential_input_cov_state_bytes(p: torch.Tensor, include_eye: bool = True) -> int:
        n = int(p.shape[1])
        tensor_count = 3 if include_eye else 2
        element_size = torch.empty((), dtype=torch.float32, device=p.device).element_size()
        return n * n * element_size * tensor_count

    @torch.no_grad()
    def _update_cheap_probe_stats(self, p: torch.Tensor, st: dict, g: torch.Tensor, q: torch.Tensor) -> None:
        probe = st.setdefault(
            "cheap_muon_probe",
            {
                "count": 0,
                "grad_rms_sum": 0.0,
                "muon_update_rms_sum": 0.0,
                "grad_muon_cos_sum": 0.0,
                "grad_muon_misalignment_sum": 0.0,
                "update_instability_sum": 0.0,
                "update_instability_count": 0,
            },
        )
        cos_grad_muon = matrix_cosine(g, q)
        probe["count"] += 1
        probe["grad_rms_sum"] += self._rms(g)
        probe["muon_update_rms_sum"] += self._rms(q)
        probe["grad_muon_cos_sum"] += cos_grad_muon
        probe["grad_muon_misalignment_sum"] += max(0.0, 1.0 - cos_grad_muon)

        prev_q = st.get("cheap_muon_prev_update")
        if isinstance(prev_q, torch.Tensor):
            cos_update = matrix_cosine(prev_q, q)
            probe["update_instability_sum"] += max(0.0, 1.0 - cos_update)
            probe["update_instability_count"] += 1
        st["cheap_muon_prev_update"] = q.detach().float().clone()

    def get_cheap_muon_probe_report(self) -> list[dict]:
        rows = []
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state.get(p, {})
                probe = st.get("cheap_muon_probe")
                if not probe:
                    continue
                count = max(1, int(probe.get("count", 0)))
                instability_count = max(1, int(probe.get("update_instability_count", 0)))
                full_k_state_bytes = self._potential_input_cov_state_bytes(p, include_eye=True)
                rows.append(
                    {
                        "name": self.param_to_name.get(p, f"param_{len(rows)}"),
                        "shape": "x".join(str(dim) for dim in p.shape),
                        "rows": int(p.shape[0]),
                        "cols": int(p.shape[1]),
                        "probe_steps": int(probe.get("count", 0)),
                        "grad_rms_mean": float(probe.get("grad_rms_sum", 0.0)) / count,
                        "muon_update_rms_mean": float(probe.get("muon_update_rms_sum", 0.0)) / count,
                        "grad_muon_cos_mean": float(probe.get("grad_muon_cos_sum", 0.0)) / count,
                        "grad_muon_misalignment_mean": (
                            float(probe.get("grad_muon_misalignment_sum", 0.0)) / count
                        ),
                        "update_instability_mean": (
                            float(probe.get("update_instability_sum", 0.0)) / instability_count
                        ),
                        "update_instability_count": int(probe.get("update_instability_count", 0)),
                        "k_state_full_bytes": int(full_k_state_bytes),
                    }
                )
        return rows

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        active = 0
        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError("PureMuon only supports 2D matrix parameters")
                st = self.state[p]
                if "momentum" not in st:
                    st["momentum"] = torch.zeros_like(p, dtype=torch.float32)

                g = p.grad.detach().float()
                buf = st["momentum"]
                update = muon_momentum_direction(
                    g,
                    buf,
                    beta=mu,
                    nesterov=group["nesterov"],
                    momentum_ema=group["momentum_ema"],
                )
                q = muon_orthogonalize(
                    update,
                    name=self.param_to_name.get(p, ""),
                    ns_steps=ns_steps,
                    eps=eps,
                    split_qkv=group["split_qkv"],
                    adjust_lr_for_shape=group["adjust_lr_for_shape"],
                    ns_compute_dtype=group["ns_compute_dtype"],
                )
                if self.cheap_muon_probe_enabled:
                    self._update_cheap_probe_stats(p, st, g, q)
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(q.to(dtype=p.dtype), alpha=-lr)
                active += 1

        self._step += 1
        probe_count = 0
        probe_grad_cos_sum = 0.0
        probe_misalignment_sum = 0.0
        probe_instability_sum = 0.0
        if self.cheap_muon_probe_enabled:
            for group in self.param_groups:
                for p in group["params"]:
                    probe = self.state.get(p, {}).get("cheap_muon_probe")
                    if not probe or probe.get("count", 0) <= 0:
                        continue
                    count = max(1, int(probe["count"]))
                    instability_count = max(1, int(probe.get("update_instability_count", 0)))
                    probe_count += 1
                    probe_grad_cos_sum += float(probe.get("grad_muon_cos_sum", 0.0)) / count
                    probe_misalignment_sum += float(probe.get("grad_muon_misalignment_sum", 0.0)) / count
                    probe_instability_sum += float(probe.get("update_instability_sum", 0.0)) / instability_count
        self.last_stats = {
            "active_params": active,
            "k_state_bytes": 0,
            "k_matrix_bytes": 0,
            "k_state_params": 0,
            "k_state_released_params": 0,
            "cheap_muon_probe_params": probe_count,
            "cheap_muon_probe_grad_muon_cos_mean": probe_grad_cos_sum / max(1, probe_count),
            "cheap_muon_probe_misalignment_mean": probe_misalignment_sum / max(1, probe_count),
            "cheap_muon_probe_update_instability_mean": probe_instability_sum / max(1, probe_count),
            "step": self._step,
        }
        return loss


class InputCovState:
    """Running input second moment K ~= E[z z.T] and its damped inverse."""

    def __init__(
        self,
        n: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        beta: float = 0.95,
        ridge: float = 0.2,
        refresh_interval: int = 32,
        max_samples: int | None = None,
        init_scale: float = 1.0,
        init_inverse_scale: float | None = None,
        first_refresh_step: int = 0,
        offdiag_alpha: float = 1.0,
    ):
        if refresh_interval <= 0:
            raise ValueError("input refresh interval must be positive")
        if first_refresh_step < 0:
            raise ValueError("first input refresh step must be non-negative")
        if init_scale <= 0:
            raise ValueError("input covariance init scale must be positive")
        if init_inverse_scale is not None and init_inverse_scale <= 0:
            raise ValueError("input covariance initial inverse scale must be positive")
        if not 0.0 <= offdiag_alpha <= 1.0:
            raise ValueError("offdiag_alpha must be in [0, 1]")
        self.n = n
        self.device = device
        self.dtype = dtype
        self.init_scale = init_scale
        self.beta = beta
        self.ridge = ridge
        self.refresh_interval = refresh_interval
        self.max_samples = max_samples
        self.first_refresh_step = int(first_refresh_step)
        self.offdiag_alpha = float(offdiag_alpha)
        self.init_inverse_scale = init_inverse_scale

        self.eye = None
        self.K = None
        self.K_inv = None
        self._materialize(init_scale=init_scale)
        self.num_updates = 0
        self.last_cond = 1.0

    @torch.no_grad()
    def _materialize(self, init_scale: float | None = None) -> None:
        if init_scale is None:
            init_scale = self.init_scale
        eye = torch.eye(self.n, device=self.device, dtype=self.dtype)
        self.eye = eye
        self.K = init_scale * eye.clone()
        inverse_scale = (
            1.0 / init_scale
            if self.init_inverse_scale is None
            else float(self.init_inverse_scale)
        )
        self.K_inv = inverse_scale * eye.clone()

    @torch.no_grad()
    def release(self) -> None:
        self.eye = None
        self.K = None
        self.K_inv = None

    def is_released(self) -> bool:
        return self.K is None or self.K_inv is None or self.eye is None

    def state_bytes(self, include_eye: bool = True) -> int:
        tensors = [self.K, self.K_inv]
        if include_eye:
            tensors.append(self.eye)
        return sum(t.numel() * t.element_size() for t in tensors if isinstance(t, torch.Tensor))

    def full_state_bytes(self, include_eye: bool = True) -> int:
        tensor_count = 3 if include_eye else 2
        element_size = torch.empty((), dtype=self.dtype, device=self.device).element_size()
        return self.n * self.n * element_size * tensor_count

    @torch.no_grad()
    def maybe_refresh(self, x: torch.Tensor, step: int, diagnostics: bool = False) -> None:
        if self.is_released():
            self._materialize()
        if (
            step < self.first_refresh_step
            or (step - self.first_refresh_step) % self.refresh_interval != 0
        ):
            return
        if x.shape[-1] != self.n:
            raise ValueError(f"tracked input has last dim {x.shape[-1]}, expected {self.n}")

        x2d = x.detach().reshape(-1, self.n).float()
        if self.max_samples is not None and x2d.shape[0] > self.max_samples:
            idx = torch.randperm(x2d.shape[0], device=x2d.device)[: self.max_samples]
            x2d = x2d[idx]

        k_batch = (x2d.T @ x2d) / max(1, x2d.shape[0])
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(k_batch, op=torch.distributed.ReduceOp.SUM)
            k_batch.div_(torch.distributed.get_world_size())

        self.K.mul_(self.beta).add_(k_batch, alpha=1.0 - self.beta)
        avg_diag = (torch.trace(self.K) / self.n).clamp_min(1e-12)
        ridge_value = self.ridge * avg_diag
        if self.offdiag_alpha == 1.0:
            k_damped = self.K + ridge_value * self.eye
        else:
            # Preserve the running diagonal exactly and scale only cross-coordinate
            # covariance. alpha=0 is the dense implementation of diag(K), while
            # alpha=1 is the original full covariance.
            k_damped = self.K * self.offdiag_alpha
            k_damped.diagonal().add_(
                self.K.diagonal(), alpha=1.0 - self.offdiag_alpha
            )
            k_damped.diagonal().add_(ridge_value)

        chol, info = torch.linalg.cholesky_ex(k_damped)
        if int(info.max().item()) != 0:
            if self.offdiag_alpha == 1.0:
                k_damped = self.K + (10.0 * ridge_value + 1e-6) * self.eye
            else:
                k_damped = self.K * self.offdiag_alpha
                k_damped.diagonal().add_(
                    self.K.diagonal(), alpha=1.0 - self.offdiag_alpha
                )
                k_damped.diagonal().add_(10.0 * ridge_value + 1e-6)
            chol = torch.linalg.cholesky(k_damped)
        self.K_inv = torch.cholesky_inverse(chol)
        self.num_updates += 1
        if diagnostics:
            self.last_cond = _condition_number_sym(k_damped)


class DiagInputCovState:
    """Diagonal running input second moment used as a c_proj mechanism control."""

    def __init__(
        self,
        n: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        beta: float = 0.95,
        ridge: float = 0.2,
        refresh_interval: int = 32,
        max_samples: int | None = None,
        init_scale: float = 1.0,
        init_inverse_scale: float | None = None,
        first_refresh_step: int = 0,
    ):
        if refresh_interval <= 0:
            raise ValueError("input refresh interval must be positive")
        if first_refresh_step < 0:
            raise ValueError("first input refresh step must be non-negative")
        self.n = n
        self.device = device
        self.dtype = dtype
        self.beta = beta
        self.ridge = ridge
        self.refresh_interval = refresh_interval
        self.max_samples = max_samples
        self.first_refresh_step = int(first_refresh_step)
        self.diag = torch.full((n,), init_scale, device=device, dtype=dtype)
        inverse_scale = (
            1.0 / init_scale
            if init_inverse_scale is None
            else float(init_inverse_scale)
        )
        self.diag_inv = torch.full((n,), inverse_scale, device=device, dtype=dtype)
        self.num_updates = 0
        self.last_cond = 1.0

    def state_bytes(self, include_eye: bool = True) -> int:
        del include_eye
        return sum(t.numel() * t.element_size() for t in (self.diag, self.diag_inv))

    def matrix_state_bytes(self) -> int:
        return self.state_bytes()

    @torch.no_grad()
    def maybe_refresh(self, x: torch.Tensor, step: int, diagnostics: bool = False) -> None:
        if (
            step < self.first_refresh_step
            or (step - self.first_refresh_step) % self.refresh_interval != 0
        ):
            return
        if x.shape[-1] != self.n:
            raise ValueError(f"tracked input has last dim {x.shape[-1]}, expected {self.n}")

        x2d = x.detach().reshape(-1, self.n).float()
        if self.max_samples is not None and x2d.shape[0] > self.max_samples:
            idx = torch.randperm(x2d.shape[0], device=x2d.device)[: self.max_samples]
            x2d = x2d[idx]

        diag_batch = torch.mean(x2d * x2d, dim=0)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(diag_batch, op=torch.distributed.ReduceOp.SUM)
            diag_batch.div_(torch.distributed.get_world_size())

        self.diag.mul_(self.beta).add_(diag_batch, alpha=1.0 - self.beta)
        avg_diag = self.diag.mean().clamp_min(1e-12)
        damped = self.diag + self.ridge * avg_diag
        self.diag_inv.copy_(damped.reciprocal())
        self.num_updates += 1
        if diagnostics:
            self.last_cond = float((damped.max() / damped.min().clamp_min(1e-12)).item())

    @torch.no_grad()
    def apply_right(self, g: torch.Tensor) -> torch.Tensor:
        return g.float() * self.diag_inv.float().unsqueeze(0)


class ScalarInputCovState:
    """Running mean diagonal second moment used as a scalar c_proj control.

    This retains only mean(diag(K)) and therefore removes both coordinate-wise
    heterogeneity and cross-coordinate covariance. The scalar is applied before
    momentum, so its temporal variation can still change the weighting of past
    gradients even though the matrix-sign map is scale invariant at one step.
    """

    def __init__(
        self,
        n: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        beta: float = 0.95,
        ridge: float = 0.2,
        refresh_interval: int = 32,
        max_samples: int | None = None,
        init_scale: float = 1.0,
        init_inverse_scale: float | None = None,
        first_refresh_step: int = 0,
    ):
        if refresh_interval <= 0:
            raise ValueError("input refresh interval must be positive")
        if first_refresh_step < 0:
            raise ValueError("first input refresh step must be non-negative")
        self.n = n
        self.device = device
        self.dtype = dtype
        self.beta = beta
        self.ridge = ridge
        self.refresh_interval = refresh_interval
        self.max_samples = max_samples
        self.first_refresh_step = int(first_refresh_step)
        self.scale = torch.full((), init_scale, device=device, dtype=dtype)
        inverse_scale = (
            1.0 / init_scale
            if init_inverse_scale is None
            else float(init_inverse_scale)
        )
        self.scale_inv = torch.full((), inverse_scale, device=device, dtype=dtype)
        self.num_updates = 0
        self.last_cond = 1.0

    def state_bytes(self, include_eye: bool = True) -> int:
        del include_eye
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.scale, self.scale_inv)
        )

    def matrix_state_bytes(self) -> int:
        return self.state_bytes()

    @torch.no_grad()
    def maybe_refresh(self, x: torch.Tensor, step: int, diagnostics: bool = False) -> None:
        if (
            step < self.first_refresh_step
            or (step - self.first_refresh_step) % self.refresh_interval != 0
        ):
            return
        if x.shape[-1] != self.n:
            raise ValueError(f"tracked input has last dim {x.shape[-1]}, expected {self.n}")

        x2d = x.detach().reshape(-1, self.n).float()
        if self.max_samples is not None and x2d.shape[0] > self.max_samples:
            idx = torch.randperm(x2d.shape[0], device=x2d.device)[: self.max_samples]
            x2d = x2d[idx]

        scale_batch = torch.mean(x2d * x2d)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(scale_batch, op=torch.distributed.ReduceOp.SUM)
            scale_batch.div_(torch.distributed.get_world_size())

        self.scale.mul_(self.beta).add_(scale_batch, alpha=1.0 - self.beta)
        damped = self.scale * (1.0 + self.ridge)
        self.scale_inv.copy_(damped.clamp_min(1e-12).reciprocal())
        self.num_updates += 1
        if diagnostics:
            self.last_cond = 1.0

    @torch.no_grad()
    def apply_right(self, g: torch.Tensor) -> torch.Tensor:
        return g.float() * self.scale_inv.float()


class BlockDiagInputCovState:
    """Block-diagonal input second moment with shared activation sampling across blocks.

    ``offdiag_alpha`` scales only the off-diagonal entries inside each block.
    Cross-block entries remain structurally zero for every alpha.  The damping
    scale remains block-local so alpha=1 exactly reproduces the existing
    ``block4`` implementation; consequently alpha=0 is a block-path control,
    not an exact alias of the globally damped efficient diagonal state.
    """

    def __init__(
        self,
        n: int,
        blocks: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        beta: float = 0.95,
        ridge: float = 0.2,
        refresh_interval: int = 32,
        max_samples: int | None = None,
        init_scale: float = 1.0,
        init_inverse_scale: float | None = None,
        first_refresh_step: int = 0,
        offdiag_alpha: float = 1.0,
    ):
        if blocks <= 0 or n % blocks != 0:
            raise ValueError(f"input dimension {n} must be divisible by blocks={blocks}")
        if not 0.0 <= offdiag_alpha <= 1.0:
            raise ValueError("offdiag_alpha must be in [0, 1]")
        self.n = n
        self.blocks = blocks
        self.block_size = n // blocks
        self.offdiag_alpha = float(offdiag_alpha)
        self.refresh_interval = refresh_interval
        self.max_samples = max_samples
        self.first_refresh_step = int(first_refresh_step)
        self.states = [
            InputCovState(
                n=self.block_size,
                device=device,
                dtype=dtype,
                beta=beta,
                ridge=ridge,
                refresh_interval=refresh_interval,
                max_samples=max_samples,
                init_scale=init_scale,
                init_inverse_scale=init_inverse_scale,
                first_refresh_step=first_refresh_step,
                offdiag_alpha=self.offdiag_alpha,
            )
            for _ in range(blocks)
        ]
        self.last_cond = 1.0
        self.last_cond_mean = 1.0

    def state_bytes(self, include_eye: bool = True) -> int:
        return sum(state.state_bytes(include_eye=include_eye) for state in self.states)

    def matrix_state_bytes(self) -> int:
        return self.state_bytes(include_eye=False)

    @torch.no_grad()
    def maybe_refresh(self, x: torch.Tensor, step: int, diagnostics: bool = False) -> None:
        if (
            step < self.first_refresh_step
            or (step - self.first_refresh_step) % self.refresh_interval != 0
        ):
            return
        if x.shape[-1] != self.n:
            raise ValueError(f"tracked input has last dim {x.shape[-1]}, expected {self.n}")

        x2d = x.detach().reshape(-1, self.n)
        if self.max_samples is not None and x2d.shape[0] > self.max_samples:
            idx = torch.randperm(x2d.shape[0], device=x2d.device)[: self.max_samples]
            x2d = x2d[idx]

        for block_idx, state in enumerate(self.states):
            start = block_idx * self.block_size
            end = start + self.block_size
            state.maybe_refresh(x2d[:, start:end], step, diagnostics)

        if diagnostics:
            conds = [state.last_cond for state in self.states]
            self.last_cond = max(conds)
            self.last_cond_mean = sum(conds) / len(conds)

    @torch.no_grad()
    def apply_right(self, g: torch.Tensor) -> torch.Tensor:
        pieces = []
        for block_idx, state in enumerate(self.states):
            start = block_idx * self.block_size
            end = start + self.block_size
            pieces.append(g[:, start:end].float() @ state.K_inv.float())
        return torch.cat(pieces, dim=1)


class DiagSigmaState:
    """Diagonal shrinkage proxy for Sigma_W, used as a negative/control baseline."""

    def __init__(
        self,
        m: int,
        n: int,
        device: torch.device,
        beta: float = 0.95,
        lambda_max: float = 0.25,
        lambda_start_step: int = 200,
        lambda_warmup_steps: int = 1000,
        eps: float = 1e-4,
    ):
        self.m = m
        self.n = n
        self.beta = beta
        self.lambda_max = lambda_max
        self.lambda_start_step = lambda_start_step
        self.lambda_warmup_steps = max(1, lambda_warmup_steps)
        self.eps = eps
        self.diag = torch.ones(m, device=device, dtype=torch.float32)
        self.last_stats = {}

    def lambda_at(self, step: int) -> float:
        if step < self.lambda_start_step:
            return 0.0
        progress = min(1.0, float(step - self.lambda_start_step) / float(self.lambda_warmup_steps))
        return self.lambda_max * progress

    @torch.no_grad()
    def update(self, u: torch.Tensor) -> None:
        row_energy = (u.detach().float() * u.detach().float()).mean(dim=1)
        self.diag.mul_(self.beta).add_(row_energy, alpha=1.0 - self.beta)

    @torch.no_grad()
    def scale(self, step: int) -> torch.Tensor:
        lam = self.lambda_at(step)
        alpha = self.diag.mean().clamp_min(1e-12)
        shrunk = (1.0 - lam) * alpha + lam * self.diag + self.eps * alpha
        shrunk = shrunk.clamp_min(1e-12)
        s = torch.sqrt(shrunk)
        self.last_stats = {
            "sigma_lambda": lam,
            "sigma_anisotropy_max": float(shrunk.max() / shrunk.min().clamp_min(1e-12)),
            "sigma_anisotropy_mean": float(shrunk.max() / shrunk.min().clamp_min(1e-12)),
            "sigma_offdiag_mean": 0.0,
        }
        return s


@dataclass
class BlockSigmaStats:
    sigma_lambda: float = 0.0
    sigma_anisotropy_mean: float = 1.0
    sigma_anisotropy_max: float = 1.0
    sigma_offdiag_mean: float = 0.0
    num_blocks: int = 0


class BlockSigmaState:
    """Block covariance + global identity shrinkage estimator for Sigma_W."""

    def __init__(
        self,
        m: int,
        n: int,
        device: torch.device,
        block_size: int = 64,
        beta: float = 0.95,
        lambda_max: float = 0.25,
        lambda_start_step: int = 200,
        lambda_warmup_steps: int = 1000,
        eps: float = 1e-4,
        refresh_interval: int = 16,
    ):
        if block_size <= 0:
            raise ValueError("sigma block size must be positive")
        if refresh_interval <= 0:
            raise ValueError("sigma refresh interval must be positive")
        self.m = m
        self.n = n
        self.block_size = block_size
        self.beta = beta
        self.lambda_max = lambda_max
        self.lambda_start_step = lambda_start_step
        self.lambda_warmup_steps = max(1, lambda_warmup_steps)
        self.eps = eps
        self.refresh_interval = refresh_interval
        self.ranges = [(s, min(s + block_size, m)) for s in range(0, m, block_size)]
        self.covs = []
        self.factors = []
        for start, end in self.ranges:
            width = end - start
            eye = torch.eye(width, device=device, dtype=torch.float32)
            self.covs.append(eye.clone())
            self.factors.append(eye.clone())
        self.last_refresh_step = -1
        self.last_stats = BlockSigmaStats(num_blocks=len(self.ranges))

    def lambda_at(self, step: int) -> float:
        if step < self.lambda_start_step:
            return 0.0
        progress = min(1.0, float(step - self.lambda_start_step) / float(self.lambda_warmup_steps))
        return self.lambda_max * progress

    @torch.no_grad()
    def update_covariance(self, u: torch.Tensor) -> None:
        u = u.detach().float()
        for idx, (start, end) in enumerate(self.ranges):
            ub = u[start:end, :]
            cov_batch = (ub @ ub.T) / max(1, self.n)
            cov_batch = 0.5 * (cov_batch + cov_batch.T)
            self.covs[idx].mul_(self.beta).add_(cov_batch, alpha=1.0 - self.beta)

    @torch.no_grad()
    def refresh_factors(self, step: int) -> None:
        if self.last_refresh_step >= 0 and step % self.refresh_interval != 0:
            return

        lam = self.lambda_at(step)
        total_trace = sum(torch.trace(cov) for cov in self.covs)
        alpha = (total_trace / self.m).clamp_min(1e-12)

        anisotropy_sum = 0.0
        anisotropy_max = 1.0
        offdiag_sum = 0.0
        for idx, cov in enumerate(self.covs):
            width = cov.shape[0]
            eye = torch.eye(width, device=cov.device, dtype=torch.float32)
            cov = 0.5 * (cov + cov.T)
            shrunk = (1.0 - lam) * alpha * eye + lam * cov + self.eps * alpha * eye
            shrunk = 0.5 * (shrunk + shrunk.T)

            chol, info = torch.linalg.cholesky_ex(shrunk)
            if int(info.max().item()) != 0:
                jitter = max(1e-6, self.eps) * alpha
                chol = torch.linalg.cholesky(shrunk + 10.0 * jitter * eye)
            self.factors[idx] = chol

            eig = torch.linalg.eigvalsh(shrunk)
            ratio = float(eig.max() / eig.min().clamp_min(1e-12))
            offdiag = cov - torch.diag_embed(torch.diagonal(cov))
            offdiag_ratio = float(offdiag.norm() / cov.norm().clamp_min(1e-12))
            anisotropy_sum += ratio
            anisotropy_max = max(anisotropy_max, ratio)
            offdiag_sum += offdiag_ratio

        n_blocks = len(self.covs)
        self.last_refresh_step = step
        self.last_stats = BlockSigmaStats(
            sigma_lambda=lam,
            sigma_anisotropy_mean=anisotropy_sum / max(1, n_blocks),
            sigma_anisotropy_max=anisotropy_max,
            sigma_offdiag_mean=offdiag_sum / max(1, n_blocks),
            num_blocks=n_blocks,
        )

    @torch.no_grad()
    def left_factor_T(self, x: torch.Tensor) -> torch.Tensor:
        pieces = []
        x = x.float()
        for factor, (start, end) in zip(self.factors, self.ranges):
            pieces.append(factor.T @ x[start:end, :])
        return torch.cat(pieces, dim=0)

    @torch.no_grad()
    def left_factor(self, x: torch.Tensor) -> torch.Tensor:
        pieces = []
        x = x.float()
        for factor, (start, end) in zip(self.factors, self.ranges):
            pieces.append(factor @ x[start:end, :])
        return torch.cat(pieces, dim=0)


class _InputTrackedMatrixOptimizer(torch.optim.Optimizer):
    """Shared state plumbing for Newton-Muon variants that need Linear inputs."""

    variant_name = "input_tracked"

    def __init__(self, params, param_to_module: dict, defaults: dict):
        super().__init__(params, defaults)
        self.param_to_module = param_to_module
        self.last_stats = {}
        self._step = 0

    def set_input_tracking_enabled(self, enabled):
        previous = getattr(self, "collect_input_stats", True)
        self.collect_input_stats = enabled
        return previous

    @torch.no_grad()
    def _init_common_state(self, p: torch.Tensor, group: dict):
        st = self.state[p]
        if "momentum" in st:
            return st

        m, n = p.shape
        st["momentum"] = torch.zeros_like(p, dtype=torch.float32)
        st["input_cov"] = InputCovState(
            n=n,
            device=p.device,
            beta=group["input_beta"],
            ridge=group["input_ridge"],
            refresh_interval=group["input_refresh"],
            max_samples=group["input_max_samples"],
            first_refresh_step=group.get("input_first_refresh_step", 0),
            init_scale=group.get("input_init_scale", 1.0),
            init_inverse_scale=group.get("input_init_inverse_scale"),
        )
        return st

    def _tracked_input_for(self, p: torch.Tensor) -> torch.Tensor:
        module = self.param_to_module.get(p)
        if module is None or not hasattr(module, "_last_input"):
            raise RuntimeError("Missing tracked input activation for a matrix parameter")
        return module._last_input


class NewtonMuon(_InputTrackedMatrixOptimizer):
    """Input-side Newton-Muon: Q = msgn(momentum(G K^{-1}))."""

    variant_name = "newton_muon"

    def __init__(
        self,
        params,
        param_to_module: dict,
        lr=0.02,
        momentum=0.95,
        weight_decay=0.0,
        ns_steps=5,
        eps=1e-8,
        input_beta=0.95,
        input_ridge=0.2,
        input_refresh=32,
        input_max_samples=None,
        input_first_refresh_step=0,
        input_init_scale=1.0,
        input_init_inverse_scale=None,
        diagnostic_interval=0,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            eps=eps,
            input_beta=input_beta,
            input_ridge=input_ridge,
            input_refresh=input_refresh,
            input_max_samples=input_max_samples,
            input_first_refresh_step=int(input_first_refresh_step),
            input_init_scale=float(input_init_scale),
            input_init_inverse_scale=input_init_inverse_scale,
            diagnostic_interval=diagnostic_interval,
        )
        super().__init__(params, param_to_module, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        active = 0
        k_cond_sum = 0.0
        k_cond_max = 0.0
        k_state_bytes = 0
        k_matrix_bytes = 0
        k_state_params = 0
        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            wd = group["weight_decay"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]
            diag_interval = group["diagnostic_interval"]
            diagnostics = diag_interval > 0 and self._step % diag_interval == 0

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError("NewtonMuon only supports 2D matrix parameters")

                st = self._init_common_state(p, group)
                input_cov: InputCovState = st["input_cov"]
                input_cov.maybe_refresh(self._tracked_input_for(p), self._step, diagnostics)
                k_state_bytes += input_cov.state_bytes(include_eye=True)
                k_matrix_bytes += input_cov.state_bytes(include_eye=False)
                k_state_params += 1

                r = p.grad.detach().float() @ input_cov.K_inv.float()
                buf = st["momentum"]
                buf.mul_(mu).add_(r)
                q = matrix_sign_ns5(buf, steps=ns_steps, eps=eps)

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(q.to(dtype=p.dtype), alpha=-lr)

                active += 1
                k_cond_sum += input_cov.last_cond
                k_cond_max = max(k_cond_max, input_cov.last_cond)

        self._step += 1
        self.last_stats = {
            "active_params": active,
            "input_k_cond_mean": k_cond_sum / active if active else 0.0,
            "input_k_cond_max": k_cond_max if active else 0.0,
            "k_state_bytes": k_state_bytes,
            "k_matrix_bytes": k_matrix_bytes,
            "k_state_params": k_state_params,
            "k_state_released_params": 0,
            "step": self._step,
        }
        return loss


class CProjKModeNewtonMuon(_InputTrackedMatrixOptimizer):
    """Keep full Newton-Muon elsewhere while varying only the mlp.c_proj K structure."""

    variant_name = "cproj_k_mode_newton_muon"
    valid_cproj_modes = (
        "none",
        "full",
        "block4",
        "diag",
        "scalar",
        "alpha",
        "block_alpha",
    )

    def __init__(
        self,
        params,
        param_to_module: dict,
        param_to_name: dict,
        lr=0.02,
        momentum=0.95,
        weight_decay=0.0,
        ns_steps=5,
        eps=1e-8,
        input_beta=0.95,
        input_ridge=0.2,
        input_refresh=32,
        input_max_samples=None,
        input_first_refresh_step=0,
        input_init_scale=1.0,
        input_init_inverse_scale=None,
        diagnostic_interval=0,
        cproj_k_mode="block4",
        cproj_k_layers=(),
        cproj_k_blocks=4,
        cproj_k_reference_mode="full",
        cproj_k_offdiag_alpha=0.5,
        cproj_shadow_k_modes=(),
        cproj_shadow_k_layers=(),
        nesterov=False,
        momentum_ema=False,
        split_qkv=False,
        adjust_lr_for_shape=False,
        ns_compute_dtype="float32",
    ):
        _resolve_ns_compute_dtype(ns_compute_dtype)
        if cproj_k_mode not in self.valid_cproj_modes:
            raise ValueError(
                f"Unknown cproj_k_mode={cproj_k_mode!r}; expected one of {self.valid_cproj_modes}"
            )
        if cproj_k_blocks <= 0:
            raise ValueError("cproj_k_blocks must be positive")
        if cproj_k_reference_mode not in ("full", "block4"):
            raise ValueError(
                "cproj_k_reference_mode must be 'full' or 'block4'; "
                f"got {cproj_k_reference_mode!r}"
            )
        if (
            cproj_k_mode in ("alpha", "block_alpha")
            and not 0.0 <= cproj_k_offdiag_alpha <= 1.0
        ):
            raise ValueError("cproj_k_offdiag_alpha must be in [0, 1]")
        target_layers = tuple(int(layer) for layer in cproj_k_layers)
        if len(target_layers) != len(set(target_layers)):
            raise ValueError(f"duplicate c_proj K target layers: {target_layers}")
        if any(layer < 0 for layer in target_layers):
            raise ValueError(
                f"c_proj K target layers must be non-negative: {target_layers}"
            )
        shadow_modes = tuple(str(mode) for mode in cproj_shadow_k_modes)
        if len(shadow_modes) != len(set(shadow_modes)):
            raise ValueError(f"duplicate c_proj shadow K modes: {shadow_modes}")
        invalid_shadow_modes = [
            mode
            for mode in shadow_modes
            if mode not in self.valid_cproj_modes or mode == "none"
        ]
        if invalid_shadow_modes:
            raise ValueError(
                "c_proj shadow K modes must be non-none modes from "
                f"{self.valid_cproj_modes}; got {invalid_shadow_modes}"
            )
        shadow_layers = tuple(int(layer) for layer in cproj_shadow_k_layers)
        if len(shadow_layers) != len(set(shadow_layers)):
            raise ValueError(f"duplicate c_proj shadow K layers: {shadow_layers}")
        if any(layer < 0 for layer in shadow_layers):
            raise ValueError(
                f"c_proj shadow K layers must be non-negative: {shadow_layers}"
            )
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            eps=eps,
            input_beta=input_beta,
            input_ridge=input_ridge,
            input_refresh=input_refresh,
            input_max_samples=input_max_samples,
            input_first_refresh_step=int(input_first_refresh_step),
            input_init_scale=float(input_init_scale),
            input_init_inverse_scale=input_init_inverse_scale,
            diagnostic_interval=diagnostic_interval,
            cproj_k_mode=cproj_k_mode,
            cproj_k_layers=target_layers,
            cproj_k_blocks=int(cproj_k_blocks),
            cproj_k_reference_mode=cproj_k_reference_mode,
            cproj_k_offdiag_alpha=float(cproj_k_offdiag_alpha),
            cproj_shadow_k_modes=shadow_modes,
            cproj_shadow_k_layers=shadow_layers,
            nesterov=bool(nesterov),
            momentum_ema=bool(momentum_ema),
            split_qkv=bool(split_qkv),
            adjust_lr_for_shape=bool(adjust_lr_for_shape),
            ns_compute_dtype=ns_compute_dtype,
        )
        super().__init__(params, param_to_module, defaults)
        self.param_to_name = param_to_name

    def _is_cproj(self, p: torch.Tensor) -> bool:
        return ".mlp.c_proj.weight" in self.param_to_name.get(p, "")

    def _cproj_layer_index(self, p: torch.Tensor) -> int:
        name = self.param_to_name.get(p, "")
        parts = name.split(".")
        if len(parts) < 3 or parts[0:2] != ["transformer", "h"]:
            raise ValueError(f"cannot recover transformer layer from {name!r}")
        return int(parts[2])

    def _mode_for_param(self, p: torch.Tensor, group: dict) -> str:
        if not self._is_cproj(p):
            return "full"
        target_layers = tuple(group["cproj_k_layers"])
        if target_layers and self._cproj_layer_index(p) not in target_layers:
            return "full"
        return str(group["cproj_k_mode"])

    def _shadow_enabled_for_param(
        self,
        p: torch.Tensor,
        group: dict,
    ) -> bool:
        if not self._is_cproj(p):
            return False
        layers = tuple(group["cproj_shadow_k_layers"])
        return not layers or self._cproj_layer_index(p) in layers

    @staticmethod
    def _full_k_state_bytes(p: torch.Tensor) -> int:
        n = int(p.shape[1])
        element_size = torch.empty((), device=p.device, dtype=torch.float32).element_size()
        return n * n * element_size * 3

    @staticmethod
    def _block_k_state_bytes(p: torch.Tensor, blocks: int) -> int:
        n = int(p.shape[1])
        if blocks <= 0 or n % blocks != 0:
            raise ValueError(f"input dimension {n} must be divisible by blocks={blocks}")
        block_size = n // blocks
        element_size = torch.empty((), device=p.device, dtype=torch.float32).element_size()
        return blocks * block_size * block_size * element_size * 3

    @classmethod
    def _reference_k_state_bytes(
        cls,
        p: torch.Tensor,
        *,
        is_cproj: bool,
        group: dict,
    ) -> int:
        if is_cproj and group["cproj_k_reference_mode"] == "block4":
            return cls._block_k_state_bytes(p, group["cproj_k_blocks"])
        return cls._full_k_state_bytes(p)

    @staticmethod
    def _new_input_cov_state(
        mode: str,
        *,
        n: int,
        device: torch.device,
        group: dict,
    ):
        common = dict(
            n=n,
            device=device,
            beta=group["input_beta"],
            ridge=group["input_ridge"],
            refresh_interval=group["input_refresh"],
            max_samples=group["input_max_samples"],
            first_refresh_step=group["input_first_refresh_step"],
            init_scale=group["input_init_scale"],
            init_inverse_scale=group["input_init_inverse_scale"],
        )
        if mode == "none":
            return None
        if mode == "full":
            return InputCovState(**common)
        if mode == "block4":
            return BlockDiagInputCovState(
                **common,
                blocks=group["cproj_k_blocks"],
            )
        if mode == "diag":
            return DiagInputCovState(**common)
        if mode == "scalar":
            return ScalarInputCovState(**common)
        if mode == "alpha":
            return InputCovState(
                **common,
                offdiag_alpha=group["cproj_k_offdiag_alpha"],
            )
        if mode == "block_alpha":
            return BlockDiagInputCovState(
                **common,
                blocks=group["cproj_k_blocks"],
                offdiag_alpha=group["cproj_k_offdiag_alpha"],
            )
        raise RuntimeError(f"unhandled K mode: {mode}")

    @torch.no_grad()
    def _init_mode_state(self, p: torch.Tensor, group: dict) -> dict:
        st = self.state[p]
        _, n = p.shape
        is_cproj = self._is_cproj(p)
        if "momentum" not in st:
            mode = self._mode_for_param(p, group)
            st["momentum"] = torch.zeros_like(p, dtype=torch.float32)
            st["k_mode"] = mode
            st["input_cov"] = self._new_input_cov_state(
                mode,
                n=n,
                device=p.device,
                group=group,
            )

        if (
            is_cproj
            and self._shadow_enabled_for_param(p, group)
            and "shadow_input_cov" not in st
        ):
            st["shadow_input_cov"] = {
                mode: self._new_input_cov_state(
                    mode,
                    n=n,
                    device=p.device,
                    group=group,
                )
                for mode in group["cproj_shadow_k_modes"]
            }
            st["shadow_momentum"] = {
                mode: torch.zeros_like(p, dtype=torch.float32)
                for mode in group["cproj_shadow_k_modes"]
            }
        elif is_cproj and "shadow_input_cov" not in st:
            st["shadow_input_cov"] = {}
            st["shadow_momentum"] = {}
        return st

    def _group_for_param(self, param: torch.Tensor) -> dict:
        for group in self.param_groups:
            if any(candidate is param for candidate in group["params"]):
                return group
        raise KeyError("parameter is not owned by this optimizer")

    @torch.no_grad()
    def get_cproj_temporal_probe_state(self, param: torch.Tensor) -> dict:
        """Expose read-only state needed for a shadow temporal mechanism probe."""
        if not self._is_cproj(param):
            raise ValueError("temporal c_proj probe state requested for a non-c_proj parameter")
        group = self._group_for_param(param)
        st = self._init_mode_state(param, group)
        shadows = {}
        for mode, input_cov in st.get("shadow_input_cov", {}).items():
            shadows[mode] = {
                "input_cov": input_cov,
                "momentum": st["shadow_momentum"][mode].detach(),
            }
        return {
            "optimizer_step": int(self._step),
            "momentum_beta": float(group["momentum"]),
            "actual_mode": str(st["k_mode"]),
            "actual_input_cov": st["input_cov"],
            "actual_momentum": st["momentum"].detach(),
            "shadows": shadows,
        }

    @staticmethod
    def _cov_state_bytes(input_cov) -> tuple[int, int, int]:
        if input_cov is None:
            return 0, 0, 0
        state_bytes = int(input_cov.state_bytes(include_eye=True))
        matrix_bytes_fn = getattr(input_cov, "matrix_state_bytes", None)
        if callable(matrix_bytes_fn):
            matrix_bytes = int(matrix_bytes_fn())
        else:
            matrix_bytes = int(input_cov.state_bytes(include_eye=False))
        factors = len(input_cov.states) if isinstance(input_cov, BlockDiagInputCovState) else 1
        return state_bytes, matrix_bytes, factors

    @staticmethod
    def _cov_condition(input_cov) -> tuple[float, float]:
        if input_cov is None:
            return 0.0, 0.0
        cond_max = float(getattr(input_cov, "last_cond", 1.0))
        cond_mean = float(getattr(input_cov, "last_cond_mean", cond_max))
        return cond_mean, cond_max

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        active = 0
        k_cond_sum = 0.0
        k_cond_max = 0.0
        k_cond_count = 0
        k_state_bytes = 0
        k_matrix_bytes = 0
        k_state_factors = 0
        full_k_state_bytes = 0
        cproj_k_state_bytes = 0
        cproj_full_k_state_bytes = 0
        cproj_params = 0
        non_cproj_k_state_bytes = 0
        shadow_k_state_bytes = 0
        shadow_momentum_bytes = 0
        shadow_cproj_params = 0
        mode_counts = {mode: 0 for mode in self.valid_cproj_modes}

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            wd = group["weight_decay"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]
            diag_interval = group["diagnostic_interval"]
            diagnostics = diag_interval > 0 and self._step % diag_interval == 0

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError("CProjKModeNewtonMuon only supports 2D matrix parameters")

                st = self._init_mode_state(p, group)
                g = p.grad.detach().float()
                input_cov = st["input_cov"]
                mode = st["k_mode"]
                is_cproj = self._is_cproj(p)
                baseline_bytes = self._reference_k_state_bytes(
                    p,
                    is_cproj=is_cproj,
                    group=group,
                )
                full_k_state_bytes += baseline_bytes
                tracked_input = None
                if input_cov is not None or (
                    is_cproj and st.get("shadow_input_cov")
                ):
                    tracked_input = self._tracked_input_for(p)

                if input_cov is None:
                    r = g
                else:
                    input_cov.maybe_refresh(tracked_input, self._step, diagnostics)
                    apply_right = getattr(input_cov, "apply_right", None)
                    if callable(apply_right):
                        r = apply_right(g)
                    else:
                        r = g @ input_cov.K_inv.float()

                    cond_mean, cond_max = self._cov_condition(input_cov)
                    k_cond_sum += cond_mean
                    k_cond_max = max(k_cond_max, cond_max)
                    k_cond_count += 1

                state_bytes, matrix_bytes, factors = self._cov_state_bytes(input_cov)
                k_state_bytes += state_bytes
                k_matrix_bytes += matrix_bytes
                k_state_factors += factors
                if is_cproj:
                    cproj_params += 1
                    cproj_k_state_bytes += state_bytes
                    cproj_full_k_state_bytes += baseline_bytes
                    mode_counts[mode] += 1
                else:
                    non_cproj_k_state_bytes += state_bytes

                buf = st["momentum"]
                update = muon_momentum_direction(
                    r,
                    buf,
                    beta=mu,
                    nesterov=group["nesterov"],
                    momentum_ema=group["momentum_ema"],
                )
                q = muon_orthogonalize(
                    update,
                    name=self.param_to_name.get(p, ""),
                    ns_steps=ns_steps,
                    eps=eps,
                    split_qkv=group["split_qkv"],
                    adjust_lr_for_shape=group["adjust_lr_for_shape"],
                    ns_compute_dtype=group["ns_compute_dtype"],
                )

                if is_cproj:
                    shadow_covs = st.get("shadow_input_cov", {})
                    shadow_buffers = st.get("shadow_momentum", {})
                    if shadow_covs:
                        shadow_cproj_params += 1
                    for shadow_mode, shadow_cov in shadow_covs.items():
                        shadow_cov.maybe_refresh(
                            tracked_input,
                            self._step,
                            diagnostics,
                        )
                        apply_shadow_right = getattr(shadow_cov, "apply_right", None)
                        if callable(apply_shadow_right):
                            shadow_r = apply_shadow_right(g)
                        else:
                            shadow_r = g @ shadow_cov.K_inv.float()
                        muon_momentum_direction(
                            shadow_r,
                            shadow_buffers[shadow_mode],
                            beta=mu,
                            nesterov=group["nesterov"],
                            momentum_ema=group["momentum_ema"],
                        )
                        shadow_state_bytes, _, _ = self._cov_state_bytes(shadow_cov)
                        shadow_k_state_bytes += shadow_state_bytes
                        shadow_momentum_bytes += (
                            shadow_buffers[shadow_mode].numel()
                            * shadow_buffers[shadow_mode].element_size()
                        )

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(q.to(dtype=p.dtype), alpha=-lr)
                active += 1

        self._step += 1
        released_bytes = max(0, full_k_state_bytes - k_state_bytes)
        cproj_released_bytes = max(0, cproj_full_k_state_bytes - cproj_k_state_bytes)
        mode = self.param_groups[0]["cproj_k_mode"] if self.param_groups else "full"
        reference_mode = (
            self.param_groups[0]["cproj_k_reference_mode"]
            if self.param_groups
            else "full"
        )
        mode_id = {
            "none": 0,
            "full": 1,
            "block4": 2,
            "diag": 3,
            "alpha": 4,
            "scalar": 5,
            "block_alpha": 6,
        }[mode]
        cproj_offdiag_alpha = (
            group["cproj_k_offdiag_alpha"]
            if mode in ("alpha", "block_alpha")
            else {"diag": 0.0, "full": 1.0}.get(mode, -1.0)
        )
        configured_target_layers = (
            tuple(self.param_groups[0]["cproj_k_layers"])
            if self.param_groups
            else ()
        )
        target_layer_count = (
            len(configured_target_layers)
            if configured_target_layers
            else cproj_params
        )
        self.last_stats = {
            "active_params": active,
            "input_k_cond_mean": k_cond_sum / k_cond_count if k_cond_count else 0.0,
            "input_k_cond_max": k_cond_max if k_cond_count else 0.0,
            "input_k_cond_count": k_cond_count,
            "k_state_bytes": k_state_bytes,
            "k_matrix_bytes": k_matrix_bytes,
            "k_state_params": active - mode_counts["none"],
            "k_state_factors": k_state_factors,
            "k_state_released_params": mode_counts["none"],
            "full_k_state_bytes": full_k_state_bytes,
            "k_state_full_bytes": full_k_state_bytes,
            "k_state_released_bytes": released_bytes,
            "k_state_released_fraction": (
                released_bytes / full_k_state_bytes if full_k_state_bytes else 0.0
            ),
            "cproj_mode_id": mode_id,
            "cproj_reference_mode_id": {"full": 0, "block4": 1}[reference_mode],
            "cproj_offdiag_alpha": cproj_offdiag_alpha,
            "cproj_target_layer_count": target_layer_count,
            "cproj_target_layers_all": int(target_layer_count == cproj_params),
            "cproj_mode_applied_params": target_layer_count,
            "cproj_params": cproj_params,
            "cproj_k_state_bytes": cproj_k_state_bytes,
            "cproj_full_k_state_bytes": cproj_full_k_state_bytes,
            "cproj_k_state_released_bytes": cproj_released_bytes,
            "cproj_k_state_released_fraction": (
                cproj_released_bytes / cproj_full_k_state_bytes
                if cproj_full_k_state_bytes
                else 0.0
            ),
            "non_cproj_k_state_bytes": non_cproj_k_state_bytes,
            "cproj_none_params": mode_counts["none"],
            "cproj_full_params": mode_counts["full"],
            "cproj_block4_params": mode_counts["block4"],
            "cproj_diag_params": mode_counts["diag"],
            "cproj_scalar_params": mode_counts["scalar"],
            "cproj_alpha_params": mode_counts["alpha"],
            "cproj_block_alpha_params": mode_counts["block_alpha"],
            "shadow_k_state_bytes": shadow_k_state_bytes,
            "shadow_momentum_bytes": shadow_momentum_bytes,
            "shadow_cproj_params": shadow_cproj_params,
            "step": self._step,
        }
        return loss


class SelectiveNewtonMuon(_InputTrackedMatrixOptimizer):
    """Use input-side Newton-Muon only on layers where it buys useful descent per cost."""

    variant_name = "selective_newton_muon"

    def __init__(
        self,
        params,
        param_to_module: dict,
        param_to_name: dict | None = None,
        lr=0.02,
        momentum=0.95,
        weight_decay=0.0,
        ns_steps=5,
        eps=1e-8,
        input_beta=0.95,
        input_ridge=0.2,
        input_refresh=32,
        input_max_samples=None,
        diagnostic_interval=0,
        selective_fraction=0.5,
        selective_min_active=1,
        selective_warmup_steps=100,
        selective_score_interval=25,
        selective_score_beta=0.9,
        selective_score_threshold=0.0,
        selective_score_mode="gain_over_cost",
        selective_cond_power=1.0,
        selective_cost_power=1.0,
        selective_freeze_after_warmup=False,
        selective_log_diagnostics=True,
        selective_release_inactive_k_state=False,
        selective_selection_mode="fraction",
        selective_release_k_fraction=0.0,
        selective_static_newton_names=None,
        selective_static_report_names=None,
        selective_static_rank_by_name=None,
        selective_static_mask_label="",
        update_similarity_probe_enabled=False,
        update_similarity_probe_interval=25,
        update_similarity_probe_start_step=0,
        update_similarity_probe_stop_step=-1,
    ):
        if not 0.0 < selective_fraction <= 1.0:
            raise ValueError("selective_fraction must be in (0, 1]")
        if selective_min_active < 0:
            raise ValueError("selective_min_active must be non-negative")
        if selective_warmup_steps < 0:
            raise ValueError("selective_warmup_steps must be non-negative")
        if selective_score_interval <= 0:
            raise ValueError("selective_score_interval must be positive")
        if not 0.0 <= selective_score_beta < 1.0:
            raise ValueError("selective_score_beta must be in [0, 1)")
        if selective_score_mode not in (
            "gain_over_cost",
            "gain_logcond_over_cost",
            "gain_logcond_cost_power",
            "gain_logcond",
            "logcond_gain",
        ):
            raise ValueError(f"Unknown selective_score_mode: {selective_score_mode}")
        if selective_cond_power < 0.0:
            raise ValueError("selective_cond_power must be non-negative")
        if selective_cost_power < 0.0:
            raise ValueError("selective_cost_power must be non-negative")
        if selective_selection_mode not in ("fraction", "k_release_budget", "oracle_static", "shape_prior"):
            raise ValueError(f"Unknown selective_selection_mode: {selective_selection_mode}")
        if not 0.0 <= selective_release_k_fraction < 1.0:
            raise ValueError("selective_release_k_fraction must be in [0, 1)")
        if update_similarity_probe_interval <= 0:
            raise ValueError("update_similarity_probe_interval must be positive")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            eps=eps,
            input_beta=input_beta,
            input_ridge=input_ridge,
            input_refresh=input_refresh,
            input_max_samples=input_max_samples,
            diagnostic_interval=diagnostic_interval,
            selective_fraction=selective_fraction,
            selective_min_active=selective_min_active,
            selective_warmup_steps=selective_warmup_steps,
            selective_score_interval=selective_score_interval,
            selective_score_beta=selective_score_beta,
            selective_score_threshold=selective_score_threshold,
            selective_score_mode=selective_score_mode,
            selective_cond_power=selective_cond_power,
            selective_cost_power=selective_cost_power,
            selective_freeze_after_warmup=selective_freeze_after_warmup,
            selective_log_diagnostics=selective_log_diagnostics,
            selective_release_inactive_k_state=selective_release_inactive_k_state,
            selective_selection_mode=selective_selection_mode,
            selective_release_k_fraction=selective_release_k_fraction,
            selective_static_newton_names=set(selective_static_newton_names or []),
            selective_static_report_names=set(selective_static_report_names or []),
            selective_static_rank_by_name=dict(selective_static_rank_by_name or {}),
            selective_static_mask_label=selective_static_mask_label,
            update_similarity_probe_enabled=bool(update_similarity_probe_enabled),
            update_similarity_probe_interval=int(update_similarity_probe_interval),
            update_similarity_probe_start_step=int(update_similarity_probe_start_step),
            update_similarity_probe_stop_step=int(update_similarity_probe_stop_step),
        )
        super().__init__(params, param_to_module, defaults)
        self.param_to_name = param_to_name or {}
        self.last_selected_names = []
        self.last_layer_selection_report = []
        self.last_selection_summary = {}
        self._state_release_event = False

    @torch.no_grad()
    def _init_selective_state(self, p: torch.Tensor, group: dict):
        st = self.state[p]
        if st.get("selective_initialized", False):
            return st

        _, n = p.shape
        static_selection = self._static_selection_enabled(group)
        name = self.param_to_name.get(p, "")
        static_report_names = group.get("selective_static_report_names", set())
        static_newton_names = group.get("selective_static_newton_names", set())
        if static_selection and name in static_report_names:
            use_newton = name in static_newton_names
        else:
            use_newton = True

        st["selective_initialized"] = True
        if (not static_selection) or (not use_newton):
            st["muon_momentum"] = torch.zeros_like(p, dtype=torch.float32)
        if (not static_selection) or use_newton:
            st["newton_momentum"] = torch.zeros_like(p, dtype=torch.float32)
            st["input_cov"] = InputCovState(
                n=n,
                device=p.device,
                beta=group["input_beta"],
                ridge=group["input_ridge"],
                refresh_interval=group["input_refresh"],
                max_samples=group["input_max_samples"],
            )
        else:
            st["input_cov"] = None
        st["selective_score"] = 0.0
        st["selective_gain"] = 0.0
        st["selective_cost_proxy"] = self._cost_proxy(p, group)
        st["selective_use_newton"] = use_newton
        st["selective_scored"] = static_selection
        return st

    @staticmethod
    def _tensor_state_bytes(x) -> int:
        if isinstance(x, torch.Tensor):
            return x.numel() * x.element_size()
        return 0

    @torch.no_grad()
    def _ensure_muon_momentum(self, p: torch.Tensor, st: dict) -> torch.Tensor:
        buf = st.get("muon_momentum")
        if not isinstance(buf, torch.Tensor):
            buf = torch.zeros_like(p, dtype=torch.float32)
            st["muon_momentum"] = buf
        return buf

    @torch.no_grad()
    def _ensure_newton_momentum(self, p: torch.Tensor, st: dict) -> torch.Tensor:
        buf = st.get("newton_momentum")
        if not isinstance(buf, torch.Tensor):
            buf = torch.zeros_like(p, dtype=torch.float32)
            st["newton_momentum"] = buf
        return buf

    def consume_state_release_event(self) -> bool:
        event = self._state_release_event
        self._state_release_event = False
        return event

    @staticmethod
    def _static_selection_enabled(group: dict) -> bool:
        return (
            group.get("selective_selection_mode") in ("oracle_static", "shape_prior")
            and len(group.get("selective_static_report_names", set())) > 0
        )

    def _is_update_similarity_probe_step(self, group: dict) -> bool:
        if not group.get("update_similarity_probe_enabled", False):
            return False
        start = int(group.get("update_similarity_probe_start_step", 0))
        stop = int(group.get("update_similarity_probe_stop_step", -1))
        if self._step < start:
            return False
        if stop >= 0 and self._step > stop:
            return False
        interval = int(group.get("update_similarity_probe_interval", 25))
        return (self._step - start) % interval == 0

    @staticmethod
    def _layer_index_from_name(name: str) -> int:
        marker = "transformer.h."
        if marker not in name:
            return -1
        tail = name.split(marker, 1)[1]
        value = tail.split(".", 1)[0]
        return int(value) if value.isdigit() else -1

    @staticmethod
    def _module_type_from_name(name: str) -> str:
        if ".attn.c_attn.weight" in name:
            return "attn.c_attn"
        if ".attn.c_proj.weight" in name:
            return "attn.c_proj"
        if ".mlp.c_fc.weight" in name:
            return "mlp.c_fc"
        if ".mlp.c_proj.weight" in name:
            return "mlp.c_proj"
        return "other"

    @staticmethod
    def _rms(x: torch.Tensor) -> float:
        return float(torch.sqrt(torch.mean(x.float() * x.float())).item())

    @torch.no_grad()
    def _update_similarity_probe_stats(
        self,
        p: torch.Tensor,
        st: dict,
        g: torch.Tensor,
        q_muon: torch.Tensor,
        q_newton: torch.Tensor,
        k_cond: float,
        eps: float,
    ) -> None:
        probe = st.setdefault(
            "update_similarity_probe",
            {
                "count": 0,
                "update_cos_sum": 0.0,
                "update_misalignment_sum": 0.0,
                "relative_update_gap_sum": 0.0,
                "symmetric_update_gap_sum": 0.0,
                "grad_muon_cos_sum": 0.0,
                "grad_newton_cos_sum": 0.0,
                "grad_rms_sum": 0.0,
                "muon_update_rms_sum": 0.0,
                "newton_update_rms_sum": 0.0,
                "muon_descent_dot_sum": 0.0,
                "newton_descent_dot_sum": 0.0,
                "newton_minus_muon_descent_dot_sum": 0.0,
                "newton_over_muon_descent_dot_sum": 0.0,
                "k_cond_sum": 0.0,
                "k_cond_max": 0.0,
            },
        )
        q_muon_f = q_muon.float()
        q_newton_f = q_newton.float()
        diff = q_newton_f - q_muon_f
        q_muon_norm = q_muon_f.norm().clamp_min(eps)
        q_newton_norm = q_newton_f.norm().clamp_min(eps)
        diff_norm = diff.norm()
        update_cos = matrix_cosine(q_muon_f, q_newton_f, eps)
        muon_descent = float((g * q_muon_f).sum())
        newton_descent = float((g * q_newton_f).sum())

        probe["count"] += 1
        probe["update_cos_sum"] += update_cos
        probe["update_misalignment_sum"] += max(0.0, 1.0 - update_cos)
        probe["relative_update_gap_sum"] += float(diff_norm / q_newton_norm)
        probe["symmetric_update_gap_sum"] += float(diff_norm / (0.5 * (q_muon_norm + q_newton_norm)).clamp_min(eps))
        probe["grad_muon_cos_sum"] += matrix_cosine(g, q_muon_f, eps)
        probe["grad_newton_cos_sum"] += matrix_cosine(g, q_newton_f, eps)
        probe["grad_rms_sum"] += self._rms(g)
        probe["muon_update_rms_sum"] += self._rms(q_muon_f)
        probe["newton_update_rms_sum"] += self._rms(q_newton_f)
        probe["muon_descent_dot_sum"] += muon_descent
        probe["newton_descent_dot_sum"] += newton_descent
        probe["newton_minus_muon_descent_dot_sum"] += newton_descent - muon_descent
        probe["newton_over_muon_descent_dot_sum"] += newton_descent / (abs(muon_descent) + eps)
        probe["k_cond_sum"] += float(k_cond)
        probe["k_cond_max"] = max(float(probe.get("k_cond_max", 0.0)), float(k_cond))

    def _update_similarity_group_means(self, window: tuple[int, int] | None = None) -> tuple[int, int, float, float]:
        param_count = 0
        sample_count = 0
        cos_sum = 0.0
        gap_sum = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state.get(p, {})
                probe = st.get("update_similarity_probe")
                if not probe or int(probe.get("count", 0)) <= 0:
                    continue
                if window is not None:
                    name = self.param_to_name.get(p, "")
                    layer_idx = self._layer_index_from_name(name)
                    module_type = self._module_type_from_name(name)
                    if module_type != "mlp.c_proj" or not (window[0] <= layer_idx <= window[1]):
                        continue
                count = int(probe["count"])
                param_count += 1
                sample_count += count
                cos_sum += float(probe.get("update_cos_sum", 0.0)) / count
                gap_sum += float(probe.get("relative_update_gap_sum", 0.0)) / count
        return (
            param_count,
            sample_count,
            cos_sum / param_count if param_count else 0.0,
            gap_sum / param_count if param_count else 0.0,
        )

    def get_update_similarity_probe_report(self) -> list[dict]:
        rows = []
        rank = 0
        for group in self.param_groups:
            for p in group["params"]:
                rank += 1
                st = self.state.get(p, {})
                probe = st.get("update_similarity_probe")
                if not probe or int(probe.get("count", 0)) <= 0:
                    continue
                count = int(probe["count"])
                name = self.param_to_name.get(p, f"param_{rank}")
                layer_idx = self._layer_index_from_name(name)
                module_type = self._module_type_from_name(name)
                is_cproj = module_type == "mlp.c_proj"

                def mean(key: str) -> float:
                    return float(probe.get(key, 0.0)) / count

                rows.append(
                    {
                        "rank": rank,
                        "name": name,
                        "shape": "x".join(str(dim) for dim in p.shape),
                        "rows": int(p.shape[0]),
                        "cols": int(p.shape[1]),
                        "layer_idx": int(layer_idx),
                        "module_type": module_type,
                        "candidate_h4_h8_released": int(is_cproj and 4 <= layer_idx <= 8),
                        "candidate_h2_h9_released": int(is_cproj and 2 <= layer_idx <= 9),
                        "candidate_h3_h8_released": int(is_cproj and 3 <= layer_idx <= 8),
                        "probe_samples": count,
                        "update_cos_mean": mean("update_cos_sum"),
                        "update_misalignment_mean": mean("update_misalignment_sum"),
                        "relative_update_gap_mean": mean("relative_update_gap_sum"),
                        "symmetric_update_gap_mean": mean("symmetric_update_gap_sum"),
                        "grad_muon_cos_mean": mean("grad_muon_cos_sum"),
                        "grad_newton_cos_mean": mean("grad_newton_cos_sum"),
                        "grad_rms_mean": mean("grad_rms_sum"),
                        "muon_update_rms_mean": mean("muon_update_rms_sum"),
                        "newton_update_rms_mean": mean("newton_update_rms_sum"),
                        "muon_descent_dot_mean": mean("muon_descent_dot_sum"),
                        "newton_descent_dot_mean": mean("newton_descent_dot_sum"),
                        "newton_minus_muon_descent_dot_mean": mean("newton_minus_muon_descent_dot_sum"),
                        "newton_over_muon_descent_dot_mean": mean("newton_over_muon_descent_dot_sum"),
                        "k_cond_mean": mean("k_cond_sum"),
                        "k_cond_max": float(probe.get("k_cond_max", 0.0)),
                        "k_state_full_bytes": int(self._potential_input_cov_state_bytes(p, include_eye=True)),
                    }
                )
        return rows

    @staticmethod
    def _selection_mode_id(mode: str) -> int:
        if mode == "k_release_budget":
            return 1
        if mode == "oracle_static":
            return 2
        if mode == "shape_prior":
            return 3
        return 0

    @staticmethod
    def _potential_input_cov_state_bytes(p: torch.Tensor, include_eye: bool = True) -> int:
        n = int(p.shape[1])
        tensor_count = 3 if include_eye else 2
        element_size = torch.empty((), dtype=torch.float32, device=p.device).element_size()
        return n * n * element_size * tensor_count

    @staticmethod
    def _cost_proxy(p: torch.Tensor, group: dict) -> float:
        m, n = p.shape
        refresh = max(1, int(group["input_refresh"]))
        # Applying G @ K_inv is O(m n^2); refreshing the inverse is amortized O(n^3 / refresh).
        return float(m * n * n + (n * n * n) / refresh)

    def _is_score_step(self, group: dict) -> bool:
        if self._static_selection_enabled(group):
            return False
        if self._step < group["selective_warmup_steps"]:
            return True
        if group["selective_freeze_after_warmup"]:
            return False
        return self._step % group["selective_score_interval"] == 0

    @staticmethod
    def _score_needs_condition(group: dict) -> bool:
        return group["selective_score_mode"] != "gain_over_cost" and group["selective_cond_power"] > 0.0

    @staticmethod
    def _selection_score(gain: float, k_cond: float, cost_proxy: float, group: dict, eps: float) -> float:
        mode = group["selective_score_mode"]
        cost_m = max(cost_proxy / 1_000_000.0, eps)
        logcond = math.log1p(max(0.0, k_cond))
        cond_bonus = max(logcond, eps) ** group["selective_cond_power"]
        cost_penalty = cost_m ** group["selective_cost_power"]

        if mode == "gain_over_cost":
            return gain / max(cost_m, eps)
        if mode == "gain_logcond_over_cost":
            return gain * cond_bonus / max(cost_m, eps)
        if mode == "gain_logcond_cost_power":
            return gain * cond_bonus / max(cost_penalty, eps)
        if mode == "gain_logcond":
            return gain * cond_bonus
        if mode == "logcond_gain":
            return cond_bonus * (1.0 + gain)
        raise ValueError(f"Unknown selective_score_mode: {mode}")

    @torch.no_grad()
    def _select_by_fraction(self, ranked: list, group: dict) -> list:
        target = int(math.ceil(group["selective_fraction"] * len(ranked)))
        target = max(int(group["selective_min_active"]), target)
        target = min(len(ranked), target)
        threshold = group["selective_score_threshold"]

        selected = []
        for p in ranked:
            if len(selected) < target and self.state[p].get("selective_score", 0.0) >= threshold:
                selected.append(p)
        if len(selected) < min(target, int(group["selective_min_active"])):
            selected = ranked[: min(target, int(group["selective_min_active"]))]
        return selected

    @torch.no_grad()
    def _select_by_k_release_budget(self, ranked: list, group: dict) -> tuple[list, int, int, int]:
        full_bytes = {}
        total_bytes = 0
        for p in ranked:
            input_cov = self.state[p].get("input_cov")
            bytes_p = input_cov.full_state_bytes(include_eye=True) if input_cov is not None else 0
            full_bytes[p] = bytes_p
            total_bytes += bytes_p

        release_fraction = group["selective_release_k_fraction"]
        target_release_bytes = int(round(total_bytes * release_fraction))
        active_budget = max(0, total_bytes - target_release_bytes)
        min_active = min(len(ranked), int(group["selective_min_active"]))
        max_release = max(0, len(ranked) - min_active)
        if target_release_bytes <= 0 or max_release <= 0:
            return list(ranked), active_budget, total_bytes, target_release_bytes

        items = [
            (
                p,
                full_bytes[p],
                float(self.state[p].get("selective_score", 0.0)),
            )
            for p in ranked
        ]

        best_release_ids = set()
        best_key = (float("inf"), float("inf"), float("inf"), float("inf"))

        if len(items) <= 20:
            for mask in range(1 << len(items)):
                release_count = mask.bit_count()
                if release_count > max_release:
                    continue
                release_bytes = 0
                release_score = 0.0
                release_rank_sum = 0
                for idx, (_, bytes_p, score_p) in enumerate(items):
                    if mask & (1 << idx):
                        release_bytes += bytes_p
                        release_score += score_p
                        release_rank_sum += idx
                key = (
                    abs(release_bytes - target_release_bytes),
                    release_score,
                    release_count,
                    -release_rank_sum,
                )
                if key < best_key:
                    best_key = key
                    best_release_ids = {id(items[idx][0]) for idx in range(len(items)) if mask & (1 << idx)}
        else:
            released = []
            release_bytes = 0
            release_score = 0.0
            candidates = sorted(
                items,
                key=lambda item: (
                    item[2] / max(item[1], 1),
                    item[2],
                    -item[1],
                ),
            )
            best_release_ids = set()
            best_key = (abs(target_release_bytes), 0.0, 0, 0)
            for rank, (p, bytes_p, score_p) in enumerate(candidates):
                if len(released) >= max_release:
                    break
                released.append(p)
                release_bytes += bytes_p
                release_score += score_p
                key = (
                    abs(release_bytes - target_release_bytes),
                    release_score,
                    len(released),
                    rank,
                )
                if key < best_key:
                    best_key = key
                    best_release_ids = {id(param) for param in released}

        selected = [p for p in ranked if id(p) not in best_release_ids]
        return selected, active_budget, total_bytes, target_release_bytes

    @torch.no_grad()
    def _refresh_active_set(self, group: dict) -> None:
        params = [p for p in group["params"] if p in self.state and "selective_score" in self.state[p]]
        if not params:
            self.last_selected_names = []
            self.last_layer_selection_report = []
            self.last_selection_summary = {}
            return

        ranked = sorted(
            params,
            key=lambda p: self.state[p].get("selective_score", float("-inf")),
            reverse=True,
        )
        active_budget_bytes = 0
        full_budget_bytes = 0
        target_release_bytes = 0
        if group["selective_selection_mode"] == "k_release_budget":
            selected, active_budget_bytes, full_budget_bytes, target_release_bytes = (
                self._select_by_k_release_budget(ranked, group)
            )
        elif group["selective_selection_mode"] in ("oracle_static", "shape_prior"):
            selected = [
                p
                for p in ranked
                if self.param_to_name.get(p, "") in group.get("selective_static_newton_names", set())
            ]
            full_budget_bytes = sum(self._potential_input_cov_state_bytes(p, include_eye=True) for p in params)
            active_budget_bytes = sum(self._potential_input_cov_state_bytes(p, include_eye=True) for p in selected)
            target_release_bytes = int(round(full_budget_bytes * group["selective_release_k_fraction"]))
        else:
            selected = self._select_by_fraction(ranked, group)
            full_budget_bytes = sum(
                self.state[p]["input_cov"].full_state_bytes(include_eye=True)
                for p in params
                if self.state[p].get("input_cov") is not None
            )
            active_budget_bytes = full_budget_bytes

        selected_ids = {id(p) for p in selected}
        selected_full_bytes = 0
        released_full_bytes = 0
        report = []
        rank_by_id = {id(p): rank for rank, p in enumerate(ranked, start=1)}
        for p in params:
            st = self.state[p]
            selected_p = id(p) in selected_ids
            st["selective_use_newton"] = selected_p
            input_cov = st.get("input_cov")
            full_bytes = input_cov.full_state_bytes(include_eye=True) if input_cov is not None else 0
            current_bytes_before_release = input_cov.state_bytes(include_eye=True) if input_cov is not None else 0
            muon_momentum_bytes_before = self._tensor_state_bytes(st.get("muon_momentum"))
            newton_momentum_bytes_before = self._tensor_state_bytes(st.get("newton_momentum"))
            if selected_p:
                selected_full_bytes += full_bytes
            else:
                released_full_bytes += full_bytes
            released_state_bytes = 0
            if (
                group["selective_release_inactive_k_state"]
                and group["selective_freeze_after_warmup"]
                and not selected_p
            ):
                if input_cov is not None:
                    released_state_bytes += input_cov.state_bytes(include_eye=True)
                    input_cov.release()
            if group["selective_freeze_after_warmup"] and not group.get(
                "update_similarity_probe_enabled", False
            ):
                if selected_p:
                    released_state_bytes += self._tensor_state_bytes(st.pop("muon_momentum", None))
                else:
                    released_state_bytes += self._tensor_state_bytes(st.pop("newton_momentum", None))
            if released_state_bytes > 0:
                self._state_release_event = True
            current_bytes = input_cov.state_bytes(include_eye=True) if input_cov is not None else 0
            muon_momentum_bytes_after = self._tensor_state_bytes(st.get("muon_momentum"))
            newton_momentum_bytes_after = self._tensor_state_bytes(st.get("newton_momentum"))
            name = self.param_to_name.get(p, f"param_{len(report)}")
            report.append(
                {
                    "rank": rank_by_id.get(id(p), 0),
                    "name": name,
                    "shape": "x".join(str(dim) for dim in p.shape),
                    "rows": int(p.shape[0]),
                    "cols": int(p.shape[1]),
                    "score": float(st.get("selective_score", 0.0)),
                    "gain": float(st.get("selective_gain", 0.0)),
                    "cost_proxy": float(st.get("selective_cost_proxy", 0.0)),
                    "k_state_full_bytes": int(full_bytes),
                    "k_state_bytes_before_release": int(current_bytes_before_release),
                    "k_state_bytes_after_release": int(current_bytes),
                    "muon_momentum_bytes_before_release": int(muon_momentum_bytes_before),
                    "muon_momentum_bytes_after_release": int(muon_momentum_bytes_after),
                    "newton_momentum_bytes_before_release": int(newton_momentum_bytes_before),
                    "newton_momentum_bytes_after_release": int(newton_momentum_bytes_after),
                    "selected": int(selected_p),
                    "released": int(input_cov.is_released()) if input_cov is not None else 0,
                    "selection_mode": group["selective_selection_mode"],
                    "target_release_k_fraction": float(group["selective_release_k_fraction"]),
                    "static_mask_label": group.get("selective_static_mask_label", ""),
                }
            )
        self.last_selected_names = [self.param_to_name.get(p, f"param_{idx}") for idx, p in enumerate(selected)]
        self.last_layer_selection_report = sorted(report, key=lambda row: row["rank"])
        self.last_selection_summary = {
            "selection_mode_id": self._selection_mode_id(group["selective_selection_mode"]),
            "target_release_k_fraction": float(group["selective_release_k_fraction"]),
            "target_active_k_state_bytes": int(active_budget_bytes),
            "target_release_k_state_bytes": int(target_release_bytes),
            "full_k_state_bytes": int(full_budget_bytes),
            "selected_k_state_bytes": int(selected_full_bytes),
            "inactive_k_state_bytes": int(released_full_bytes),
            "inactive_k_state_fraction": (
                released_full_bytes / full_budget_bytes if full_budget_bytes > 0 else 0.0
            ),
            "release_budget_error_bytes": int(abs(released_full_bytes - target_release_bytes)),
        }

    @torch.no_grad()
    def _refresh_static_selection_report(self) -> None:
        report = []
        full_budget_bytes = 0
        selected_full_bytes = 0
        inactive_full_bytes = 0
        target_release_bytes = 0

        for group in self.param_groups:
            if not self._static_selection_enabled(group):
                continue
            initialized_params = [
                p
                for p in group["params"]
                if p in self.state and "selective_score" in self.state[p]
            ]
            group_full_bytes = sum(
                self._potential_input_cov_state_bytes(p, include_eye=True)
                for p in initialized_params
            )
            target_release_bytes += int(round(group_full_bytes * group["selective_release_k_fraction"]))
            rank_by_name = group.get("selective_static_rank_by_name", {})

            for idx, p in enumerate(initialized_params, start=1):
                st = self.state[p]
                name = self.param_to_name.get(p, f"param_{len(report)}")
                selected_p = bool(st.get("selective_use_newton", False))
                input_cov = st.get("input_cov")
                full_bytes = self._potential_input_cov_state_bytes(p, include_eye=True)
                current_bytes = input_cov.state_bytes(include_eye=True) if input_cov is not None else 0

                full_budget_bytes += full_bytes
                if selected_p:
                    selected_full_bytes += full_bytes
                else:
                    inactive_full_bytes += full_bytes

                report.append(
                    {
                        "rank": int(rank_by_name.get(name, idx)),
                        "name": name,
                        "shape": "x".join(str(dim) for dim in p.shape),
                        "rows": int(p.shape[0]),
                        "cols": int(p.shape[1]),
                        "score": float(st.get("selective_score", 0.0)),
                        "gain": float(st.get("selective_gain", 0.0)),
                        "cost_proxy": float(st.get("selective_cost_proxy", 0.0)),
                        "k_state_full_bytes": int(full_bytes),
                        "k_state_bytes_before_release": int(full_bytes),
                        "k_state_bytes_after_release": int(current_bytes),
                        "muon_momentum_bytes_before_release": int(p.numel() * 4),
                        "muon_momentum_bytes_after_release": int(
                            self._tensor_state_bytes(st.get("muon_momentum"))
                        ),
                        "newton_momentum_bytes_before_release": int(p.numel() * 4),
                        "newton_momentum_bytes_after_release": int(
                            self._tensor_state_bytes(st.get("newton_momentum"))
                        ),
                        "selected": int(selected_p),
                        "released": int(not selected_p),
                        "selection_mode": group["selective_selection_mode"],
                        "target_release_k_fraction": float(group["selective_release_k_fraction"]),
                        "static_mask_label": group.get("selective_static_mask_label", ""),
                    }
                )

        if report:
            self.last_selected_names = [row["name"] for row in report if row["selected"]]
            self.last_layer_selection_report = sorted(report, key=lambda row: row["rank"])
            self.last_selection_summary = {
                "selection_mode_id": self._selection_mode_id(
                    self.param_groups[0].get("selective_selection_mode", "")
                    if self.param_groups
                    else ""
                ),
                "target_release_k_fraction": float(
                    self.param_groups[0].get("selective_release_k_fraction", 0.0)
                    if self.param_groups
                    else 0.0
                ),
                "target_active_k_state_bytes": int(max(0, full_budget_bytes - target_release_bytes)),
                "target_release_k_state_bytes": int(target_release_bytes),
                "full_k_state_bytes": int(full_budget_bytes),
                "selected_k_state_bytes": int(selected_full_bytes),
                "inactive_k_state_bytes": int(inactive_full_bytes),
                "inactive_k_state_fraction": (
                    inactive_full_bytes / full_budget_bytes if full_budget_bytes > 0 else 0.0
                ),
                "release_budget_error_bytes": int(abs(inactive_full_bytes - target_release_bytes)),
            }

    def get_selective_layer_report(self) -> list[dict]:
        return list(self.last_layer_selection_report)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        active = 0
        newton_active = 0
        score_count = 0
        k_cond_sum = 0.0
        k_cond_max = 0.0
        k_cond_count = 0
        score_sum = 0.0
        score_max = float("-inf")
        gain_sum = 0.0
        gain_max = float("-inf")
        cos_update_base_sum = 0.0
        cos_grad_update_sum = 0.0
        cos_grad_base_sum = 0.0
        cos_diag_count = 0
        update_dual_eval = 0
        update_newton_only = 0
        update_muon_only = 0
        frozen_fast_path = 0
        log_diagnostics_enabled = False

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            wd = group["weight_decay"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]
            diag_interval = group["diagnostic_interval"]
            log_diagnostics = group["selective_log_diagnostics"]
            log_diagnostics_enabled = log_diagnostics_enabled or log_diagnostics
            static_selection = self._static_selection_enabled(group)
            score_step = self._is_score_step(group)
            cond_for_score = score_step and self._score_needs_condition(group)
            diagnostics = (
                (log_diagnostics and ((diag_interval > 0 and self._step % diag_interval == 0) or score_step))
                or cond_for_score
            )
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError("SelectiveNewtonMuon only supports 2D matrix parameters")

                st = self._init_selective_state(p, group)
                g = p.grad.detach().float()
                use_newton = bool(st.get("selective_use_newton", True))
                in_warmup = False if static_selection else self._step < group["selective_warmup_steps"]
                frozen_after_warmup = group["selective_freeze_after_warmup"] and not in_warmup
                probe_sample = self._is_update_similarity_probe_step(group)
                probe_can_compare = probe_sample and st.get("input_cov") is not None
                probe_keep_muon_momentum = bool(group.get("update_similarity_probe_enabled", False))

                need_muon_direction = score_step or (not use_newton and not in_warmup) or probe_can_compare
                need_muon_momentum = (
                    need_muon_direction
                    or not group["selective_freeze_after_warmup"]
                    or in_warmup
                    or probe_keep_muon_momentum
                )
                need_newton_direction = use_newton or score_step or in_warmup or probe_can_compare

                q_muon = None
                if need_muon_momentum:
                    muon_buf = self._ensure_muon_momentum(p, st)
                    muon_buf.mul_(mu).add_(g)
                    if need_muon_direction:
                        q_muon = matrix_sign_ns5(muon_buf, steps=ns_steps, eps=eps)

                q_newton = None
                if need_newton_direction:
                    input_cov: InputCovState = st["input_cov"]
                    if input_cov is None:
                        raise RuntimeError("Newton path requires input covariance state")
                    input_cov.maybe_refresh(self._tracked_input_for(p), self._step, diagnostics)
                    r = g @ input_cov.K_inv.float()
                    newton_buf = self._ensure_newton_momentum(p, st)
                    newton_buf.mul_(mu).add_(r)
                    q_newton = matrix_sign_ns5(newton_buf, steps=ns_steps, eps=eps)

                    if score_step:
                        if q_muon is None:
                            raise RuntimeError("Selective scoring requires a Muon comparison direction")
                        base_gain = float((g * q_muon).sum())
                        newton_gain = float((g * q_newton).sum())
                        rel_gain = (newton_gain - base_gain) / (abs(base_gain) + eps)
                        gain = max(0.0, rel_gain)
                        score = self._selection_score(
                            gain,
                            input_cov.last_cond,
                            st["selective_cost_proxy"],
                            group,
                            eps,
                        )
                        beta = group["selective_score_beta"]
                        if st.get("selective_scored", False):
                            st["selective_score"] = beta * st["selective_score"] + (1.0 - beta) * score
                            st["selective_gain"] = beta * st["selective_gain"] + (1.0 - beta) * gain
                        else:
                            st["selective_score"] = score
                            st["selective_gain"] = gain
                            st["selective_scored"] = True
                        score_count += 1
                        score_sum += st["selective_score"]
                        score_max = max(score_max, st["selective_score"])
                        gain_sum += st["selective_gain"]
                        gain_max = max(gain_max, st["selective_gain"])

                    q = q_newton if use_newton or in_warmup else q_muon
                    if use_newton or in_warmup:
                        newton_active += 1
                    if log_diagnostics or score_step:
                        k_cond_sum += input_cov.last_cond
                        k_cond_max = max(k_cond_max, input_cov.last_cond)
                        k_cond_count += 1
                    if log_diagnostics:
                        cos_update_base_sum += matrix_cosine(q, q_newton, eps)
                        cos_grad_base_sum += matrix_cosine(g, q_newton, eps)
                else:
                    if q_muon is None:
                        raise RuntimeError("Selective Muon path requires a Muon update direction")
                    q = q_muon
                    if log_diagnostics:
                        cos_update_base_sum += 1.0
                        cos_grad_base_sum += matrix_cosine(g, q_muon, eps)

                if probe_can_compare and q_muon is not None and q_newton is not None:
                    input_cov = st.get("input_cov")
                    k_cond = float(input_cov.last_cond) if input_cov is not None else 0.0
                    self._update_similarity_probe_stats(p, st, g, q_muon, q_newton, k_cond, eps)

                if q_muon is not None and q_newton is not None:
                    update_dual_eval += 1
                elif q_newton is not None:
                    update_newton_only += 1
                    if frozen_after_warmup:
                        frozen_fast_path += 1
                elif q_muon is not None:
                    update_muon_only += 1
                    if frozen_after_warmup:
                        frozen_fast_path += 1

                if log_diagnostics:
                    cos_grad_update_sum += matrix_cosine(g, q, eps)
                    cos_diag_count += 1

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(q.to(dtype=p.dtype), alpha=-lr)
                active += 1

            if score_step and self._step >= group["selective_warmup_steps"] - 1:
                self._refresh_active_set(group)

        self._step += 1
        all_scores = []
        all_gains = []
        active_scores = []
        k_state_bytes = 0
        k_matrix_bytes = 0
        k_state_full_bytes = 0
        k_state_released_bytes = 0
        k_state_params = 0
        k_state_released_params = 0
        selected_k_state_bytes = 0
        muon_momentum_bytes = 0
        newton_momentum_bytes = 0
        momentum_bytes = 0
        full_momentum_bytes = 0
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state.get(p, {})
                if "selective_score" not in st:
                    continue
                score = float(st.get("selective_score", 0.0))
                gain = float(st.get("selective_gain", 0.0))
                all_scores.append(score)
                all_gains.append(gain)
                if st.get("selective_use_newton", False):
                    active_scores.append(score)
                input_cov = st.get("input_cov")
                if input_cov is not None:
                    state_bytes = input_cov.state_bytes(include_eye=True)
                    matrix_bytes = input_cov.state_bytes(include_eye=False)
                    full_state_bytes = input_cov.full_state_bytes(include_eye=True)
                    k_state_bytes += state_bytes
                    k_matrix_bytes += matrix_bytes
                    k_state_full_bytes += full_state_bytes
                    k_state_released_bytes += max(0, full_state_bytes - state_bytes)
                    if st.get("selective_use_newton", False):
                        selected_k_state_bytes += full_state_bytes
                    if state_bytes > 0:
                        k_state_params += 1
                    if input_cov.is_released():
                        k_state_released_params += 1
                elif self._static_selection_enabled(group) and "selective_use_newton" in st:
                    full_state_bytes = self._potential_input_cov_state_bytes(p, include_eye=True)
                    k_state_full_bytes += full_state_bytes
                    k_state_released_bytes += full_state_bytes
                    if not st.get("selective_use_newton", False):
                        k_state_released_params += 1
                muon_bytes = self._tensor_state_bytes(st.get("muon_momentum"))
                newton_bytes = self._tensor_state_bytes(st.get("newton_momentum"))
                muon_momentum_bytes += muon_bytes
                newton_momentum_bytes += newton_bytes
                momentum_bytes += muon_bytes + newton_bytes
                full_momentum_bytes += p.numel() * 4 * 2
        if any(self._static_selection_enabled(group) for group in self.param_groups):
            self._refresh_static_selection_report()
        probe_params, probe_samples, probe_cos_mean, probe_gap_mean = self._update_similarity_group_means()
        h4h8_probe_params, h4h8_probe_samples, h4h8_probe_cos_mean, h4h8_probe_gap_mean = (
            self._update_similarity_group_means((4, 8))
        )
        h2h9_probe_params, h2h9_probe_samples, h2h9_probe_cos_mean, h2h9_probe_gap_mean = (
            self._update_similarity_group_means((2, 9))
        )
        h3h8_probe_params, h3h8_probe_samples, h3h8_probe_cos_mean, h3h8_probe_gap_mean = (
            self._update_similarity_group_means((3, 8))
        )
        n = max(1, active)
        self.last_stats = {
            "active_params": active,
            "selective_newton_params": newton_active,
            "selective_muon_params": max(0, active - newton_active),
            "selective_newton_fraction": newton_active / n if active else 0.0,
            "selective_score_count": score_count,
            "selective_score_mean": sum(all_scores) / len(all_scores) if all_scores else 0.0,
            "selective_score_max": max(all_scores) if all_scores else 0.0,
            "selective_gain_mean": sum(all_gains) / len(all_gains) if all_gains else 0.0,
            "selective_gain_max": max(all_gains) if all_gains else 0.0,
            "selective_active_score_mean": (
                sum(active_scores) / len(active_scores) if active_scores else 0.0
            ),
            "input_k_cond_mean": k_cond_sum / k_cond_count if k_cond_count else 0.0,
            "input_k_cond_max": k_cond_max if k_cond_count else 0.0,
            "input_k_cond_count": k_cond_count,
            "k_state_bytes": k_state_bytes,
            "k_matrix_bytes": k_matrix_bytes,
            "k_state_full_bytes": k_state_full_bytes,
            "k_state_released_bytes": k_state_released_bytes,
            "k_state_released_fraction": (
                k_state_released_bytes / k_state_full_bytes if k_state_full_bytes > 0 else 0.0
            ),
            "k_state_params": k_state_params,
            "k_state_released_params": k_state_released_params,
            "selective_selected_k_state_bytes": selected_k_state_bytes,
            "selective_muon_momentum_bytes": muon_momentum_bytes,
            "selective_newton_momentum_bytes": newton_momentum_bytes,
            "selective_momentum_bytes": momentum_bytes,
            "selective_full_momentum_bytes": full_momentum_bytes,
            "selective_released_momentum_bytes": max(0, full_momentum_bytes - momentum_bytes),
            "selective_released_momentum_fraction": (
                max(0, full_momentum_bytes - momentum_bytes) / full_momentum_bytes
                if full_momentum_bytes > 0
                else 0.0
            ),
            "cos_update_vs_base": cos_update_base_sum / cos_diag_count if cos_diag_count else 0.0,
            "cos_grad_vs_update": cos_grad_update_sum / cos_diag_count if cos_diag_count else 0.0,
            "cos_grad_vs_base": cos_grad_base_sum / cos_diag_count if cos_diag_count else 0.0,
            "selective_log_diagnostics": 1 if log_diagnostics_enabled else 0,
            "selective_update_dual_eval_params": update_dual_eval,
            "selective_update_newton_only_params": update_newton_only,
            "selective_update_muon_only_params": update_muon_only,
            "selective_frozen_fast_path_params": frozen_fast_path,
            "selective_selected_names_count": len(self.last_selected_names),
            "update_similarity_probe_params": probe_params,
            "update_similarity_probe_samples": probe_samples,
            "update_similarity_probe_update_cos_mean": probe_cos_mean,
            "update_similarity_probe_relative_gap_mean": probe_gap_mean,
            "update_similarity_probe_h4_h8_params": h4h8_probe_params,
            "update_similarity_probe_h4_h8_samples": h4h8_probe_samples,
            "update_similarity_probe_h4_h8_update_cos_mean": h4h8_probe_cos_mean,
            "update_similarity_probe_h4_h8_relative_gap_mean": h4h8_probe_gap_mean,
            "update_similarity_probe_h2_h9_params": h2h9_probe_params,
            "update_similarity_probe_h2_h9_samples": h2h9_probe_samples,
            "update_similarity_probe_h2_h9_update_cos_mean": h2h9_probe_cos_mean,
            "update_similarity_probe_h2_h9_relative_gap_mean": h2h9_probe_gap_mean,
            "update_similarity_probe_h3_h8_params": h3h8_probe_params,
            "update_similarity_probe_h3_h8_samples": h3h8_probe_samples,
            "update_similarity_probe_h3_h8_update_cos_mean": h3h8_probe_cos_mean,
            "update_similarity_probe_h3_h8_relative_gap_mean": h3h8_probe_gap_mean,
            "step": self._step,
        }
        self.last_stats.update(self.last_selection_summary)
        return loss


class DiagSigmaNewtonMuon(_InputTrackedMatrixOptimizer):
    """Diagonal Sigma_W proxy on top of input-side Newton-Muon."""

    variant_name = "diag_sigma_newton_muon"

    def __init__(
        self,
        params,
        param_to_module: dict,
        lr=0.02,
        momentum=0.95,
        weight_decay=0.0,
        ns_steps=5,
        eps=1e-8,
        input_beta=0.95,
        input_ridge=0.2,
        input_refresh=32,
        input_max_samples=None,
        sigma_beta=0.95,
        sigma_lambda_max=0.25,
        sigma_lambda_start=200,
        sigma_lambda_warmup=1000,
        sigma_eps=1e-4,
        stat_source="base",
        stat_mixed_rho=0.0,
        match_base_fro_norm=True,
        diagnostic_interval=0,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            eps=eps,
            input_beta=input_beta,
            input_ridge=input_ridge,
            input_refresh=input_refresh,
            input_max_samples=input_max_samples,
            sigma_beta=sigma_beta,
            sigma_lambda_max=sigma_lambda_max,
            sigma_lambda_start=sigma_lambda_start,
            sigma_lambda_warmup=sigma_lambda_warmup,
            sigma_eps=sigma_eps,
            stat_source=stat_source,
            stat_mixed_rho=stat_mixed_rho,
            match_base_fro_norm=match_base_fro_norm,
            diagnostic_interval=diagnostic_interval,
        )
        super().__init__(params, param_to_module, defaults)

    @torch.no_grad()
    def _init_state(self, p: torch.Tensor, group: dict):
        st = self._init_common_state(p, group)
        if "diag_sigma" not in st:
            m, n = p.shape
            st["diag_sigma"] = DiagSigmaState(
                m=m,
                n=n,
                device=p.device,
                beta=group["sigma_beta"],
                lambda_max=group["sigma_lambda_max"],
                lambda_start_step=group["sigma_lambda_start"],
                lambda_warmup_steps=group["sigma_lambda_warmup"],
                eps=group["sigma_eps"],
            )
        return st

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        totals = _RunningStats()
        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            wd = group["weight_decay"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]
            diag_interval = group["diagnostic_interval"]
            diagnostics = diag_interval > 0 and self._step % diag_interval == 0

            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self._init_state(p, group)
                input_cov: InputCovState = st["input_cov"]
                input_cov.maybe_refresh(self._tracked_input_for(p), self._step, diagnostics)

                r = p.grad.detach().float() @ input_cov.K_inv.float()
                buf = st["momentum"]
                buf.mul_(mu).add_(r)
                q_base = matrix_sign_ns5(buf, steps=ns_steps, eps=eps)

                sigma: DiagSigmaState = st["diag_sigma"]
                s = sigma.scale(self._step)
                y = buf * s.view(-1, 1)
                q_sigma = matrix_sign_ns5(y, steps=ns_steps, eps=eps) * s.view(-1, 1)
                if group["match_base_fro_norm"]:
                    q = fro_norm_match(q_sigma, q_base, eps=eps)
                else:
                    q = q_sigma

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(q.to(dtype=p.dtype), alpha=-lr)

                source = group["stat_source"]
                if source == "base":
                    u_stat = q_base
                elif source == "sigma":
                    u_stat = q
                elif source == "mixed":
                    rho = group["stat_mixed_rho"]
                    u_stat = (1.0 - rho) * q_base + rho * q
                else:
                    raise ValueError(f"Unknown sigma stat_source: {source}")
                sigma.update(u_stat)

                totals.add(
                    grad=p.grad.detach(),
                    base=q_base,
                    update=q,
                    input_cond=input_cov.last_cond,
                    sigma_stats=sigma.last_stats,
                    num_blocks=0,
                )

        self._step += 1
        self.last_stats = totals.as_dict(self._step)
        return loss


class BlockSigmaNewtonMuon(_InputTrackedMatrixOptimizer):
    """Block-shrunk Sigma_W Newton-Muon."""

    variant_name = "block_sigma_newton_muon"

    def __init__(
        self,
        params,
        param_to_module: dict,
        lr=0.02,
        momentum=0.95,
        weight_decay=0.0,
        ns_steps=5,
        eps=1e-8,
        input_beta=0.95,
        input_ridge=0.2,
        input_refresh=32,
        input_max_samples=None,
        sigma_block_size=64,
        sigma_beta=0.95,
        sigma_lambda_max=0.25,
        sigma_lambda_start=200,
        sigma_lambda_warmup=1000,
        sigma_eps=1e-4,
        sigma_refresh=16,
        stat_source="base",
        stat_mixed_rho=0.0,
        match_base_fro_norm=True,
        diagnostic_interval=0,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            eps=eps,
            input_beta=input_beta,
            input_ridge=input_ridge,
            input_refresh=input_refresh,
            input_max_samples=input_max_samples,
            sigma_block_size=sigma_block_size,
            sigma_beta=sigma_beta,
            sigma_lambda_max=sigma_lambda_max,
            sigma_lambda_start=sigma_lambda_start,
            sigma_lambda_warmup=sigma_lambda_warmup,
            sigma_eps=sigma_eps,
            sigma_refresh=sigma_refresh,
            stat_source=stat_source,
            stat_mixed_rho=stat_mixed_rho,
            match_base_fro_norm=match_base_fro_norm,
            diagnostic_interval=diagnostic_interval,
        )
        super().__init__(params, param_to_module, defaults)

    @torch.no_grad()
    def _init_state(self, p: torch.Tensor, group: dict):
        st = self._init_common_state(p, group)
        if "block_sigma" not in st:
            m, n = p.shape
            st["block_sigma"] = BlockSigmaState(
                m=m,
                n=n,
                device=p.device,
                block_size=group["sigma_block_size"],
                beta=group["sigma_beta"],
                lambda_max=group["sigma_lambda_max"],
                lambda_start_step=group["sigma_lambda_start"],
                lambda_warmup_steps=group["sigma_lambda_warmup"],
                eps=group["sigma_eps"],
                refresh_interval=group["sigma_refresh"],
            )
        return st

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        totals = _RunningStats()
        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            wd = group["weight_decay"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]
            diag_interval = group["diagnostic_interval"]
            diagnostics = diag_interval > 0 and self._step % diag_interval == 0

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError("BlockSigmaNewtonMuon only supports 2D matrix parameters")

                st = self._init_state(p, group)
                input_cov: InputCovState = st["input_cov"]
                input_cov.maybe_refresh(self._tracked_input_for(p), self._step, diagnostics)

                r = p.grad.detach().float() @ input_cov.K_inv.float()
                buf = st["momentum"]
                buf.mul_(mu).add_(r)
                q_base = matrix_sign_ns5(buf, steps=ns_steps, eps=eps)

                sigma: BlockSigmaState = st["block_sigma"]
                sigma.refresh_factors(self._step)
                y = sigma.left_factor_T(buf)
                s = matrix_sign_ns5(y, steps=ns_steps, eps=eps)
                q_sigma = sigma.left_factor(s)
                if group["match_base_fro_norm"]:
                    q = fro_norm_match(q_sigma, q_base, eps=eps)
                else:
                    q = q_sigma

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(q.to(dtype=p.dtype), alpha=-lr)

                source = group["stat_source"]
                if source == "base":
                    u_stat = q_base
                elif source == "sigma":
                    u_stat = q
                elif source == "mixed":
                    rho = group["stat_mixed_rho"]
                    u_stat = (1.0 - rho) * q_base + rho * q
                else:
                    raise ValueError(f"Unknown sigma stat_source: {source}")
                sigma.update_covariance(u_stat)

                stats = sigma.last_stats
                totals.add(
                    grad=p.grad.detach(),
                    base=q_base,
                    update=q,
                    input_cond=input_cov.last_cond,
                    sigma_stats={
                        "sigma_lambda": stats.sigma_lambda,
                        "sigma_anisotropy_mean": stats.sigma_anisotropy_mean,
                        "sigma_anisotropy_max": stats.sigma_anisotropy_max,
                        "sigma_offdiag_mean": stats.sigma_offdiag_mean,
                    },
                    num_blocks=stats.num_blocks,
                )

        self._step += 1
        self.last_stats = totals.as_dict(self._step)
        return loss


class _RunningStats:
    def __init__(self):
        self.active = 0
        self.input_cond_sum = 0.0
        self.input_cond_max = 0.0
        self.sigma_lambda_sum = 0.0
        self.sigma_anisotropy_sum = 0.0
        self.sigma_anisotropy_max = 0.0
        self.sigma_offdiag_sum = 0.0
        self.cos_update_base_sum = 0.0
        self.cos_grad_update_sum = 0.0
        self.cos_grad_base_sum = 0.0
        self.num_blocks = 0

    def add(
        self,
        grad: torch.Tensor,
        base: torch.Tensor,
        update: torch.Tensor,
        input_cond: float,
        sigma_stats: dict,
        num_blocks: int,
    ) -> None:
        self.active += 1
        self.input_cond_sum += input_cond
        self.input_cond_max = max(self.input_cond_max, input_cond)
        self.sigma_lambda_sum += float(sigma_stats.get("sigma_lambda", 0.0))
        self.sigma_anisotropy_sum += float(sigma_stats.get("sigma_anisotropy_mean", 1.0))
        self.sigma_anisotropy_max = max(
            self.sigma_anisotropy_max,
            float(sigma_stats.get("sigma_anisotropy_max", 1.0)),
        )
        self.sigma_offdiag_sum += float(sigma_stats.get("sigma_offdiag_mean", 0.0))
        self.cos_update_base_sum += matrix_cosine(update, base)
        self.cos_grad_update_sum += matrix_cosine(grad, update)
        self.cos_grad_base_sum += matrix_cosine(grad, base)
        self.num_blocks += num_blocks

    def as_dict(self, step: int) -> dict:
        n = max(1, self.active)
        return {
            "active_params": self.active,
            "input_k_cond_mean": self.input_cond_sum / n if self.active else 0.0,
            "input_k_cond_max": self.input_cond_max if self.active else 0.0,
            "sigma_lambda": self.sigma_lambda_sum / n if self.active else 0.0,
            "sigma_anisotropy_mean": self.sigma_anisotropy_sum / n if self.active else 0.0,
            "sigma_anisotropy_max": self.sigma_anisotropy_max if self.active else 0.0,
            "sigma_offdiag_mean": self.sigma_offdiag_sum / n if self.active else 0.0,
            "cos_update_vs_base": self.cos_update_base_sum / n if self.active else 0.0,
            "cos_grad_vs_update": self.cos_grad_update_sum / n if self.active else 0.0,
            "cos_grad_vs_base": self.cos_grad_base_sum / n if self.active else 0.0,
            "num_blocks": self.num_blocks,
            "step": step,
        }


class HybridOptimizer:
    """Combine AdamW with a Muon-family optimizer over disjoint parameters."""

    def __init__(self, optimizer_adamw, optimizer_muon):
        self.optimizer_adamw = optimizer_adamw
        self.optimizer_muon = optimizer_muon
        self.param_groups = self.optimizer_adamw.param_groups + self.optimizer_muon.param_groups

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        self.optimizer_adamw.step()
        self.optimizer_muon.step()
        return loss

    def zero_grad(self, set_to_none=True):
        self.optimizer_adamw.zero_grad(set_to_none=set_to_none)
        self.optimizer_muon.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            "adamw": self.optimizer_adamw.state_dict(),
            "muon": self.optimizer_muon.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.optimizer_adamw.load_state_dict(state_dict["adamw"])
        self.optimizer_muon.load_state_dict(state_dict["muon"])

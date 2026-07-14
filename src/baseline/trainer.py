#!/usr/bin/env python3
"""
trainer.py — Vanilla conditional diffusion trainer.

Checkpoint selection:
  - Uses validation denoising loss (MSE between predicted and target on val set).
  - This is the standard approach used by LDM, Stable Diffusion, and DDPM internally.
  - Cheap: no DDIM sampling needed. Directly measures what the model is learning.
  - Val timesteps are FIXED per eval (seeded) so loss is deterministic across epochs.

Generative eval:
  - Runs full DDIM every eval_every epochs as a DIAGNOSTIC only.
  - Multi-seed (mean ± std) following NeurIPS/ICLR/ICML reproducibility requirements.
  - Does NOT affect checkpoint selection. Watch in TensorBoard to confirm training.

Final test numbers:
  - Run sample.py on test set using best.pt AFTER training ends.
  - Never report diagnostic gen eval numbers as final results.
"""
import os, time, json
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from config import TrainConfig
from data import make_loaders
from diffusion import build_betas          # single source of truth
from models import ConditionalUNet, sinusoidal_embedding
from utils_ema import EMA
from utils_log import AvgMeter, CSVLogger, TBLogger, gpu_mem_gb


# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────
def seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# ──────────────────────────────────────────────────────────────────────────────
# EMA swap helper
# ──────────────────────────────────────────────────────────────────────────────
class _EMASwap:
    """Temporarily replace model weights with EMA weights for evaluation."""
    def __init__(self, ema, net):
        self.ema = ema
        self.net = net

    def swap_in(self) -> bool:
        if self.ema is None:
            return False
        self.ema.store(self.net)
        self.ema.copy_to(self.net)
        return True

    def swap_out(self):
        if self.ema is not None:
            self.ema.restore(self.net)


# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────
class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        os.makedirs(cfg.out_dir, exist_ok=True)
        seed_everything(cfg.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.pred_type = str(cfg.pred_type).lower()
        assert self.pred_type in ("v", "eps"), (
            f"pred_type must be 'v' or 'eps', got '{self.pred_type}'"
        )

        # ── Data ──────────────────────────────────────────────────────────────
        self.ds_train, self.ds_val, self.loader_train, self.loader_val = make_loaders(
            csv_train=cfg.csv_train,
            csv_val=cfg.csv_val,
            feature_dir=cfg.feature_dir,
            label_dir=cfg.label_dir,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        )
        self.C_feat  = self.ds_train.C_feat
        self.C_label = self.ds_train.C_label
        self.H, self.W = self.ds_train.H, self.ds_train.W
        assert (self.H, self.W) == (256, 256), (
            f"Expected 256×256, got {self.H}×{self.W}"
        )

        # Persist invariants so sampler can hard-validate alignment
        inv = {
            "C_feat":          self.C_feat,
            "C_label":         self.C_label,
            "H":               self.H,
            "W":               self.W,
            "pred_type":       self.pred_type,
            "beta_schedule":   cfg.beta_schedule,
            "diffusion_steps": cfg.diffusion_steps,
            "use_self_cond":   cfg.use_self_cond,
        }
        with open(os.path.join(cfg.out_dir, "invariants.json"), "w") as f:
            json.dump(inv, f, indent=2)

        # ── Model ─────────────────────────────────────────────────────────────
        # in_ch = x_t (C_label) + features (C_feat) + optional self_cond (C_label)
        self.in_ch = (
            self.C_label
            + self.C_feat
            + (self.C_label if cfg.use_self_cond else 0)
        )
        self.net = ConditionalUNet(
            in_ch=self.in_ch,
            out_ch=self.C_label,
            base=cfg.base_channels,
            t_emb_dim=128,
            dropout=cfg.dropout,
        ).to(self.device)
        n_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)

        # ── Optimizer ─────────────────────────────────────────────────────────
        self.optimizer = AdamW(
            self.net.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        self.global_step = 0

        # ── EMA ───────────────────────────────────────────────────────────────
        self.ema      = EMA(self.net, decay=cfg.ema_decay) if cfg.ema else None
        self.ema_swap = _EMASwap(self.ema, self.net)

        # ── Diffusion schedule (single source: diffusion.py) ──────────────────
        self.T = int(cfg.diffusion_steps)
        betas_np = build_betas(self.T, cfg.beta_schedule)
        self.betas     = torch.tensor(betas_np, dtype=torch.float32, device=self.device)
        self.alphas    = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

        # ── Checkpoint tracking (by val denoising loss) ───────────────────────
        self.best_val_loss = float("inf")
        self.best_epoch    = -1

        if cfg.eval_seeds is None:
            cfg.eval_seeds = [cfg.seed]

        # ── Loggers ───────────────────────────────────────────────────────────
        self.csv_log = CSVLogger(
            os.path.join(cfg.out_dir, "train_log.csv"),
            fieldnames=["epoch", "train_loss", "val_loss", "lr", "time_s"],
        )
        self.tb_log = TBLogger(os.path.join(cfg.out_dir, "tb"))

        # ── Print summary (once) ──────────────────────────────────────────────
        print("=" * 70)
        print(f"  VANILLA CONDITIONAL DIFFUSION BASELINE")
        print(f"  Task: {cfg.task} | Tech: {cfg.tech} | Device: {self.device}")
        print(f"  Data: {len(self.ds_train)} train / {len(self.ds_val)} val")
        print(f"  Features: {self.C_feat}ch → Label: {self.C_label}ch | {self.H}×{self.W}")
        print(
            f"  Model input: {self.in_ch}ch "
            f"(x_t={self.C_label} + feat={self.C_feat}"
            + (f" + self_cond={self.C_label}" if cfg.use_self_cond else "")
            + ")"
        )
        print(f"  Parameters: {n_params / 1e6:.2f}M")
        print(
            f"  Training: {cfg.epochs} epochs, bs={cfg.batch_size}, "
            f"lr={cfg.lr}, warmup={cfg.warmup_steps} steps"
        )
        print(
            f"  Diffusion: T={self.T}, schedule={cfg.beta_schedule}, "
            f"pred={self.pred_type}"
        )
        print(
            f"  CFG: drop_prob={cfg.cfg_drop_prob} | "
            f"Aux loss: weight={cfg.aux_weight}, t<{cfg.aux_t_cutoff}"
        )
        print(
            f"  Min-SNR: gamma={cfg.min_snr_gamma} | "
            f"EMA: {cfg.ema} (decay={cfg.ema_decay})"
        )
        print(
            f"  Checkpoint: best.pt selected by val denoising loss "
            f"(seeded, deterministic)"
        )
        print(
            f"  Gen eval (diagnostic): every {cfg.eval_every} epochs, "
            f"{cfg.eval_gen_steps} DDIM steps, "
            f"cfg_scale={cfg.eval_cfg_scale}, "
            f"{len(cfg.eval_seeds)} seeds={cfg.eval_seeds}"
        )
        print(f"  Output: {cfg.out_dir}")
        print("=" * 70)

        # ── Sanity check ──────────────────────────────────────────────────────
        feats0, lbls0, _ = next(iter(self.loader_train))
        assert feats0.shape[1] == self.C_feat, (
            f"Feature channels: expected {self.C_feat}, got {feats0.shape[1]}"
        )
        assert lbls0.shape[1] == self.C_label, (
            f"Label channels: expected {self.C_label}, got {lbls0.shape[1]}"
        )
        print(f"[Sanity] feats={tuple(feats0.shape)}, lbls={tuple(lbls0.shape)}")

    # ──────────────────────────────────────────────────────────────────────────
    # Diffusion helpers
    # ──────────────────────────────────────────────────────────────────────────
    def _abar(self, t: torch.Tensor) -> torch.Tensor:
        return self.alpha_bar[t].view(-1, 1, 1, 1)

    def q_sample(self, x0, t, noise):
        ab = self._abar(t)
        return torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * noise

    def _x0_from_pred(self, x_t, pred, t):
        ab = self._abar(t)
        if self.pred_type == "eps":
            x0 = (x_t - torch.sqrt(1.0 - ab) * pred) / torch.sqrt(ab + 1e-12)
        else:
            x0 = torch.sqrt(ab) * x_t - torch.sqrt(1.0 - ab) * pred
        return x0.clamp(-1, 1)

    def _eps_from_pred(self, x_t, pred, t):
        ab = self._abar(t)
        if self.pred_type == "eps":
            return pred
        return torch.sqrt(1.0 - ab) * x_t + torch.sqrt(ab) * pred

    def _compute_target(self, x0, noise, t):
        if self.pred_type == "eps":
            return noise
        ab = self._abar(t)
        return torch.sqrt(ab) * noise - torch.sqrt(1.0 - ab) * x0

    def _min_snr_weights(self, t):
        """
        Min-SNR weighting (Hang et al., 2023).
          eps: w = min(SNR, γ) / SNR
          v:   w = min(SNR, γ) / (SNR + 1)    ← correct for v-prediction
        Returns None if disabled.
        """
        if self.cfg.min_snr_gamma <= 0:
            return None
        ab  = self._abar(t)
        snr = ab / torch.clamp(1.0 - ab, min=1e-12)
        γ   = torch.tensor(self.cfg.min_snr_gamma, device=snr.device, dtype=snr.dtype)
        if self.pred_type == "v":
            w = torch.minimum(snr, γ) / (snr + 1.0)
        else:
            w = torch.minimum(snr, γ) / torch.clamp(snr, min=1e-12)
        return w.view(-1)

    def _get_lr(self) -> float:
        if self.global_step < self.cfg.warmup_steps:
            return self.cfg.lr * (self.global_step + 1) / self.cfg.warmup_steps
        return self.cfg.lr

    def _apply_lr(self):
        lr = self._get_lr()
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    # ──────────────────────────────────────────────────────────────────────────
    # Model input assembly (single function — training and sampling both use this)
    # ──────────────────────────────────────────────────────────────────────────
    def _build_model_input(self, x_t, feats, self_cond=None):
        """
        Assemble [x_t, (self_cond,) features] along channel dim.
        Called identically in training and sampling — no mismatch possible.
        """
        parts = [x_t]
        if self.cfg.use_self_cond:
            parts.append(self_cond if self_cond is not None else torch.zeros_like(x_t))
        parts.append(feats)
        return torch.cat(parts, dim=1)

    # ──────────────────────────────────────────────────────────────────────────
    # CFG dropout (training only)
    # ──────────────────────────────────────────────────────────────────────────
    def _cfg_dropout(self, feats):
        """
        Zero out features for cfg_drop_prob fraction of the batch.
        Returns (feats_maybe_dropped, drop_mask).
        drop_mask: (B,) bool — True where features were zeroed.
        """
        B = feats.shape[0]
        if self.cfg.cfg_drop_prob <= 0:
            return feats, torch.zeros(B, dtype=torch.bool, device=feats.device)
        drop = torch.rand(B, device=feats.device) < self.cfg.cfg_drop_prob
        if not drop.any():
            return feats, drop
        feats_out = feats.clone()
        feats_out[drop] = 0.0
        return feats_out, drop

    # ──────────────────────────────────────────────────────────────────────────
    # DDIM sampler (used for diagnostic generative eval during training)
    # ──────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _ddim_sample(self, feats, steps=100, eta=0.0, cfg_scale=1.5, seed=1234):
        """
        DDIM sampling conditioned on feature images.
        Returns: (B, C_label, H, W) in [0, 1].
        """
        self.net.eval()
        B, _, H, W = feats.shape
        g = torch.Generator(device=feats.device)
        g.manual_seed(int(seed))

        x          = torch.randn(B, self.C_label, H, W, device=feats.device, generator=g)
        self_cond  = None
        ts         = torch.linspace(self.T - 1, 0, steps, device=feats.device).long()
        abar       = self.alpha_bar
        feats_null = torch.zeros_like(feats)

        for i in range(steps):
            t     = ts[i].expand(B)
            t_emb = sinusoidal_embedding(t, dim=128)
            ab_t  = abar[t].view(B, 1, 1, 1)

            if cfg_scale > 0.0:
                # Unconditional pass: null features, no self-cond
                # (never trained with conditional self-cond on unconditional pass)
                pred_u = self.net(self._build_model_input(x, feats_null, None), t_emb)
                pred_c = self.net(self._build_model_input(x, feats, self_cond), t_emb)
                pred   = pred_u + cfg_scale * (pred_c - pred_u)
            else:
                pred = self.net(self._build_model_input(x, feats, self_cond), t_emb)

            x0_hat = self._x0_from_pred(x, pred, t)
            eps    = self._eps_from_pred(x, pred, t)

            if self.cfg.use_self_cond:
                self_cond = x0_hat.detach()

            if i == steps - 1:
                x = x0_hat
                break

            t_next  = ts[i + 1].expand(B)
            ab_next = abar[t_next].view(B, 1, 1, 1)
            sigma   = (
                eta
                * torch.sqrt((1 - ab_next) / (1 - ab_t + 1e-12))
                * torch.sqrt(torch.clamp(1 - ab_t / (ab_next + 1e-12), min=0.0))
            )
            noise = torch.randn_like(x, generator=g) if eta > 0 else 0.0
            x = (
                torch.sqrt(ab_next) * x0_hat
                + torch.sqrt(torch.clamp(1 - ab_next - sigma ** 2, min=0.0)) * eps
                + sigma * noise
            )

        return ((x + 1) / 2).clamp(0, 1)

    # ──────────────────────────────────────────────────────────────────────────
    # Training loop
    # ──────────────────────────────────────────────────────────────────────────
    def train(self):
        cfg = self.cfg
        print(f"\n[TRAIN] Starting {cfg.epochs} epochs\n")

        for epoch in range(1, cfg.epochs + 1):
            self.net.train()
            loss_meter = AvgMeter()
            t_epoch    = time.time()

            for bi, (feats, lbls, _) in enumerate(self.loader_train, 1):
                feats = feats.float().to(self.device)
                lbls  = lbls.float().to(self.device)
                B     = lbls.size(0)

                self._apply_lr()

                # Labels [0,1] → [-1,1] for diffusion
                x0    = (lbls * 2 - 1).clamp(-1, 1)
                t     = torch.randint(0, self.T, (B,), device=self.device)
                t_emb = sinusoidal_embedding(t, dim=128)
                noise = torch.randn_like(x0)
                x_t   = self.q_sample(x0, t, noise)

                # CFG: randomly drop features for 10% of batch
                feats_in, drop_mask = self._cfg_dropout(feats)

                # Self-conditioning (optional)
                self_cond = None
                if cfg.use_self_cond and torch.rand(1).item() < cfg.self_cond_prob:
                    with torch.no_grad():
                        pred_sc   = self.net(self._build_model_input(x_t, feats_in, None), t_emb)
                        self_cond = self._x0_from_pred(x_t, pred_sc, t).detach()

                # Forward pass
                pred   = self.net(self._build_model_input(x_t, feats_in, self_cond), t_emb)
                target = self._compute_target(x0, noise, t)

                # Main loss: MSE with Min-SNR weighting
                mse_per = ((pred - target) ** 2).mean(dim=(1, 2, 3))
                weights = self._min_snr_weights(t)
                loss_main = (mse_per * weights).mean() if weights is not None else mse_per.mean()

                # Aux loss: direct x0 L1 for low-t steps (skip CFG-dropped)
                loss_aux = torch.tensor(0.0, device=self.device)
                if cfg.aux_weight > 0:
                    aux_mask = (t < cfg.aux_t_cutoff) & (~drop_mask)
                    if aux_mask.any():
                        x0_hat   = self._x0_from_pred(x_t[aux_mask], pred[aux_mask], t[aux_mask])
                        loss_aux = F.l1_loss(x0_hat, x0[aux_mask])

                loss = loss_main + cfg.aux_weight * loss_aux

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), cfg.grad_clip)
                self.optimizer.step()
                self.global_step += 1

                if cfg.ema:
                    self.ema.update(self.net)

                loss_meter.update(loss.item(), B)

                if bi % 50 == 0:
                    print(
                        f"  [E{epoch:03d} B{bi:04d}] "
                        f"loss={loss_meter.avg:.5f} "
                        f"lr={self.optimizer.param_groups[0]['lr']:.2e} "
                        f"mem={gpu_mem_gb():.1f}GB"
                    )

            elapsed = time.time() - t_epoch
            lr_now  = self.optimizer.param_groups[0]["lr"]
            print(f"[Epoch {epoch:03d}] loss={loss_meter.avg:.6f} time={elapsed:.0f}s")

            # ── Periodic evaluation ────────────────────────────────────────────
            val_loss = None
            if epoch % cfg.eval_every == 0:
                val_loss = self._evaluate(epoch)

            # ── CSV logging ───────────────────────────────────────────────────
            self.csv_log.log({
                "epoch":      epoch,
                "train_loss": f"{loss_meter.avg:.6f}",
                "val_loss":   f"{val_loss:.6f}" if val_loss is not None else "",
                "lr":         f"{lr_now:.2e}",
                "time_s":     f"{elapsed:.1f}",
            })
            self.tb_log.add_scalar("train/loss", loss_meter.avg, epoch)
            self.tb_log.add_scalar("train/lr",   lr_now,         epoch)
            self.tb_log.flush()

            # Always save latest for crash recovery
            self._save_ckpt(os.path.join(cfg.out_dir, "latest.pt"), epoch)

        best_str = (
            f"{self.best_val_loss:.6f} @ epoch {self.best_epoch}"
            if self.best_epoch >= 0
            else "N/A (no eval ran)"
        )
        print(f"\n[TRAIN] Done. Best val_loss={best_str}")

    # ──────────────────────────────────────────────────────────────────────────
    # Checkpointing
    # ──────────────────────────────────────────────────────────────────────────
    def _save_ckpt(self, path: str, epoch: int):
        ck = {
            "net":             self.net.state_dict(),
            "opt":             self.optimizer.state_dict(),
            "epoch":           epoch,
            "global_step":     self.global_step,
            "best_val_loss":   float(self.best_val_loss),
            "best_epoch":      self.best_epoch,
            # Architecture contract — sampler MUST match all of these
            "pred_type":       self.pred_type,
            "beta_schedule":   self.cfg.beta_schedule,
            "diffusion_steps": self.T,
            "task":            self.cfg.task,
            "tech":            self.cfg.tech,
            "use_self_cond":   self.cfg.use_self_cond,
            "C_feat":          self.C_feat,
            "C_label":         self.C_label,
            "in_ch":           self.in_ch,
            "base_channels":   self.cfg.base_channels,
            "dropout":         self.cfg.dropout,
        }
        if self.cfg.ema:
            ck["ema"] = self.ema.state_dict()
        torch.save(ck, path)

    @staticmethod
    def load_for_inference(ckpt_path: str, device: torch.device = None) -> dict:
        """
        Load checkpoint and rebuild model for inference.
        Hard-validates all architecture parameters against config and invariants.json.
        Returns dict: {'model', 'cfg', 'ckpt'} — model is eval-mode with EMA applied.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt     = torch.load(ckpt_path, map_location=device)
        ckpt_dir = os.path.dirname(ckpt_path)
        cfg      = TrainConfig.load(os.path.join(ckpt_dir, "run_config.json"))

        # Hard guards: ckpt must match run_config
        for name, ckpt_val, cfg_val in [
            ("task",            ckpt.get("task"),            cfg.task),
            ("tech",            ckpt.get("tech"),            cfg.tech),
            ("pred_type",       ckpt.get("pred_type"),       cfg.pred_type),
            ("beta_schedule",   ckpt.get("beta_schedule"),   cfg.beta_schedule),
            ("diffusion_steps", ckpt.get("diffusion_steps"), cfg.diffusion_steps),
            ("use_self_cond",   ckpt.get("use_self_cond"),   cfg.use_self_cond),
        ]:
            assert ckpt_val == cfg_val, (
                f"Checkpoint mismatch: {name}={ckpt_val} (ckpt) vs {cfg_val} (config)"
            )

        # Cross-validate against invariants.json
        inv_path = os.path.join(ckpt_dir, "invariants.json")
        if os.path.exists(inv_path):
            with open(inv_path) as f:
                inv = json.load(f)
            assert int(ckpt["C_feat"])  == int(inv["C_feat"]),  f"C_feat mismatch"
            assert int(ckpt["C_label"]) == int(inv["C_label"]), f"C_label mismatch"
            expected_in_ch = (
                int(inv["C_label"]) + int(inv["C_feat"])
                + (int(inv["C_label"]) if inv["use_self_cond"] else 0)
            )
            assert int(ckpt["in_ch"]) == expected_in_ch, (
                f"in_ch mismatch: ckpt={ckpt['in_ch']} vs expected={expected_in_ch}"
            )
        else:
            print(f"[Load] WARNING: invariants.json not found, skipping cross-validation")

        # Rebuild model
        in_ch   = int(ckpt["in_ch"])
        C_label = int(ckpt["C_label"])
        base    = int(ckpt.get("base_channels", cfg.base_channels))

        model = ConditionalUNet(
            in_ch=in_ch, out_ch=C_label, base=base, t_emb_dim=128, dropout=0.0
        ).to(device)
        model.load_state_dict(ckpt["net"], strict=True)

        if "ema" in ckpt:
            ema = EMA(model)
            ema.load_state_dict(ckpt["ema"])
            ema.copy_to(model)
            print("[Load] EMA weights applied")

        model.eval()
        print(
            f"[Load] {os.path.basename(ckpt_path)}: "
            f"epoch={ckpt['epoch']} | "
            f"task={ckpt['task']} tech={ckpt['tech']} | "
            f"in_ch={in_ch} out_ch={C_label} base={base} | "
            f"pred={ckpt['pred_type']} schedule={ckpt['beta_schedule']}"
        )
        return {"model": model, "cfg": cfg, "ckpt": ckpt}

    # ──────────────────────────────────────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _evaluate(self, epoch: int) -> float:
        """
        Returns val_loss (used for checkpoint selection).

        Two-part eval:
          1. Val denoising loss  — cheap, deterministic (seeded timesteps),
                                   used to select best.pt.
          2. Generative eval     — expensive (full DDIM), multi-seed,
                                   diagnostic only. Does NOT affect best.pt.
        """
        cfg = self.cfg
        print(f"\n[EVAL E{epoch:03d}]")

        # Swap in EMA weights — this is what we deploy
        swapped = self.ema_swap.swap_in()
        if swapped:
            print(f"[EVAL E{epoch:03d}] EMA weights active")
        self.net.eval()

        # ── Part 1: Val denoising loss (checkpoint selection) ─────────────────
        # Timesteps are FIXED by a seeded generator so the loss is deterministic
        # across epochs. If it goes down, the model genuinely improved.
        # Reference: LDM, Stable Diffusion, DDPM all use this internally.
        val_loss_meter = AvgMeter()
        val_rng = torch.Generator(device=self.device)
        val_rng.manual_seed(cfg.seed)   # same seed every eval → deterministic

        for feats, lbls, _ in self.loader_val:
            feats = feats.float().to(self.device)
            lbls  = lbls.float().to(self.device)
            x0    = (lbls * 2 - 1).clamp(-1, 1)
            B     = x0.size(0)

            # Fixed timesteps — seeded, deterministic
            t     = torch.randint(0, self.T, (B,), device=self.device, generator=val_rng)
            t_emb = sinusoidal_embedding(t, dim=128)
            noise = torch.randn_like(x0)
            x_t   = self.q_sample(x0, t, noise)

            target  = self._compute_target(x0, noise, t)
            # No CFG dropout during validation — always fully conditioned
            x_in    = self._build_model_input(x_t, feats, None)
            pred    = self.net(x_in, t_emb)

            mse_per = ((pred - target) ** 2).mean(dim=(1, 2, 3))
            weights = self._min_snr_weights(t)
            loss    = (mse_per * weights).mean() if weights is not None else mse_per.mean()
            val_loss_meter.update(loss.item(), B)

        val_loss = val_loss_meter.avg
        print(f"[EVAL E{epoch:03d}][ValLoss] {val_loss:.6f}  (checkpoint selection criterion)")

        # Checkpoint selection
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch    = epoch
            self._save_ckpt(os.path.join(cfg.out_dir, "best.pt"), epoch)
            print(
                f"[EVAL E{epoch:03d}] ★ New best!  "
                f"val_loss={self.best_val_loss:.6f} @ epoch {self.best_epoch}"
            )

        # ── Part 2: Generative eval (diagnostic — does NOT affect best.pt) ────
        # Multi-seed following NeurIPS/ICLR/ICML reproducibility requirements.
        # Report mean ± std across seeds. Single-seed eval has high variance
        # in diffusion models (seed FID gap can be 10+ points).
        # Reference: ICML 2024 guidelines; Bouthillier et al. MLSys 2021.
        print(
            f"[EVAL E{epoch:03d}][Gen] Running {len(cfg.eval_seeds)} seeds "
            f"(diagnostic only — does not affect best.pt)"
        )
        seed_results = []
        t_gen = time.time()

        for sd in cfg.eval_seeds:
            gen_psnr, gen_ssim, gen_mae = [], [], []
            n_batches = 0

            for feats, lbls, _ in self.loader_val:
                feats  = feats.float().to(self.device)
                lbls   = lbls.float().to(self.device)
                pred01 = self._ddim_sample(
                    feats,
                    steps=cfg.eval_gen_steps,
                    eta=cfg.eval_gen_eta,
                    cfg_scale=cfg.eval_cfg_scale,
                    seed=sd,
                )
                gt = lbls.clamp(0, 1).cpu().numpy()[:, 0]
                pr = pred01.cpu().numpy()[:, 0]

                for i in range(gt.shape[0]):
                    gen_psnr.append(peak_signal_noise_ratio(gt[i], pr[i], data_range=1))
                    gen_ssim.append(structural_similarity(gt[i], pr[i], data_range=1))
                    gen_mae.append(float(np.mean(np.abs(gt[i] - pr[i]))))

                n_batches += 1
                if cfg.eval_gen_batches > 0 and n_batches >= cfg.eval_gen_batches:
                    break

            seed_results.append({
                "seed":      sd,
                "psnr":      float(np.mean(gen_psnr)),
                "ssim":      float(np.mean(gen_ssim)),
                "mae":       float(np.mean(gen_mae)),
                "n_samples": len(gen_mae),
            })

        # Aggregate across seeds — mean ± std
        macro_psnr = float(np.mean([r["psnr"] for r in seed_results]))
        macro_ssim = float(np.mean([r["ssim"] for r in seed_results]))
        macro_mae  = float(np.mean([r["mae"]  for r in seed_results]))
        std_psnr   = float(np.std( [r["psnr"] for r in seed_results]))
        std_ssim   = float(np.std( [r["ssim"] for r in seed_results]))
        std_mae    = float(np.std( [r["mae"]  for r in seed_results]))

        print(
            f"[EVAL E{epoch:03d}][Gen] "
            f"MAE={macro_mae:.5f}±{std_mae:.5f}  "
            f"SSIM={macro_ssim:.4f}±{std_ssim:.4f}  "
            f"PSNR={macro_psnr:.3f}±{std_psnr:.3f}  "
            f"({time.time() - t_gen:.0f}s)"
        )
        if cfg.task == "DRC":
            print(
                f"[EVAL E{epoch:03d}] NOTE: DRC PSNR inflated by ~91% zero background. "
                f"Track MAE and SSIM."
            )

        if swapped:
            self.ema_swap.swap_out()

        # ── Logging ───────────────────────────────────────────────────────────
        self.tb_log.add_scalar("eval/val_loss",  val_loss,    epoch)
        self.tb_log.add_scalar("eval/gen_mae",   macro_mae,   epoch)
        self.tb_log.add_scalar("eval/gen_ssim",  macro_ssim,  epoch)
        self.tb_log.add_scalar("eval/gen_psnr",  macro_psnr,  epoch)
        self.tb_log.flush()

        with open(os.path.join(cfg.out_dir, "eval_last.json"), "w") as f:
            json.dump({
                "epoch":    epoch,
                "val_loss": val_loss,
                "gen": {
                    "macro": {
                        "psnr": macro_psnr, "psnr_std": std_psnr,
                        "ssim": macro_ssim, "ssim_std": std_ssim,
                        "mae":  macro_mae,  "mae_std":  std_mae,
                    },
                    "per_seed": seed_results,
                    "note": (
                        "Diagnostic only. best.pt selected by val_loss, not gen metrics. "
                        "Run sample.py on test set for final reported numbers."
                    ),
                },
                "best": {"val_loss": self.best_val_loss, "epoch": self.best_epoch},
            }, f, indent=2)

        print()  # blank line
        return val_loss
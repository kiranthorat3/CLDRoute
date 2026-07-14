#!/usr/bin/env python3
"""
diffusion.py — Single source of truth for noise schedules.

RULE: Every other file imports from here. No local copies.
"""
import math
import numpy as np


def cosine_beta_schedule(T: int, s: float = 0.008) -> np.ndarray:
    """
    Cosine schedule from Nichol & Dhariwal (Improved DDPM).
    Returns betas of shape (T,).
    """
    steps = int(T) + 1
    x = np.linspace(0, T, steps, dtype=np.float64)
    alpha_cum = np.cos(((x / T) + s) / (1.0 + s) * math.pi / 2.0) ** 2
    alpha_cum = alpha_cum / alpha_cum[0]
    betas = 1.0 - (alpha_cum[1:] / (alpha_cum[:-1] + 1e-12))
    return np.clip(betas, 1e-6, 0.999).astype(np.float32)


def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> np.ndarray:
    """
    Linear schedule from original DDPM (Ho et al. 2020).
    """
    return np.linspace(beta_start, beta_end, int(T), dtype=np.float32)


def build_betas(T: int, schedule: str) -> np.ndarray:
    """
    Unified entry point. Use this everywhere.

    Args:
        T: number of diffusion timesteps
        schedule: "cosine" or "linear"

    Returns:
        betas: np.ndarray of shape (T,)
    """
    schedule = str(schedule).strip().lower()
    if schedule == "cosine":
        return cosine_beta_schedule(T)
    elif schedule == "linear":
        return linear_beta_schedule(T)
    else:
        raise ValueError(f"Unknown beta schedule: '{schedule}'. Use 'cosine' or 'linear'.")
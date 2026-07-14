#!/usr/bin/env python3
"""
train.py — Entry point for vanilla conditional diffusion training.

Usage:
    python train.py --task congestion --tech N28
    python train.py --task DRC --tech N28 --epochs 120 --batch_size 16
    python train.py --task DRC --tech N28 --self_cond  # enable self-conditioning
"""
import os
import argparse
from config import TrainConfig
from trainer import Trainer


# ------------------------------------------------------------------
# Dataset paths (update these to match your filesystem)
# ------------------------------------------------------------------

PATHS = {
    "N28": {
        "congestion": dict(
            csv_train="/data2/kgt22001/CircuitNet-N28/training_set/congestion/files_design/train_N28.csv",
            csv_val="/data2/kgt22001/CircuitNet-N28/training_set/congestion/files_design/val_N28.csv",
            csv_test="/data2/kgt22001/CircuitNet-N28/training_set/congestion/files_design/test_N28.csv",
            feat_dir="/data2/kgt22001/CircuitNet-N28/training_set/congestion/feature",
            lbl_dir="/data2/kgt22001/CircuitNet-N28/training_set/congestion/label",
        ),
        "DRC": dict(
            csv_train="/data2/kgt22001/CircuitNet-N28/training_set/DRC/files_design/train_N28.csv",
            csv_val="/data2/kgt22001/CircuitNet-N28/training_set/DRC/files_design/val_N28.csv",
            csv_test="/data2/kgt22001/CircuitNet-N28/training_set/DRC/files_design/test_N28.csv",
            feat_dir="/data2/kgt22001/CircuitNet-N28/training_set/DRC/feature",
            lbl_dir="/data2/kgt22001/CircuitNet-N28/training_set/DRC/label",
        ),
    },
    "N14": {
        "congestion": dict(
            csv_train="/data2/kgt22001/CircuitNet-N14/training_set/congestion/files_design/train_N14.csv",
            csv_val="/data2/kgt22001/CircuitNet-N14/training_set/congestion/files_design/val_N14.csv",
            csv_test="/data2/kgt22001/CircuitNet-N14/training_set/congestion/files_design/test_N14.csv",
            feat_dir="/data2/kgt22001/CircuitNet-N14/training_set/congestion/feature",
            lbl_dir="/data2/kgt22001/CircuitNet-N14/training_set/congestion/label",
        ),
        "DRC": dict(
            csv_train="/data2/kgt22001/CircuitNet-N14/training_set/DRC/files_design/train_N14.csv",
            csv_val="/data2/kgt22001/CircuitNet-N14/training_set/DRC/files_design/val_N14.csv",
            csv_test="/data2/kgt22001/CircuitNet-N14/training_set/DRC/files_design/test_N14.csv",
            feat_dir="/data2/kgt22001/CircuitNet-N14/training_set/DRC/feature",
            lbl_dir="/data2/kgt22001/CircuitNet-N14/training_set/DRC/label",
        ),
    },
}


def build_args():
    p = argparse.ArgumentParser("Vanilla conditional diffusion — CircuitNet")
    p.add_argument("--task", required=True, choices=["congestion", "DRC"])
    p.add_argument("--tech", default="N28", choices=["N28", "N14"])
    p.add_argument("--out_root", default="./runs")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--no_ema", action="store_true")
    p.add_argument("--cfg_drop", type=float, default=0.1)
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--self_cond", action="store_true")
    p.add_argument("--pred_type", default="v", choices=["v", "eps"])
    p.add_argument("--beta_schedule", default="cosine", choices=["cosine", "linear"])
    return p.parse_args()


def main():
    args = build_args()
    paths = PATHS[args.tech][args.task]

    tag = f"vanilla_{args.task}_{args.tech}_seed{args.seed}"
    out_dir = os.path.join(args.out_root, tag)
    os.makedirs(out_dir, exist_ok=True)

    cfg = TrainConfig(
        task=args.task,
        tech=args.tech,
        out_dir=out_dir,
        csv_train=paths["csv_train"],
        csv_val=paths["csv_val"],
        csv_test=paths["csv_test"],
        feature_dir=paths["feat_dir"],
        label_dir=paths["lbl_dir"],
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        ema=not args.no_ema,
        cfg_drop_prob=args.cfg_drop,
        eval_every=args.eval_every,
        use_self_cond=args.self_cond,
        pred_type=args.pred_type,
        beta_schedule=args.beta_schedule,
        eval_seeds=[1234, 2345, 3456],
    )

    # Save config for reproducibility
    cfg.save(os.path.join(out_dir, "run_config.json"))
    print(f"[RUN] Config saved to {out_dir}/run_config.json")

    trainer = Trainer(cfg)
    trainer.train()
    trainer._evaluate(epoch=cfg.epochs)


if __name__ == "__main__":
    main()
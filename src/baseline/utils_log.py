#!/usr/bin/env python3
import csv, os, time, torch

class CSVLogger:
    def __init__(self, path, fieldnames):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.fieldnames = fieldnames
        self._fp = open(path, "w", newline="")
        self._w = csv.DictWriter(self._fp, fieldnames=fieldnames)
        self._w.writeheader()
        self._fp.flush()

    def log(self, row: dict):
        self._w.writerow(row)
        self._fp.flush()

    def close(self):
        try:
            self._fp.close()
        except Exception:
            pass

class TBLogger:
    def __init__(self, logdir):
        self.enabled = False
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(logdir=logdir)
            self.enabled = True
        except Exception:
            self.writer = None
            self.enabled = False

    def add_scalar(self, tag, value, step):
        if self.enabled:
            self.writer.add_scalar(tag, value, step)

    def add_histogram(self, tag, values, step):
        if self.enabled:
            self.writer.add_histogram(tag, values, step)

    def flush(self):
        if self.enabled:
            self.writer.flush()

    def close(self):
        if self.enabled:
            self.writer.close()

class AvgMeter:
    def __init__(self):
        self.n = 0
        self.sum = 0.0
    def update(self, v, k=1):
        self.sum += float(v) * k
        self.n += k
    @property
    def avg(self):
        return self.sum / max(1, self.n)

def grad_global_norm(parameters):
    total = 0.0
    maxn = 0.0
    for p in parameters:
        if p.grad is None: 
            continue
        g = p.grad.detach()
        val = g.norm().item()
        total += val * val
        if val > maxn: maxn = val
    return (total ** 0.5), maxn

def gpu_mem_gb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    return 0.0

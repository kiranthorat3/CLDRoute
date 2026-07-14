import torch

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = float(decay)
        self.shadow = {}
        self.collected = None  # <-- for store/restore

        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.requires_grad:
                    self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name not in self.shadow:
                self.shadow[name] = p.detach().clone()
            else:
                self.shadow[name].mul_(d).add_(p.detach(), alpha=(1.0 - d))

    @torch.no_grad()
    def copy_to(self, model):
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name in self.shadow:
                p.data.copy_(self.shadow[name].data)

    # ---------- ADD THESE TWO METHODS ----------
    @torch.no_grad()
    def store(self, model):
        """Save current (non-EMA) weights so we can restore later."""
        self.collected = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.collected[name] = p.detach().clone()

    @torch.no_grad()
    def restore(self, model):
        """Restore weights saved by store()."""
        if self.collected is None:
            return
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name in self.collected:
                p.data.copy_(self.collected[name].data)
        self.collected = None
    # ------------------------------------------

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state):
        self.decay = float(state.get("decay", self.decay))
        shadow = state.get("shadow", {})
        self.shadow = {k: v.detach().clone() for k, v in shadow.items()}

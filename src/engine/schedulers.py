"""Learning-rate schedulers used by the training engine.

KneeLRScheduler reproduces the "knee" schedule from the official Bicubic++
implementation (aselsan-research-imaging-team/bicubic-plusplus): an optional
warmup, an optional constant "explore" phase at the peak LR, and a linear decay
down to ``min_lr``. It is implemented as a standard ``LRScheduler`` subclass so
it integrates with the existing trainer/accelerate stepping (stepped once per
epoch, so ``total_steps`` should equal the number of epochs).
"""

from torch.optim.lr_scheduler import LRScheduler


class KneeLRScheduler(LRScheduler):
    def __init__(
        self,
        optimizer,
        peak_lr,
        warmup_steps=0,
        explore_steps=0,
        total_steps=1000,
        min_lr=0.0,
        last_epoch=-1,
    ):
        self.peak_lr = float(peak_lr)
        self.warmup_steps = int(warmup_steps)
        self.explore_steps = int(explore_steps)
        self.total_steps = int(total_steps)
        self.min_lr = float(min_lr)
        self.decay_steps = max(1, self.total_steps - (self.warmup_steps + self.explore_steps))
        super().__init__(optimizer, last_epoch)

    def _knee_lr(self, step):
        if self.warmup_steps > 0 and step <= self.warmup_steps:
            return self.peak_lr * step / self.warmup_steps
        if step <= (self.warmup_steps + self.explore_steps):
            return self.peak_lr
        slope = -self.peak_lr / self.decay_steps
        decayed = self.peak_lr + slope * (step - (self.warmup_steps + self.explore_steps))
        return max(self.min_lr, decayed)

    def get_lr(self):
        step = max(0, self.last_epoch)
        return [self._knee_lr(step) for _ in self.optimizer.param_groups]

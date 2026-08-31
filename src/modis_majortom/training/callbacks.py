from __future__ import annotations

import logging

import torch
from pytorch_lightning.callbacks import Callback

log = logging.getLogger(__name__)


class EMACallback(Callback):
    """EMA of trainable parameters; val/loss and inference use EMA weights.

    EMA state is saved into checkpoints under the key ``ema_shadow`` so it
    survives job restarts.  When resuming from a checkpoint that has no
    ``ema_shadow`` (e.g. a pre-EMA run), EMA initialises from the loaded
    model weights — effectively a warm-start with decay=0.9999.
    """

    def __init__(self, decay: float = 0.9999) -> None:
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self._backup: dict[str, torch.Tensor] = {}
        self._last_step = -1

    # ── initialise / restore ────────────────────────────────────────────────

    def on_train_start(self, trainer, pl_module) -> None:
        dev = next(pl_module.parameters()).device
        if not self.shadow:
            log.warning("EMACallback: no shadow found — initialising EMA from current model weights.")
            for name, param in pl_module.named_parameters():
                if param.requires_grad:
                    self.shadow[name] = param.data.detach().clone().to(dev)
        else:
            # on_load_checkpoint stores on CPU; move to training device now
            log.info("EMACallback: moving %d shadow tensors to %s.", len(self.shadow), dev)
            self.shadow = {k: v.to(dev) for k, v in self.shadow.items()}

    def on_load_checkpoint(self, trainer, pl_module, checkpoint) -> None:
        if "ema_shadow" in checkpoint:
            # Store on CPU — on_train_start moves shadows to the training device.
            # Avoid calling pl_module.parameters() here: pl_module may be None
            # in Lightning 2.x when on_load_checkpoint fires before the module
            # is fully attached to the trainer.
            self.shadow = {k: v.cpu() for k, v in checkpoint["ema_shadow"].items()}
            log.info("EMACallback: loaded %d shadow tensors from checkpoint.", len(self.shadow))
        else:
            log.warning("EMACallback: checkpoint has no 'ema_shadow' — EMA will reinitialise from raw weights.")

    def on_save_checkpoint(self, trainer, pl_module, checkpoint) -> None:
        checkpoint["ema_shadow"] = {k: v.cpu() for k, v in self.shadow.items()}

    # ── update EMA once per optimiser step ──────────────────────────────────

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if trainer.global_step == self._last_step or not self.shadow:
            return
        self._last_step = trainer.global_step
        d = self.decay
        for name, param in pl_module.named_parameters():
            if param.requires_grad and name in self.shadow:
                if self.shadow[name].device != param.device:
                    self.shadow[name] = self.shadow[name].to(param.device)
                self.shadow[name].mul_(d).add_(param.data.detach(), alpha=1.0 - d)

    # ── swap EMA weights in/out around validation ───────────────────────────

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        for name, param in pl_module.named_parameters():
            if name in self.shadow:
                self._backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        for name, param in pl_module.named_parameters():
            if name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup.clear()

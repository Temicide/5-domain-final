from __future__ import annotations

import numpy as np
import torch
from torch import nn


def _batch_inputs_targets(batch):
    if len(batch) == 3:
        inputs, targets, _ = batch
    else:
        inputs, targets = batch
    return inputs, targets


def train_one_epoch(model: nn.Module, loader, optimizer, device: torch.device, use_amp: bool) -> float:
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    losses = []
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")
    for batch in loader:
        inputs, targets = _batch_inputs_targets(batch)
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
            logits = model(inputs)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def predict_logits(model: nn.Module, loader, device: torch.device) -> np.ndarray:
    model.eval()
    outputs = []
    for batch in loader:
        inputs = batch[0]
        logits = model(inputs.to(device))
        outputs.append(logits.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)

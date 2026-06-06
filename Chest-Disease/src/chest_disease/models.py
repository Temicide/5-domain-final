from __future__ import annotations

from torch import nn

from .config import LABEL_COLUMNS, RunConfig


def _create_timm_model(model_name: str, pretrained: bool) -> nn.Module:
    import timm

    return timm.create_model(model_name, pretrained=pretrained, num_classes=len(LABEL_COLUMNS))


def _create_torchxrayvision_model() -> nn.Module:
    import torchxrayvision as xrv

    backbone = xrv.models.DenseNet(weights="densenet121-res224-all")
    in_features = backbone.classifier.in_features
    backbone.classifier = nn.Linear(in_features, len(LABEL_COLUMNS))
    return backbone


def create_model(config: RunConfig) -> nn.Module:
    if config.model_name.startswith("torchxrayvision"):
        if not config.allow_external_weights:
            return _create_timm_model("resnet18", pretrained=False)
        return _create_torchxrayvision_model()
    return _create_timm_model(config.model_name, pretrained=config.allow_external_weights)

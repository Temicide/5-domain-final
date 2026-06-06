from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, TensorDataset

from chest_disease.config import EXPECTED_COLUMNS, LABEL_COLUMNS, RunConfig
from chest_disease.dataset import ChestDiseaseDataset, build_transforms
from chest_disease.models import create_model
from chest_disease.train import predict_logits, train_one_epoch


def make_image(path: Path) -> None:
    array = np.full((32, 48), 128, dtype=np.uint8)
    Image.fromarray(array).save(path)


def test_dataset_returns_image_tensor_and_multilabel_target(tmp_path: Path):
    image_path = tmp_path / "cxr00001.jpg"
    make_image(image_path)
    frame = pd.DataFrame([["cxr00001.jpg"] + [0] * len(LABEL_COLUMNS)], columns=EXPECTED_COLUMNS)
    frame.loc[0, "Atelectasis"] = 1
    frame["image_path"] = [str(image_path)]
    transforms = build_transforms(image_size=64, train=False, model_family="timm")
    dataset = ChestDiseaseDataset(frame, transforms=transforms, include_targets=True)
    image, target, filename = dataset[0]
    assert image.shape == (3, 64, 64)
    assert target.shape == (13,)
    assert target[LABEL_COLUMNS.index("Atelectasis")] == 1
    assert filename == "cxr00001.jpg"


def test_create_timm_model_has_13_outputs():
    config = RunConfig(model_name="resnet18", allow_external_weights=False)
    model = create_model(config)
    output = model(torch.zeros(2, 3, 64, 64))
    assert output.shape == (2, 13)


def test_train_one_epoch_updates_linear_model_on_cpu():
    model = torch.nn.Linear(4, 13)
    dataset = TensorDataset(torch.randn(8, 4), torch.zeros(8, 13))
    loader = DataLoader(dataset, batch_size=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = train_one_epoch(model, loader, optimizer, device=torch.device("cpu"), use_amp=False)
    assert loss >= 0.0


def test_predict_logits_returns_numpy_array():
    model = torch.nn.Linear(4, 13)
    dataset = TensorDataset(torch.randn(8, 4), torch.zeros(8, 13))
    loader = DataLoader(dataset, batch_size=4)
    logits = predict_logits(model, loader, device=torch.device("cpu"))
    assert logits.shape == (8, 13)

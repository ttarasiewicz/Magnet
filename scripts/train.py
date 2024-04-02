import os
import importlib
from pathlib import Path
import shutil
import hydra
from hydra.utils import get_original_cwd
import wandb
import omegaconf

import torch
import torchdatasets as td
import torch_geometric as tg
from sr_core import models
from sr_core.data.dataset import PatchConfig, PatchedDataset, GraphDataset
from sr_core.constants import Subsets
from sr_core.training import runners, trainers, observers



@hydra.main(config_name="config", config_path="configs")
def main(cfg):
    run = wandb.init(project="magnet", entity="misr-polsl")
    print(run.id)
    print(run.name)
    print(run.path)
    wandb.config = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    patch_size = cfg.params.patch_size
    scale = cfg.dataset.scale

    device = f'cuda:{cfg.system.device}' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    model = getattr(importlib.import_module("sr_core.models"), cfg.model.name)()
    model.to(device)
    configurator = getattr(importlib.import_module("sr_core.data.dataset.configurators"),
                           cfg.dataset.configurator)(**cfg.dataset)
    optimizer = getattr(importlib.import_module("torch.optim"),
                           cfg.params.optimizer)(model.parameters(), lr=cfg.params.lr)
    loss_fn = getattr(importlib.import_module("sr_core.loss"),
                           cfg.params.loss)()

    graph_builder = getattr(importlib.import_module("sr_core.data.graph_builder"),
                           cfg.graph_builder.name)()

    lr_config = PatchConfig((patch_size, patch_size))
    hr_config = PatchConfig((patch_size * scale, patch_size * scale))

    print("Loading dataset")
    train_dataset = configurator.get_dataset(Subsets.TRAIN, register_lrs=cfg.dataset.register_images, register_mode='int')
    print("Train examples: ", len(train_dataset))
    train_dataset = PatchedDataset(hr_config, lr_config)(train_dataset)
    print("Patched train examples: ", len(train_dataset))
    val_dataset = configurator.get_dataset(Subsets.VALID, register_lrs=cfg.dataset.register_images, register_mode='int')
    val_dataset = PatchedDataset(hr_config, lr_config)(val_dataset)

    cache_path = Path(get_original_cwd()) / 'cache' / device.replace(':', '_')
    print(cache_path, cache_path.absolute())
    os.makedirs(cache_path, exist_ok=True)
    if os.path.isdir(cache_path) and cfg.system.recache:
        shutil.rmtree(cache_path)

    if isinstance(model, models.GraphModel):
        train_dataset = GraphDataset(train_dataset, graph_builder)
        val_dataset = GraphDataset(val_dataset, graph_builder)
        train_dataset = train_dataset.cache(td.cachers.Pickle(cache_path / 'train'))
        val_dataset = val_dataset.cache(td.cachers.Pickle(cache_path / 'val'))
        train_loader = tg.loader.DataLoader(train_dataset, batch_size=cfg.params.batch_size, shuffle=True)
        val_loader = tg.loader.DataLoader(val_dataset, batch_size=cfg.params.batch_size, shuffle=False)
    else:
        return

    wandb.watch(model, criterion=loss_fn, log="all", log_freq=100)

    train_runner = runners.TrainingEpochRunner(optimizer, criterion=loss_fn, device=device)
    train_runner.add_observer(observers.ProgressLogger("T"))
    val_runner = runners.ValidationEpochRunner(criterion=loss_fn, device=device)
    val_runner.add_observer(observers.ProgressLogger("V"))
    val_runner.add_observer(observers.FirstValidationBatchSaver())

    trainer = trainers.PyTorchTrainer(model, train_runner, val_runner)
    trainer.add_observer(observers.LossLogger())
    trainer.perform_training(cfg.params.epochs, train_loader, val_loader)

    run.finish()

if __name__ == "__main__":
    main()

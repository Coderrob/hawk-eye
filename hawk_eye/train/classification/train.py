#!/usr/bin/env python3
"""Train a model to classify images as background or targets. This script will use
distributed training if available any PyTorch AMP to speed up training."""

import argparse
import datetime
import json
import pathlib
import time
from typing import Tuple
import tarfile
import shutil
import os
import yaml

import torch
import numpy as np

from hawk_eye.core import asset_manager
from hawk_eye.core import classifier
from hawk_eye.data_generation import generate_config
from hawk_eye.train import augmentations
from hawk_eye.train.classification import dataset
from hawk_eye.train.train_utils import ema
from hawk_eye.train.train_utils import logger
from hawk_eye.train.train_utils import utils

_LOG_INTERVAL = 10
_SAVE_DIR = pathlib.Path("~/runs/uav-classifier").expanduser()


def train(
    local_rank: int,
    world_size: int,
    model_cfg: dict,
    train_cfg: dict,
    save_dir: pathlib.Path,
    initial_timestamp: str = None,
) -> None:
    """Entrypoint for training. This is where most of the logic is executed.

    Args:
        local_rank: Which GPU subprocess rank this is executed in. For CPU and 1 GPU,
            this is 0.
        world_size: How many processes are being run.
        model_cfg: The model definition dictionary.
        train_cfg: The training config dictionary.
        save_dir: Where to write checkpoints.
        initial_timestamp: Which model to start from.
    """

    # Do some general setup. When using distributed training and Apex, the device needs
    # to be set before loading the model.
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(local_rank)

    is_main = local_rank == 0
    if is_main:
        log = logger.Log(save_dir / "log.txt")

    # If we are using distributed training, initialize the backend through which process
    # can communicate to each other.
    if world_size > 1:
        torch.distributed.init_process_group(
            "nccl", world_size=world_size, rank=local_rank
        )

    # TODO(alex) these paths should be in the generate config
    batch_size = train_cfg.get("batch_size", 4)
    train_loader, train_sampler = create_data_loader(
        batch_size,
        generate_config.DATA_DIR / "clf_train",
        world_size=world_size,
        val=False,
        img_size=model_cfg.get("image_size", 224),
    )
    eval_loader, _ = create_data_loader(
        batch_size,
        generate_config.DATA_DIR / "clf_val",
        world_size=world_size,
        val=True,
        img_size=model_cfg.get("image_size", 224),
    )

    if is_main:
        log.info(f"Train dataset: {train_loader.dataset}")
        log.info(f"Val dataset: {eval_loader.dataset}")

    scores = {"best_model_score": 0, "best_ema_score": 0}
    best_scores_path = pathlib.Path(save_dir / "best_scores.json")
    best_scores_path.write_text(json.dumps({}))

    clf_model = classifier.Classifier(
        backbone=model_cfg.get("backbone", None),
        num_classes=model_cfg.get("num_classes", 2),
    )
    if initial_timestamp is not None:
        clf_model.load_state_dict(
            torch.load(initial_timestamp / "classifier.pt", map_location="cpu")
        )
    clf_model.to(device)

    if is_main:
        log.info(f"Model: \n {clf_model}")

    optimizer = utils.create_optimizer(train_cfg["optimizer"], clf_model)
    use_mixed_precision = train_cfg.get(
        "mixed_precision", train_cfg.get("mixed-precision", True)
    )
    if use_mixed_precision:
        if is_main:
            log.info("Mixed-precision (AMP) enabled.")
        scaler = torch.cuda.amp.GradScaler()
        scaler

    ema_model = ema.Ema(clf_model)

    if world_size > 1:
        clf_model = torch.nn.parallel.DistributedDataParallel(
            clf_model, device_ids=[local_rank]
        )

    epochs = train_cfg.get("epochs", 0)
    assert epochs > 0, "Please supply epoch > 0"

    # Create the learning rate scheduler.
    lr_scheduler = None
    if train_cfg["optimizer"]["type"].lower() == "sgd":
        lr_config = train_cfg.get("lr_schedule", {})
        warm_up_percent = lr_config.get("warmup_fraction", 0)
        start_lr = float(lr_config.get("start_lr"))
        max_lr = float(lr_config.get("max_lr"))
        end_lr = float(lr_config.get("end_lr"))
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=len(train_loader) * epochs,
            final_div_factor=start_lr / end_lr,
            div_factor=max_lr / start_lr,
            pct_start=warm_up_percent,
        )

    loss_fn = torch.nn.CrossEntropyLoss()
    global_step = 0

    for epoch in range(epochs):
        all_losses = []

        # Set the train loader's epoch so data will be re-shuffled.
        if world_size > 1:
            train_sampler.set_epoch(epoch)

        for idx, (data, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            global_step += 1

            # BHWC -> BCHW
            data = data.permute(0, 3, 1, 2)
            data = data.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            out = clf_model(data)

            loss = loss_fn(out, labels)
            all_losses.append(loss.item())

            # Propogate the gradients back through the model.
            if use_mixed_precision:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

            ema_model.update(clf_model)

            if idx % _LOG_INTERVAL == 0 and is_main:
                lr = optimizer.param_groups[0]["lr"]
                log.info(
                    f"Epoch: {epoch} step {idx}, loss "
                    f"{sum(all_losses) / len(all_losses):.5}. lr: {lr:.4}"
                )

        # Call evaluation function
        if is_main and epoch >= train_cfg.get("eval_start_epoch", 10):
            improved_scores = set()
            log.info("Starting eval.")
            start_val = time.perf_counter()
            clf_model.eval()
            model_score = evaluate(clf_model, eval_loader, device)
            clf_model.train()

            if model_score > scores["best_model_score"]:
                scores["best_model_score"] = model_score
                improved_scores.add("best_model_score")
                # TODO(alex): Fix this .module
                utils.save_model(clf_model, save_dir / "classifier.pt")

            ema_score = evaluate(ema_model, eval_loader, device)
            if ema_score > scores["best_ema_score"]:
                scores["best_ema_score"] = ema_score
                improved_scores.add("ema-acc")
                utils.save_model(ema_model.ema_model, save_dir / "ema-classifier.pt")

            # Write the best metrics to a file so we know which model weights to load.
            if improved_scores:
                best_scores = json.loads(best_scores_path.read_text())
                best_scores.update(scores)
                best_scores_path.write_text(json.dumps(best_scores))

            log.info(f"Eval took {time.perf_counter() - start_val:.4f}s.")
            log.info(f"Improved metrics: {improved_scores}.")
            log.info(
                f"Epoch {epoch}, Training loss {sum(all_losses) / len(all_losses):.5f}\n"
                f"Best model accuracy: {scores['best_model_score']:.5f}\n"
                f"Best EMA accuracy: {scores['best_ema_score']:.5f} \n"
            )
            log.metric("Model score", model_score, epoch)
            log.metric("Best model score", scores["best_model_score"], epoch)
            log.metric("EMA score", ema_score, epoch)
            log.metric("Best EMA score", scores["best_ema_score"], epoch)
            log.metric("Training loss", sum(all_losses) / len(all_losses), epoch)


@torch.no_grad()
def evaluate(
    clf_model: torch.nn.Module,
    eval_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """Evaluate the model against the evaulation set. Save the best
    weights if specified.

    Args:
        clf_model: The model to evaluate.
        eval_loader: The eval dataset loader.
        device: Which device to send data to.

    Returns:
        The accuracy over the evaluation set.
    """
    num_correct = total_num = 0

    for data, labels in eval_loader:
        data = data.permute(0, 3, 1, 2)
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        out = clf_model(data)
        _, predicted = torch.max(out.data, 1)

        num_correct += (predicted == labels).sum().item()
        total_num += data.shape[0]

    return num_correct / total_num


def create_data_loader(
    batch_size: dict, data_dir: pathlib.Path, world_size: int, val: bool, img_size: int
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.Sampler]:
    """This function acts as a DataLoader factory. Depending on the type of input
    data, either train or val, we construct a DataLoader which will be used during
    training. If we are using distributed training, a DistributedSampler is used to
    equally divide up the data between processes.

    Args:
        batch_size: Number of images per batch.
        data_dir: Where the images are located.
        world_size: How many training processes.
        val: Whether or not this is val or train data. This influences the sampler.

    Returns:
        The DataLoader and the sampler for the load. We need access to the sampler
            during _training_ so the data is shuffled each epoch.
    """
    assert data_dir.is_dir(), data_dir

    augmentations.clf_eval_augs(
        img_size, img_size
    ) if val else augmentations.clf_train_augs(img_size, img_size)
    clf_dataset = dataset.ClfDataset(
        data_dir,
        img_ext=generate_config.IMAGE_EXT,
        augs=augmentations.clf_eval_augs(img_size, img_size)
        if val
        else augmentations.clf_train_augs(img_size, img_size),
    )
    # If using distributed training, use a DistributedSampler to load exclusive sets
    # of data per process.
    sampler = None
    if world_size > 1:
        if not val:
            sampler = torch.utils.data.DistributedSampler(clf_dataset, shuffle=True)
        else:
            sampler = torch.utils.data.SequentialSampler(clf_dataset)

    loader = torch.utils.data.DataLoader(
        clf_dataset,
        batch_size=batch_size,
        pin_memory=True,
        sampler=sampler,
        drop_last=True,
    )
    return loader, sampler


if __name__ == "__main__":
    # Training will probably always be undeterministic due to async CUDA calls,
    # but this gets us closer to repeatability.
    torch.cuda.random.manual_seed(42)
    torch.random.manual_seed(42)
    np.random.seed(42)

    parser = argparse.ArgumentParser(
        description="Trainer code for classifcation models."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=pathlib.Path,
        help="Path to yaml model definition.",
    )
    parser.add_argument(
        "--initial_timestamp",
        default=None,
        type=str,
        help="Model timestamp to load as a starting point.",
    )
    args = parser.parse_args()

    # Download initial timestamp.
    initial_timestamp = None
    if args.initial_timestamp is not None:
        initial_timestamp = asset_manager.download_model(
            "classifier", args.initial_timestamp
        )

    config_path = args.config.expanduser()
    assert config_path.is_file(), f"Can't find {config_path}."

    # Load the model config
    config = yaml.safe_load(config_path.read_text())

    model_cfg = config["model"]
    train_cfg = config["training"]

    # Copy in this config file to the save dir. The config file will be used to load the
    # saved model.
    save_dir = _SAVE_DIR / (
        datetime.datetime.now().isoformat().split(".")[0].replace(":", ".")
    )
    save_dir.mkdir(parents=True)
    shutil.copy(config_path, save_dir / "config.yaml")

    use_cuda = torch.cuda.is_available()
    world_size = torch.cuda.device_count() if use_cuda else 1  # GPUS or a CPU

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12345"

    if world_size > 1:
        torch.multiprocessing.spawn(
            train,
            (world_size, model_cfg, train_cfg, save_dir, initial_timestamp),
            nprocs=world_size,
            join=True,
        )
    else:
        train(0, world_size, model_cfg, train_cfg, save_dir, initial_timestamp)

    # Create tar archive.
    save_archive = save_dir / f"{save_dir.name}.tar.gz"
    with tarfile.open(save_archive, mode="w:gz") as tar:
        for model_file in save_dir.glob("*"):
            tar.add(model_file, arcname=model_file.name)

    asset_manager.upload_model("classifier", save_archive)

    print(f"Saved model to {save_dir / save_dir.name}.tar.gz")

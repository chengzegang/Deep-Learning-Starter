import argparse
from functools import partial
import random
from typing import Type
from deep_learning_starter.datapipes.image_folder import ImageFolder
from deep_learning_starter.models import MaskedAutoEncoder
from deep_learning_starter.datapipes import imagenet1k
import torch
import os
import math
import torch.multiprocessing as mp
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel
from torch.distributed.device_mesh import init_device_mesh
import logging
from torch.optim import SGD, AdamW
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.dataset import IterableDataset
import torchvision.transforms.v2.functional as TF
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torch import nn
import torch.distributed.algorithms.model_averaging.averagers as averagers
from lion_pytorch import Lion
from torch.distributed.optim import PostLocalSGDOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks.post_localSGD_hook import (
    PostLocalSGDState,
    post_localSGD_hook,
)
from torch.optim import swa_utils
import glob
import webdataset as wds
from datasets import load_dataset, DownloadConfig
from torch.utils.tensorboard import SummaryWriter
from deep_learning_starter.models.vector_quantized_vae import VQVAE, VQVAE2d

logger = logging.getLogger(__name__)


def parse_args():

    parser = argparse.ArgumentParser(description="Train Masked Auto Encoder")
    parser.add_argument("--model-args", type=str, default="", help="Model arguments", nargs="*")
    parser.add_argument("--data-loader", type=str, default="webdataset", help="Data loader", choices=["webdataset", "datasets"])
    parser.add_argument("--data-dir", type=str, default="data", help="Path to the data directory")
    parser.add_argument("--image-size", type=int, default=256, help="Size of the input images")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Minimum learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--warmup-steps", type=int, default=2000, help="Number of warmup steps")
    parser.add_argument("--max-steps", type=int, default=100000, help="Number of max steps")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=16, help="Number of workers")
    parser.add_argument("--log-interval", type=int, default=100, help="Log interval")
    parser.add_argument("--save-interval", type=int, default=1000, help="Save interval")
    parser.add_argument("--model-dir", type=str, default="models/", help="Path to the model directory")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--tqdm", action="store_true", help="Use tqdm")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--dtype", type=str, default="float32", help="Data type")
    parser.add_argument("--train-dtype", type=str, default="bf16", help="Training data type")
    parser.add_argument("--ddp", action="store_true", help="Use DDP")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--log-level", type=str, default="ERROR", help="Log level")

    args = parser.parse_args()

    logger.setLevel(args.log_level)
    match args.dtype:
        case "float32":
            args.dtype = torch.float32
        case "float64":
            args.dtype = torch.float64
        case "float16":
            args.dtype = torch.float16
        case "bfloat16":
            args.dtype = torch.bfloat16

    match args.train_dtype:
        case "bf16":
            args.train_dtype = torch.bfloat16
        case "f32":
            args.train_dtype = torch.float32
        case "f16":
            args.train_dtype = torch.float16
    return args


def init_model(args):
    import deep_learning_starter.models as models

    move_to_device = torch.device(args.device)
    autoencoder = models.VAE2d(device=move_to_device, dtype=args.dtype)
    model = models.LatentDiffusionText2Image(
        autoencoder, **{k: v for k, v in (arg.split("=") for arg in args.model_args)}, device=move_to_device, dtype=args.dtype
    )
    model_path = os.path.join(args.model_dir, args.model.lower())
    if model_path is not None and os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(os.path.join(model_path, "model.pth"), map_location=move_to_device, mmap=True), assign=True)
        except Exception as e:
            short_e = str(e)[:128]
            logger.warning(f"Failed to load model from {model_path}. Starting from scratch. {short_e}")
    model.autoencoder.load_state_dict(torch.load("models/vae2d/model.pth", map_location=move_to_device, mmap=True), assign=True)
    model.autoencoder.requires_grad_(False)
    model.diffusion_model.clip.requires_grad_(False)
    model.to(memory_format=torch.channels_last)
    return model


def cosine_warmup_scheduler(optimizer, warmup_steps, max_steps, min_lr, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / warmup_steps
        elif current_step < max_steps:
            return 0.5 * (1 + math.cos(math.pi * (current_step - warmup_steps) / (max_steps - warmup_steps)))
        else:
            return min_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def transform(data, image_size):

    data["image"] = data["image"].convert("RGB")
    data["image"] = TF.pil_to_tensor(data["image"])
    data["image"] = TF.resize(data["image"], image_size, antialias=True)
    data["image"] = TF.center_crop(data["image"], (image_size, image_size))
    data["image"] = TF.to_dtype(data["image"], torch.float32, scale=True)
    return data


def load(data, image_size):
    data = Image.open(data)
    data = data.convert("RGB")
    data = TF.pil_to_tensor(data)
    data = TF.resize(data, image_size, antialias=True)
    top = random.randint(0, data.size(1) - image_size)
    left = random.randint(0, data.size(2) - image_size)
    data = TF.crop_image(data, top, left, image_size, image_size)
    data = TF.to_dtype(data, torch.float32, scale=True)
    return data


def setup(rank, world_size, args):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    logger.info(f"[rank {rank}: {world_size} started")
    args.world_size = world_size
    args.rank = rank
    args.device = torch.device(rank)
    train(args)


def spawn(args):
    if args.ddp:
        world_size = torch.cuda.device_count()
        mp.spawn(
            setup,
            args=(
                world_size,
                args,
            ),
            nprocs=world_size,
        )
    else:
        args.world_size = 1
        args.rank = 0
        train(args)


def sharding_filter(world_size, rank, data, idx):
    return idx % world_size == rank


from PIL import Image
import httpx
import io


def download(data):
    try:
        resp = httpx.get(
            data["url"],
            follow_redirects=True,
            timeout=1,
        )
        resp.raise_for_status()
        data["url"] = Image.open(io.BytesIO(resp.content))
        return data
    except Exception:
        data["url"] = None
        return data


def train(args):
    args.model = "ldm"
    model = init_model(args)
    traced_model = model

    if args.ddp:
        dist.init_process_group("nccl")
        traced_model = DDP(model.to(args.device), bucket_cap_mb=32, gradient_as_bucket_view=True, static_graph=True)
        state = PostLocalSGDState(process_group=None, subgroup=None, start_localSGD_iter=100)
        traced_model.register_comm_hook(state, post_localSGD_hook)

    local_optimizer = AdamW(traced_model.diffusion_model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=True)
    scheduler = cosine_warmup_scheduler(local_optimizer, args.warmup_steps, args.max_steps, args.min_lr / args.lr)
    if args.ddp:
        optimizer = PostLocalSGDOptimizer(local_optimizer, averager=averagers.PeriodicModelAverager(period=4, warmup_steps=100))
    else:
        optimizer = local_optimizer

    # dataset = (
    #    wds.WebDataset(glob.iglob(args.data_dir), nodesplitter=wds.shardlists.split_by_worker)
    #    .decode("torchrgb")
    #    .rename_keys(image="jpg", caption="txt")
    #    .shuffle(1000)
    # )
    dataset = (
        load_dataset("kakaobrain/coyo-700m", split="train", streaming=True)
        .filter(partial(sharding_filter, args.world_size, args.rank), with_indices=True)
        .shuffle()
        .map(download)
        .filter(lambda x: x["url"] is not None)
        .rename_column("url", "image")
        .map(partial(transform, image_size=args.image_size))
    )
    total_images = 11990000
    dl = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False if args.ddp or isinstance(dataset, IterableDataset) else True,
        num_workers=args.num_workers,
        pin_memory=True if not args.ddp else False,
        pin_memory_device=str(args.device) if not args.ddp else "",
        persistent_workers=True,
    )
    step = 0
    model_path = os.path.join(args.model_dir, args.model.lower())
    os.makedirs(model_path, exist_ok=True)

    total_samples = total_images // args.world_size // args.batch_size
    avg_model = swa_utils.AveragedModel(traced_model, device=args.device, avg_fn=swa_utils.get_ema_avg_fn(0.999), use_buffers=True)
    writer = SummaryWriter(os.path.join(model_path, "logs"))
    for epoch in range(args.epochs):

        for batch in (pbar := tqdm(dl, total=total_samples, disable=not args.tqdm or not args.rank == 0, dynamic_ncols=True)):
            traced_model.train()
            image = batch["image"].to(device=args.device, dtype=args.dtype, non_blocking=True).contiguous(memory_format=torch.channels_last)
            text = batch["text"]
            if args.ddp and step % args.grad_accum != 0:
                with traced_model.no_sync():
                    with torch.autocast("cuda", args.train_dtype, enabled=args.train_dtype != torch.float32):
                        output = traced_model(image, text)
                    (output.loss / args.grad_accum).backward()
            else:
                with torch.autocast("cuda", args.train_dtype, enabled=args.train_dtype != torch.float32):
                    output = traced_model(image, text)
                (output.loss / args.grad_accum).backward()
            if step % args.grad_accum == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, foreach=True)
                writer.add_scalar("GradNorm", grad_norm, step, new_style=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                writer.add_scalar("LR", scheduler.get_last_lr()[0], step, new_style=True)
                avg_model.update_parameters(traced_model)
            pbar.set_description(f"Epoch {epoch}, step {step}, {output.desc}, lr {optimizer.param_groups[0]['lr']:.4e}")
            writer.add_scalar("Loss", output.loss.item(), step, new_style=True)
            if step % args.log_interval == 0 and args.rank == 0:
                logger.info(f"Epoch {epoch}, step {step}, loss {output.loss.item()}")
                fig = output.plot
                fig.savefig(os.path.join(model_path, "output.png"))
                writer.add_figure("Rec", fig, step)
            if step % args.save_interval == 0 and args.rank == 0:
                model.eval()
                path = os.path.join(model_path, "model.pth")
                torch.save(model.state_dict(), path + ".tmp")
                os.replace(path + ".tmp", path)
            step += 1


if __name__ == "__main__":
    args = parse_args()
    spawn(args)

import argparse
from functools import partial
from typing import Type
from deep_learning_starter.models import MaskedAutoEncoder
from deep_learning_starter.datapipes import imagenet1k
import torch
import os
import math
import torch.multiprocessing as mp
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel
from torch.distributed.device_mesh import init_device_mesh
import logging
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.dataset import IterableDataset
import torchvision.transforms.v2.functional as TF
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torch import nn
from lion_pytorch import Lion

logger = logging.getLogger(__name__)


def parse_args():

    parser = argparse.ArgumentParser(description="Train Masked Auto Encoder")
    parser.add_argument("--model", type=str, required=True, help="Model class")
    parser.add_argument("--model-args", type=str, default="", help="Model arguments", nargs="*")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to the data directory")
    parser.add_argument("--image-size", type=int, default=256, help="Size of the input images")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Minimum learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--warmup-steps", type=int, default=2000, help="Number of warmup steps")
    parser.add_argument("--max-steps", type=int, default=100000, help="Number of max steps")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of workers")
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
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level")

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
        case _:
            raise ValueError(f"Unsupported dtype {args.dtype}")

    match args.train_dtype:
        case "bf16":
            args.train_dtype = torch.bfloat16
        case "f32":
            args.train_dtype = torch.float32
        case "f16":
            args.train_dtype = torch.float16
        case _:
            raise ValueError(f"Unsupported dtype {args.dtype}")
    return args


def init_model(args):
    import deep_learning_starter.models as models

    model_type = models.__dict__.get(args.model)
    model = model_type(**{k: v for k, v in (arg.split("=") for arg in args.model_args)}, device=args.device, dtype=args.dtype)
    model_path = os.path.join(args.model_dir, args.model.lower())
    if model_path is not None and os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(os.path.join(model_path, "model.pth"), map_location=args.device, mmap=True), assign=True)
        except Exception as e:
            short_e = str(e)[:128]
            logger.warning(f"Failed to load model from {model_path}. Starting from scratch. {short_e}")
    model.to(memory_format=torch.channels_last)
    from torch.nn.parallel import DistributedDataParallel as DDP

    traced_model = None
    if args.ddp:
        tp_mesh = init_device_mesh("cuda", (torch.cuda.device_count(),))
        traced_model = DDP(model, gradient_as_bucket_view=True, static_graph=True, device_mesh=tp_mesh)

        #def build_parrellel_plan(model):
        #    plan = {}
        #    for name, mod in model.named_modules():
        #        if isinstance(mod, (nn.Linear, nn.Embedding)):
        #            plan[name] = ColwiseParallel()
        #    return plan
        #
        #pplan = build_parrellel_plan(traced_model)
        #traced_model = parallelize_module(traced_model, tp_mesh, pplan)
    return model, traced_model


def cosine_warmup_scheduler(optimizer, warmup_steps, max_steps, min_lr, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / warmup_steps
        elif current_step < max_steps:
            return 0.5 * (1 + torch.cos(torch.tensor(current_step - warmup_steps) * (3.14159 / (max_steps - warmup_steps))))
        else:
            return min_lr / optimizer.param_groups[0]["lr"]

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def transform(data, image_size):

    data["image"] = data["image"].convert("RGB")
    data["image"] = TF.pil_to_tensor(data["image"])
    data["image"] = TF.resize(data["image"], image_size, antialias=True)
    data["image"] = TF.center_crop(data["image"], (image_size, image_size))
    data["image"] = TF.to_dtype(data["image"], torch.float32, scale=True)
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
    if rank != 0:
        args.tqdm = False
        args.save_interval = math.inf
        args.log_interval = math.inf
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

def train(args):
    model, traced_model = init_model(args)
    traced_model = traced_model or model
    optimizer = Lion(traced_model.parameters(), lr=args.lr, weight_decay=args.weight_decay, use_triton=False)
    scheduler = cosine_warmup_scheduler(optimizer, args.warmup_steps, args.max_steps, args.min_lr)

    dataset = imagenet1k(args.data_dir)
    total_images = 1281167
    dataset = (
        dataset.filter(partial(sharding_filter, args.world_size, args.rank), with_indices=True)
        .map(partial(transform, image_size=args.image_size))
        .shuffle()
    )
    dl = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False if args.ddp or isinstance(dataset, IterableDataset) else True,
        num_workers=min(args.num_workers, dataset.n_shards),
        pin_memory=True if not args.ddp else False,
        pin_memory_device=str(args.device) if not args.ddp else "",
        persistent_workers=True,
    )
    step = 0
    model_path = os.path.join(args.model_dir, args.model.lower())
    os.makedirs(model_path, exist_ok=True)

    for epoch in range(args.epochs):
        # if sampler is not None:
        #    sampler.set_epoch(epoch)
        for batch in (pbar := tqdm(dl, total=total_images // args.batch_size, disable=not args.tqdm, dynamic_ncols=True)):
            traced_model.train()
            batch = batch["image"].to(device=args.device, dtype=args.dtype, non_blocking=True).contiguous(memory_format=torch.channels_last)
            if args.ddp and step % args.grad_accum != 0:
                with traced_model.no_sync():
                    with torch.autocast("cuda", args.train_dtype, enabled=args.train_dtype != torch.float32):
                        output = traced_model(batch, batch)
                    output.loss.backward()
            else:
                with torch.autocast("cuda", args.train_dtype, enabled=args.train_dtype != torch.float32):
                    output = traced_model(batch, batch)
                output.loss.backward()
            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, foreach=True)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            pbar.set_description(f"Epoch {epoch}, step {step}, {output.desc}, lr {optimizer.param_groups[0]['lr']:.4e}")

            if step % args.log_interval == 0:
                logger.info(f"Epoch {epoch}, step {step}, loss {output.loss.item()}")
                fig = output.plot
                fig.savefig(os.path.join(model_path, "output.png"))
                plt.close(fig)
            if step % args.save_interval == 0:
                model.eval()
                path = os.path.join(model_path, "model.pth")
                torch.save(model.state_dict(), path + ".tmp")
                os.replace(path + ".tmp", path)
            step += 1


if __name__ == "__main__":
    args = parse_args()
    spawn(args)

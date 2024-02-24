import argparse
from functools import partial
from deep_learning_starter.models import MaskedAutoEncoder
from deep_learning_starter.datapipes import imagenet1k
import torch
import os
import logging
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.dataset import IterableDataset
import torchvision.transforms.v2.functional as TF
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def parse_args():

    parser = argparse.ArgumentParser(description="Train Masked Auto Encoder")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to the data directory")
    parser.add_argument("--image-size", type=int, default=256, help="Size of the input images")
    parser.add_argument("--mask-ratio", type=float, default=0.75, help="Ratio of the input to mask")
    parser.add_argument("--patch-size", type=int, default=16, help="Size of the patch")
    parser.add_argument("--hidden-size", type=int, default=768, help="Size of the hidden layer")
    parser.add_argument("--head-size", type=int, default=64, help="Size of the head")
    parser.add_argument("--num-heads", type=int, default=768 // 64, help="Number of heads")
    parser.add_argument("--num-encoder-layers", type=int, default=16, help="Number of encoder layers")
    parser.add_argument("--num-decoder-layers", type=int, default=8, help="Number of decoder layers")
    parser.add_argument("--eps", type=float, default=1e-5, help="Epsilon for RMSNorm")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Minimum learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--warmup-steps", type=int, default=2000, help="Number of warmup steps")
    parser.add_argument("--max-steps", type=int, default=100000, help="Number of max steps")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of workers")
    parser.add_argument("--log-interval", type=int, default=100, help="Log interval")
    parser.add_argument("--save-interval", type=int, default=1000, help="Save interval")
    parser.add_argument("--model-path", type=str, default="models/mae", help="Path to the model")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--tqdm", action="store_true", help="Use tqdm")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--dtype", type=str, default="float32", help="Data type")
    parser.add_argument("--train-dtype", type=str, default="bf16", help="Training data type")
    parser.add_argument("--ddp", action="store_true", help="Use DDP")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")

    args = parser.parse_args()
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
    model = MaskedAutoEncoder(
        args.mask_ratio,
        args.patch_size,
        args.hidden_size,
        args.head_size,
        args.num_heads,
        args.num_encoder_layers,
        args.num_decoder_layers,
        args.eps,
        device=args.device,
        dtype=args.dtype,
    )
    if args.model_path is not None and os.path.exists(args.model_path):
        try:
            model.load_state_dict(torch.load(os.path.join(args.model_path, "model.pth"), map_location=args.device, mmap=True), assign=True)
        except Exception as e:
            short_e = str(e)[:128]
            logger.warning(f"Failed to load model from {args.model_path}. Starting from scratch. {short_e}")
    return model


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


def show_result(pred, target):

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(TF.to_pil_image(pred))
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(TF.to_pil_image(target))
    fig.tight_layout()
    return fig


def train(args):
    model = init_model(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=True)
    scheduler = cosine_warmup_scheduler(optimizer, args.warmup_steps, args.max_steps, args.min_lr)

    dataset = imagenet1k(args.data_dir)
    total_images = 1281167
    dataset = dataset.map(partial(transform, image_size=args.image_size))
    sampler = DistributedSampler(dataset) if args.ddp else None
    dl = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False if args.ddp or isinstance(dataset, IterableDataset) else True,
        num_workers=args.num_workers,
        pin_memory=True,
        pin_memory_device=args.device,
        persistent_workers=True,
        sampler=sampler,
    )
    step = 0
    os.makedirs(args.model_path, exist_ok=True)
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in (pbar := tqdm(dl, total=total_images // args.batch_size, disable=not args.tqdm, dynamic_ncols=True)):
            model.train()
            with torch.autocast("cuda", args.train_dtype, enabled=args.train_dtype != torch.float32):
                batch = (
                    batch["image"].to(device=args.device, dtype=args.dtype, non_blocking=True).contiguous(memory_format=torch.channels_last)
                )
                output = model(batch, batch)
            output.loss.backward()
            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            pbar.set_description(f"Epoch {epoch}, step {step}, loss {output.loss.item():.8f}, lr {optimizer.param_groups[0]['lr']:.4e}")

            if step % args.log_interval == 0:
                logger.info(f"Epoch {epoch}, step {step}, loss {output.loss.item()}")
                fig = show_result(output.logits[0].cpu().detach().float().clamp(0, 1), batch[0].cpu().detach().float())
                fig.savefig(os.path.join(args.model_path, "output.png"))
                plt.close(fig)
            if step % args.save_interval == 0:
                model.eval()
                path = os.path.join(args.model_path, "model.pth")
                torch.save(model.state_dict(), path + ".tmp")
                os.replace(path + ".tmp", path)
            step += 1


if __name__ == "__main__":
    args = parse_args()
    train(args)

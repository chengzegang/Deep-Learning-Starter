from datasets import load_dataset

# If the dataset is gated/private, make sure you have run huggingface-cli login


def imagenet1k(cache_dir: str):
    dataset = load_dataset("imagenet-1k", split="train", cache_dir="data", streaming=True, trust_remote_code=True)
    return dataset

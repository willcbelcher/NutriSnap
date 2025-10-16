import os
import numpy as np
from PIL import Image

from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    set_seed,
)


def create_transforms(processor):
    def transforms(examples):
        # Ensure PIL RGB
        images = []
        for img in examples["image"]:
            if hasattr(img, "convert"):
                images.append(img.convert("RGB"))
            else:
                images.append(Image.open(img).convert("RGB"))

        inputs = processor(images=images, return_tensors="pt")
        inputs["labels"] = examples["label"]
        return inputs
    return transforms


def main():
    set_seed(42)
    print("Running preprocess.py")

    # -------------------------
    # 0) Fast dev run toggles
    # -------------------------
    FAST_DEV_RUN = os.getenv("FAST_DEV_RUN", "1").lower() in {"1", "true", "yes"}
    TRAIN_SAMPLES = int(os.getenv("TRAIN_SAMPLES", "200"))
    EVAL_SAMPLES = int(os.getenv("EVAL_SAMPLES", "200"))

    if FAST_DEV_RUN:
        print("Fast dev run enabled")
    else:
        print("Full run enabled")

    # -------------------------
    # 1) Data & preprocessing
    # -------------------------
    print("Loading Food101 dataset...")
    ds = load_dataset("ethz/food101")
    id2label = {i: c for i, c in enumerate(ds["train"].features["label"].names)}
    label2id = {c: i for i, c in id2label.items()}

    # Small subsample for quick local test, or use the full splits
    if FAST_DEV_RUN:
        train_ds = ds["train"].shuffle(seed=42).select(range(min(TRAIN_SAMPLES, len(ds["train"]))))
        eval_ds = ds["validation"].shuffle(seed=42).select(range(min(EVAL_SAMPLES, len(ds["validation"]))))
    else:
        train_ds = ds["train"]
        eval_ds = ds["validation"]

    model_ckpt = "google/vit-base-patch16-224-in21k"
    processor = AutoImageProcessor.from_pretrained(model_ckpt)
    transforms = create_transforms(processor)

    # -------------------------
    # 2) Save processed datasets (without transforms)
    # -------------------------
    print("Saving processed datasets to shared volume...")
    train_ds.save_to_disk("/app/data/train")
    eval_ds.save_to_disk("/app/data/eval")
    
    # Save transform function separately for training script
    import pickle
    with open("/app/data/transforms.pkl", "wb") as f:
        pickle.dump(transforms, f)
    
    # Save metadata for training script
    metadata = {
        "id2label": id2label,
        "label2id": label2id,
        "model_ckpt": model_ckpt,
        "fast_dev_run": FAST_DEV_RUN,
        "train_samples": len(train_ds),
        "eval_samples": len(eval_ds)
    }
    
    import json
    with open("/app/data/metadata.json", "w") as f:
        json.dump(metadata, f)
    
    print(f"Preprocessing complete! Saved {len(train_ds)} train and {len(eval_ds)} eval samples.")


if __name__ == "__main__":
    main()
import os
os.environ.setdefault("WANDB_SILENT", "true")
os.environ.setdefault("FAST_DEV_RUN", "0")

import numpy as np
from PIL import Image

import torch
from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
    set_seed,
)
import evaluate
import wandb


def main():
    set_seed(42)
    print("Running train.py")

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

    # Apply on-the-fly transforms
    train_ds = train_ds.with_transform(transforms)
    eval_ds = eval_ds.with_transform(transforms)

    # -------------------------
    # 2) Model
    # -------------------------
    model = AutoModelForImageClassification.from_pretrained(
        model_ckpt,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )

    # -------------------------
    # 3) Metrics
    # -------------------------
    acc = evaluate.load("accuracy")
    acc5 = evaluate.load("accuracy")

    def compute_metrics(p):
        # p.predictions can be (loss, logits) or logits
        logits = p.predictions[0] if isinstance(p.predictions, (tuple, list)) else p.predictions
        y_true = p.label_ids

        # top-1
        y_pred = np.argmax(logits, axis=1)
        top1 = float((y_pred == y_true).mean())

        # top-5: check if the true label is among the top-5 logits
        # (argsort descending then take first 5)
        top5_idx = np.argsort(-logits, axis=1)[:, :5]
        top5 = float(np.any(top5_idx == y_true[:, None], axis=1).mean())

        return {"top1": top1, "top5": top5}

    # -------------------------
    # 4) TrainingArguments
    # -------------------------
    # Auto device: CUDA if available; otherwise use MPS on Apple; else CPU
    use_cuda = torch.cuda.is_available()
    use_mps = (not use_cuda) and torch.backends.mps.is_available()

    # Light defaults for quick run; bump for full training
    args = TrainingArguments(
        output_dir="./food101-vit",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        num_train_epochs=1 if FAST_DEV_RUN else 3,
        weight_decay=0.0,
        logging_steps=10,
        remove_unused_columns=False,
        fp16=use_cuda,              # mixed precision on CUDA
        no_cuda=not use_cuda,       # False if CUDA, True otherwise
        dataloader_num_workers=0,   # adjust as needed
        # removed: evaluation_strategy, save_strategy, report_to, run_name, use_mps_device, bf16
        # evaluation_strategy="epoch",
        # save_strategy="no",  # final save below via save_pretrained
        # report_to=["wandb"], 
        # run_name="vit-base-ft",
        # use_mps_device=use_mps,     # Apple Silicon
        # bf16=False,                 # enable if you have bf16 support
    )

    # Data collator handles stacking of pixel_values/labels from transforms
    data_collator = DefaultDataCollator()

    # -------------------------
    # 5) W&B (minimal setup)
    # -------------------------
    wandb.init(
        project="NutriSnap",
        name="vit-base-ft" + ("-fast" if FAST_DEV_RUN else "-full"),
        mode=os.getenv("WANDB_MODE", "online"),
        config={
            "fast_dev_run": FAST_DEV_RUN,
            "train_samples": len(train_ds),
            "eval_samples": len(eval_ds),
            "model_ckpt": model_ckpt,
        },
    )

    # -------------------------
    # 6) Train
    # -------------------------
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=processor,          # so it's saved with the trainer
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # -------------------------
    # 7) Final eval + save
    # -------------------------
    metrics = trainer.evaluate()
    wandb.log(metrics)

    save_dir = "./food101-vit-model"
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)

    artifact = wandb.Artifact(
        name="food101-vit-model",
        type="model",
        description="ViT fine-tuned on Food101",
        metadata={"model_ckpt": model_ckpt, "num_labels": len(id2label)},
    )
    artifact.add_dir(save_dir)
    wandb.log_artifact(artifact)
    wandb.run.summary.update(metrics)
    wandb.finish()


if __name__ == "__main__":
    main()

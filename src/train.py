import os
import sys
import time
from datetime import datetime

# Ensure output is flushed immediately
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

os.environ.setdefault("WANDB_SILENT", "false")

from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
import evaluate, numpy as np

from PIL import Image
import wandb


def train():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting NutriSnap training...")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading Food101 dataset...")
    
    # 1) Data & preprocessing
    ds = load_dataset("ethz/food101")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Dataset loaded successfully!")
    
    id2label = {i: c for i, c in enumerate(ds["train"].features["label"].names)}
    label2id = {c: i for i, c in id2label.items()}
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Number of classes: {len(id2label)}")

    model_ckpt = "google/vit-base-patch16-224-in21k"
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading model checkpoint: {model_ckpt}")
    processor = AutoImageProcessor.from_pretrained(model_ckpt, use_fast=True)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Image processor loaded successfully!")

    # Init W&B (minimal setup)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Initializing Weights & Biases...")
    wandb.init(
        project="NutriSnap",
        name="vit-base-ft",
        mode=os.getenv("WANDB_MODE", "online"),
    )
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] W&B initialized successfully!")

    def transforms(examples):
        # Dataset schema confirms columns: ['image', 'label']
        raw_images = examples["image"]

        images = []
        for img in raw_images:
            if hasattr(img, "convert"):
                images.append(img.convert("RGB"))
            else:
                images.append(Image.open(img).convert("RGB"))

        inputs = processor(images=images, return_tensors="pt")
        inputs["labels"] = examples["label"]
        return inputs

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Preparing datasets with transforms...")
    prepared = {}
    for split in ds.keys():
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processing {split} split...")
        prepared[split] = ds[split].with_transform(transforms)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Dataset preparation completed!")

    # 2) Model
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading Vision Transformer model...")
    model = AutoModelForImageClassification.from_pretrained(
        model_ckpt,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Model loaded successfully!")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 3) Metrics
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading evaluation metrics...")
    acc = evaluate.load("accuracy")
    acc5 = evaluate.load("accuracy")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Metrics loaded successfully!")

    def compute_metrics(p):
        logits = p.predictions
        top1 = acc.compute(predictions=np.argmax(logits, axis=1), references=p.label_ids)["accuracy"]
        top5 = acc5.compute(
            predictions=np.argsort(logits, axis=1)[:, -5:], references=p.label_ids, top_k=5
        )["accuracy"]
        return {"top1": top1, "top5": top5}

    # 4) Train
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Setting up training arguments...")
    args = TrainingArguments(
        output_dir="./food101-vit",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        num_train_epochs=1,
        weight_decay=0.0,
        no_cuda=True,
        fp16=False,
        logging_steps=10,  # More frequent logging
        report_to=["wandb"],
        run_name="vit-base-ft",
        remove_unused_columns=False,
        save_strategy="steps",
        save_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_top1",
        greater_is_better=True,
    )
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training arguments configured!")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Batch size: {args.per_device_train_batch_size}, Epochs: {args.num_train_epochs}, Learning rate: {args.learning_rate}")

    # Use validation split
    _eval_split = "validation"
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Using {_eval_split} split for evaluation")

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=prepared["train"],
        eval_dataset=prepared[_eval_split],
        tokenizer=processor,
        compute_metrics=compute_metrics,
    )
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Trainer initialized successfully!")
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting training...")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training will log progress every {args.logging_steps} steps")
    start_time = time.time()
    
    trainer.train()
    
    end_time = time.time()
    training_duration = end_time - start_time
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training completed in {training_duration:.2f} seconds ({training_duration/60:.2f} minutes)")

    # 5) Push or save
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Evaluating model...")
    metrics = trainer.evaluate()
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Final evaluation metrics: {metrics}")
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Logging metrics to W&B...")
    wandb.log(metrics)
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Saving model to ./food101-vit-model...")
    model.save_pretrained("./food101-vit-model")
    processor.save_pretrained("./food101-vit-model")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Model saved successfully!")
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Creating W&B artifact...")
    artifact = wandb.Artifact(
        name="food101-vit-model",
        type="model",
        description="ViT fine-tuned on Food101",
        metadata={"model_ckpt": model_ckpt, "num_labels": len(id2label)},
    )
    artifact.add_dir("./food101-vit-model")
    wandb.log_artifact(artifact)
    wandb.run.summary.update(metrics)
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Finalizing W&B run...")
    wandb.finish()
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training pipeline completed successfully!")


if __name__ == "__main__":
    train()



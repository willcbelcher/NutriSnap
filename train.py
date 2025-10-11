import os
os.environ.setdefault("WANDB_SILENT", "true")

from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
import evaluate, numpy as np

from PIL import Image
import wandb


def train():
    # 1) Data & preprocessing
    ds = load_dataset("ethz/food101")
    id2label = {i: c for i, c in enumerate(ds["train"].features["label"].names)}
    label2id = {c: i for i, c in id2label.items()}

    model_ckpt = "google/vit-base-patch16-224-in21k"
    processor = AutoImageProcessor.from_pretrained(model_ckpt, use_fast=True)

    # Init W&B (minimal setup)
    wandb.init(
        project="NutriSnap",
        name="vit-base-ft",
        mode=os.getenv("WANDB_MODE", "online"),
    )

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

    prepared = {}
    for split in ds.keys():
        prepared[split] = ds[split].with_transform(transforms)

    # 2) Model
    model = AutoModelForImageClassification.from_pretrained(
        model_ckpt,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )

    # 3) Metrics
    acc = evaluate.load("accuracy")
    acc5 = evaluate.load("accuracy")

    def compute_metrics(p):
        logits = p.predictions
        top1 = acc.compute(predictions=np.argmax(logits, axis=1), references=p.label_ids)["accuracy"]
        top5 = acc5.compute(
            predictions=np.argsort(logits, axis=1)[:, -5:], references=p.label_ids, top_k=5
        )["accuracy"]
        return {"top1": top1, "top5": top5}

    # 4) Train
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
        logging_steps=50,
        report_to=["wandb"],
        run_name="vit-base-ft",
        remove_unused_columns=False,
    )

    # Use validation split if available; otherwise fall back to test
    _eval_split = "validation" if "validation" in prepared else "test"

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=prepared["train"],
        eval_dataset=prepared[_eval_split],
        tokenizer=processor,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # 5) Push or save
    metrics = trainer.evaluate()
    wandb.log(metrics)
    model.save_pretrained("./food101-vit-model")
    processor.save_pretrained("./food101-vit-model")
    artifact = wandb.Artifact(
        name="food101-vit-model",
        type="model",
        description="ViT fine-tuned on Food101",
        metadata={"model_ckpt": model_ckpt, "num_labels": len(id2label)},
    )
    artifact.add_dir("./food101-vit-model")
    wandb.log_artifact(artifact)
    wandb.run.summary.update(metrics)
    wandb.finish()


if __name__ == "__main__":
    train()



# AC215 - Milestone2 - NutriSnap

**Team Members**
William Belcher, Prakrit Baruah, Vineet Jammalamadaka

**Group Name**
NutriSnap

**Project**
The project aims to simplify the process of food tracking and nutritional analysis by replacing cumbersome manual data entry with a seamless, AI-powered system that accepts multi-modal input like photos and voice.

### Milestone2

In this milestone, we have the components for data management, including versioning, as well as the computer vision and language models.

**Data**
We use the Food 101 dataset from HuggingFace for
- Size and scope: 101,000 RGB food images across 101 classes (≈1,000 per class).
- Standard split: ~75,750 for training and ~25,250 for validation/test (per class: 750 train, 250 val/test).
- Source: Collected from Foodspotting; images vary in resolution and background context.
- Typical use: Benchmark for image classification and transfer learning (often resized to 224×224, normalized with ImageNet stats).
- Evaluation: Commonly reported with top-1 accuracy on the validation/test split.

**Model Finetuning Overview**
## Training setup
- No layers are frozen. We use full fine-tuning.
- TrainingArguments: batch size 64 (train) / 64 (eval), lr=5e-5, epochs = 1 (fast) or 3 (full), weight decay 0.0.
- Mixed precision (fp16) on CUDA; otherwise CPU. MPS detection exists but not enabled.
- No periodic eval/checkpointing configured during training; final eval after training.
- Default LR scheduler (linear, no warmup) implied.
- Experiment with finetuning both `google/vit-base-patch16-224-in21k` and `facebook/deit-tiny-patch16-224` 

## Metrics
- Custom compute_metrics: reports Top-1 and Top-5 accuracy from logits.
- Initializes Weights & Biases run with minimal config.

## Results
See the training loss curve at: `docs/train_loss.png`
Best results are from model `facebook/deit-tiny-patch16-224`
- eval_top_1: 0.613
- eval_top_5: 0.866

where top_*n* means the true label is in the top *n* predictions from the model

**Data Pipeline**

## Data Pipeline Overview

1. **`src/preprocess.py`**
   This script handles preprocessing the Food 101 dataset. It creates label maps, initializes the `AutoImageProcessor`. Image preprocessing involves reducing image sizes and normalizing image contents. Converts all images to RGB.

2. **`src/train_only.py`**
   Finetunes models using the preprocessed image data. Finetuning targets are Vision Transformers such as google VIT.

3. **`src/train.py `**
   A combination of (1) and (2) which runs end-to-end

## Data Pipeline Overview
**`src/Dockerfile.train`, `src/Dockerfile.preprocess`**
Our Dockerfiles used for running the train and preprocessing scripts. 

## Running

The following commands run the pipeline:

```bash
docker-compose up -d
docker exec -it ns-preprocess bash -c "source /home/app/.venv/bin/activate && bash"
python preprocess.py

# to reset containers after code changes - should be a faster way but this just works for now
docker-compose down && docker-compose build && docker-compose up -d
```

## App Mockup

[Here](https://www.figma.com/proto/Ztdsl6iNBXV3wxQly5oRDY/Tummy?node-id=117-429&t=Cqv92EjHamGnqijE-1) is a link to our Figma mockup of a potential prototype of this application.

## Artifacts

In the `docs` folder we have uploaded screenshots to satisfy the requirements for milestone 2. For objective 1, the virtual environments, the relevant image is `environment.jpeg`. For the containerized pipeline, the `modelrun.jpeg` files show the output of our model running on a small set of data. The full model is being trained in GCP. The pipeline is split into preprocessing and training steps.

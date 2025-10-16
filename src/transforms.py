from transformers import AutoImageProcessor
from PIL import Image


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
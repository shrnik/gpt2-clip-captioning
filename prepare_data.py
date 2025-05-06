# flickr30k dataset and create embeddings for each image using CLIP. save captions and embeddings in a json file

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset
import pickle
from pathlib import Path


def load_flickr30k_dataset():
    dataset = load_dataset("nlphuji/flickr30k")
    return dataset

# load preprocessor from clip


def load_clip_processor():
    model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_name)
    return processor
# load model from clip


def load_ms_coco_dataset():
    dataset = load_dataset("shunk031/MSCOCO",
                           year=2017,
                           coco_task="captions",)
    return dataset
# load preprocessor from clip


def load_clip_model():
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    return model
# preprocess image


def preprocess_image(image_path, processor):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    return inputs


def create_clip_embeddings():
    """
    Create CLIP image embeddings for Flickr30K dataset and save them to a pickle file
    along with their corresponding captions.
    """
    # Create output directory if it doesn't exist
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    # Load dataset, model and processor
    print("Loading Flickr30K dataset...")
    dataset = load_ms_coco_dataset()["train"]

    print("Loading CLIP model and processor...")
    processor = load_clip_processor()
    model = load_clip_model()

    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Prepare data storage
    embeddings = []
    captions = []

    # Process images in batches
    batch_size = 32
    save_frequency = 32*64  # Save after processing this many images
    total_processed = 0
    len_dataset = len(dataset)
    print(f"Processing {len(dataset)} images...")

    for i in tqdm(range(0, len_dataset, batch_size)):
        batch = dataset[i:i+batch_size]
        batch_images = batch["image"]
        batch_inputs = processor(
            images=batch_images, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            # Extract image features
            image_features = model.get_image_features(**batch_inputs)

            # Normalize embeddings
            normalized_embeddings = image_features / \
                image_features.norm(dim=1, keepdim=True)

            # Move to CPU for storage
            embeddings.extend(normalized_embeddings.cpu().numpy())

        # Store all captions (each image in Flickr30K has multiple captions)
        for caption in batch["caption"]:
            captions.append(caption)
            print(f"Processed caption: {caption}", len(caption))

         # Save intermediate results every 1000 images

        total_processed += len(batch)
        if total_processed % save_frequency == 0 and total_processed > 0:
            # Concatenate current embeddings
            current_embeddings = np.vstack(embeddings)

            # Create dictionary with data
            interim_data = {
                "embeddings": current_embeddings,
                "captions": captions
            }

            # Save to interim pickle file
            interim_path = output_dir / \
                f"coco_clip_embeddings_interim.pkl"
            print(f"\nSaving interim results to {interim_path}...")
            with open(interim_path, "wb") as f:
                pickle.dump(interim_data, f)
            print(
                f"Saved {len(captions)} captions and {current_embeddings.shape[0]} embeddings.")

    # Concatenate all embeddings
    all_embeddings = np.vstack(embeddings)

    # Create dictionary with data
    data = {
        "embeddings": all_embeddings,
        "captions": captions
    }

    # Save to pickle file
    output_path = output_dir / "coco_clip_embeddings.pkl"
    print(f"Saving embeddings to {output_path}...")
    with open(output_path, "wb") as f:
        pickle.dump(data, f)

    print(
        f"Done! Saved {len(captions)} captions and {all_embeddings.shape[0]} embeddings.")
    return output_path


if __name__ == "__main__":
    create_clip_embeddings()

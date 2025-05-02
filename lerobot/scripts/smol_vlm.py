import logging
import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from PIL import Image

import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from transformers import AutoProcessor, AutoModelForImageTextToText

# TODO: ideas
"""
Create a multi-step agent model :
First get task description and video. 
If task description is not informative enough or do not exist, use the video to improve it:
    - First selected few frames from the dataset and identify what are the objects in the scene with a first agent.
    - Then, generate a prompt for a second agent that will be used to generate the task description.
"""

def improve_task_description(
        images: list, 
        current_task: str, 
        model: AutoModelForImageTextToText,
        processor: AutoProcessor,
    ) -> str:
    """Use SmolVLM2 to improve the task description based on images and original task."""

    # Build the text instruction
    instruction = (
f"""
You are a robot capable of performing tasks in the real world. Your goal is to infer the action performed in an environment based on the video.
Here is the current task: {current_task}

Your task is to deduce the specific action that was executed. The action should be described as a command that a robot can execute, 
written in the imperative form as a very short, clear, and complete one-sentence instruction (max 30 characters).
Start directly with an action verb like 'Pick', 'Place', 'Open', etc.
Do not include unnecessary words. Be concise.
#######
Here are some examples of the desired format:
    "Pick up the red ball and place it in the blue box on the right."
    "Put the Lego piece on the right into the top drawer."
    "Push the red button located in the middle of the table."
Provide 5 examples of the task being performed based on the images (if you change, otherwise only one). Do not include any additional information.
"""
    )

    # Only text goes into the chat template
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "path": "exemple.mp4"},
                {"type": "text", "text": instruction},
            ]
        }
    ]

    # Preprocess text input
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    # Preprocess images separately
    processed_images = processor(images=images, return_tensors="pt").pixel_values.to(
        model.device, dtype=torch.bfloat16
    )

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            pixel_values=processed_images,  # Pass images separately here
            do_sample=False,
            max_new_tokens=64,
        )
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    raw_output = generated_texts[0]

    # POST-PROCESS: Extract only the Assistant's answer
    if "Assistant:" in raw_output:
        cleaned_output = raw_output.split("Assistant:")[-1].strip()
    else:
        cleaned_output = raw_output.strip()

    # Final cleanup:
    if not cleaned_output.endswith("."):
        cleaned_output += "."

    return cleaned_output

def preprocess_image(image_tensor: torch.Tensor) -> Image.Image:
    """Convert a tensor (C, H, W) to a PIL Image for SmolVLM2."""
    image = image_tensor.cpu().permute(1, 2, 0).numpy()  # (H, W, C)
    image = (image * 255).astype("uint8")
    return Image.fromarray(image)

def main(args):
    logging.basicConfig(level=logging.INFO)

    #### VLM MODEL
    # Load SmolVLM2 model
    # model_path = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    model_path = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    logging.info(f"Loading {model_path}...")
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        #_attn_implementation="flash_attention_2"
    ).to("cuda")
    logging.info("Model loaded.")

    ### DATA CURATION
    # Load all datasets
    df = pd.read_csv(args.csv_path)
    improved_tasks = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Improving tasks"):
        repo_id = row["repo_id"]
        logging.info(f"Processing {repo_id}...")
        # Load current dataset
        dataset = LeRobotDataset(
            repo_id,
            root=args.root,
            revision=args.revision,
        )

        images = []
        # Get current dataset images (from the first episode)
        sample = dataset[0]
        for key, img_tensor in sample.items():
            if key.startswith("observation.images."):
                img = preprocess_image(img_tensor)
                images.append(img)
        if not images:
            logging.warning(f"No images found for repo {repo_id}")
            improved_tasks.append(row.get("tasks", ""))
            continue
        
        # Get current dataset task description
        original_task = row.get("tasks", "").strip()
        if not original_task:
            original_task = "unknown task"
        elif isinstance(original_task, str):
            original_task = json.loads(original_task)
        elif isinstance(original_task, dict):
            original_task = original_task
        else:
            raise ValueError("Unexpected type for task field.")

        improved_task = improve_task_description(images, original_task, model, processor)
        print(improved_task)
        # improved_tasks.append(improved_task)
    # df["improved_task"] = improved_tasks
    # output_path = Path(args.output_path)
    # df.to_csv(output_path, index=False)
    # logging.info(f"Saved updated CSV to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Improve task descriptions in current_dataset.csv")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to current_dataset.csv")
    parser.add_argument("--root", type=Path, default=None, help="Local cache root")
    parser.add_argument("--output_path", type=str, default="current_dataset_semi_automatic_descriptions.csv", help="Output CSV path")    
    args = parser.parse_args()
    main(args)
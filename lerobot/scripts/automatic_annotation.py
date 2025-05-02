from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import argparse
import cv2
import torch
import os
from mistralai import Mistral
import io
import base64
import torchvision.transforms as transforms
import time


def tensor_to_base64(tensor):
    try:
        to_pil = transforms.ToPILImage()
        pil_image = to_pil(tensor)
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_bytes = buffered.getvalue()

        # Encode the bytes to a base64 string
        base64_image = base64.b64encode(img_bytes).decode('utf-8')
        return base64_image
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automatic annotation script")
    parser.add_argument(
        "--list_dataset_name",
        type=str,
        help="Name of the dataset to be annotated.",
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        default="dataset.csv",
        help="Path to the dataset file.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Whether to visualize the images.",
    )

    args = parser.parse_args()

    assert args.list_dataset_name is not None or args.dataset_file is not None, "Please provide a dataset name or a dataset file."
    if args.dataset_file is not None:
        # Load the dataset file and get the dataset names in pd.DataFrame
        import pandas as pd
        df = pd.read_csv(args.dataset_file)
        # data_list = df["repo_id"].tolist()
        # Get 6 random samples from the dataset
        data_list = df.sample(n=6)["repo_id"].tolist()
    else:
        data_list = args.list_dataset_name.split(",")

    api_key = os.environ["MISTRAL_API_KEY"]
    model = "mistral-small-latest" # "pixtral-12b-2409"
    client = Mistral(api_key=api_key)

    final_annotations = {}

    print(f">> Begin automatic annotation for {data_list}")
    print(f">> Using model {model}")
    episode_index = 0
    for dataset_name in data_list:
        print(f">>>> Annotating {dataset_name}")
        before_load = time.time()
        dataset = LeRobotDataset(
            repo_id=f"{dataset_name}",
            episodes=[episode_index],
        )
        task_annotation = dataset.meta.tasks[0]
        print(f">>>> Task annotation: {task_annotation}")
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        camera_key = dataset.meta.camera_keys[0]
        middle_frame_idx = (from_idx + to_idx) // 2
        first_frame, middle_frame, last_frame = dataset[from_idx][camera_key], dataset[middle_frame_idx][camera_key], dataset[to_idx-1][camera_key]
        after_load = time.time()
        print(f">>>> Time to load dataset: {after_load - before_load:.2f} seconds")
        base64_images = []
        for frame in [first_frame, middle_frame, last_frame]:
            base64_image = tensor_to_base64(frame)
            base64_images.append(base64_image)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""
You are a robot capable of performing tasks in the real world. Your goal is to infer the action performed in an environment based on three images.

Your task is to deduce the specific action that was executed. The action should be described as a command that a robot can execute, written in the imperative form as a single sentence. Be as precise as possible.
You have the following annotation that might be wrong, if it is, please correct it, otherwise improve it if necessary:
{task_annotation}
#######
Here are some examples of the desired format:
    "Pick up the red ball and place it in the blue box on the right."
    "Put the Lego piece on the right into the top drawer."
    "Push the red button located in the middle of the table."
Provide 5 examples of the task being performed based on the images (if you change, otherwise only one). Do not include any additional information.

Images:
""",
                    },
                ]+[
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{base64_image}"
                    } for base64_image in base64_images
                ]
            }
        ]
        before_chat = time.time()
        chat_response = client.chat.complete(
            model= model,
            messages = messages,
            temperature=0.6,
        )
        after_chat = time.time()
        print(f">>>> Response for {dataset_name} received in {after_chat - before_chat:.2f} seconds")
        print(chat_response.choices[0].message.content)
        first_annotation = chat_response.choices[0].message.content.split("\n")[0].split(". ")[1]
        second_annotation = chat_response.choices[0].message.content.split("\n")[1].split(". ")[1]
        print(f">>>> First annotation: {first_annotation}")
        print(f">>>> Second annotation: {second_annotation}")
        final_annotations[dataset_name] = [first_annotation, second_annotation]

        if args.visualize:
            concat_frames = torch.cat([first_frame, middle_frame, last_frame], dim=2)
            concat_frames = concat_frames.permute(1, 2, 0).numpy()
            concat_frames = cv2.cvtColor(concat_frames, cv2.COLOR_RGB2BGR)
            cv2.imshow("concat_frames", concat_frames)
            cv2.waitKey(0)
        
    # save the annotations to a file
    with open("automatic_annotations.txt", "w") as f:
        for dataset_name, annotations in final_annotations.items():
            f.write(f"{dataset_name}: {annotations}\n")
    print(">> Finished automatic annotation")
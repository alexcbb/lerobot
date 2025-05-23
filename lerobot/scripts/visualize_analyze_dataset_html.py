#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Visualize data of **all** frames of any episode of a dataset of type LeRobotDataset.

Note: The last frame of the episode doesnt always correspond to a final state.
That's because our datasets are composed of transition from state to state up to
the antepenultimate state associated to the ultimate action to arrive in the final state.
However, there might not be a transition from a final state to another state.

Note: This script aims to visualize the data used to train the neural networks.
~What you see is what you get~. When visualizing image modality, it is often expected to observe
lossly compression artifacts since these images have been decoded from compressed mp4 videos to
save disk space. The compression factor applied has been tuned to not affect success rate.

Example of usage:

- Visualize data stored on a local machine:
```bash
local$ python lerobot/scripts/visualize_dataset_html.py \
    --repo-id lerobot/pusht

local$ open http://localhost:9090
```

- Visualize data stored on a distant machine with a local viewer:
```bash
distant$ python lerobot/scripts/visualize_dataset_html.py \
    --repo-id lerobot/pusht

local$ ssh -L 9090:localhost:9090 distant  # create a ssh tunnel
local$ open http://localhost:9090
```

- Select episodes to visualize:
```bash
python lerobot/scripts/visualize_dataset_html.py \
    --repo-id lerobot/pusht \
    --episodes 7 3 5 1 4
```
"""

import argparse
import csv
import json
import logging
import re
import shutil
import tempfile
from io import StringIO
from pathlib import Path
import os 
import io
import time

import numpy as np
import pandas as pd
import requests
from flask import Flask, redirect, render_template, request, url_for, jsonify, send_file, make_response

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import IterableNamespace
from lerobot.common.utils.utils import init_logging

def filter_tasks(tasks_json):
    """Filter out tasks that are too short and contain weird names"""
    try:
        tasks = json.loads(tasks_json)
        valid_tasks = [task for task in tasks.values() 
                    if task and isinstance(task, str) and len(task.strip()) > 10 
                    and len(task.split("_")) < 2 and "test" not in task.lower()]
        return len(valid_tasks) > 0
    except Exception as e:
        print(f"Error while filtering tasks: {e}")
        return False

def filtering_metadata(
        df,
        num_episodes, 
        num_frames, 
        robot_set,
        filter_unlabeled_tasks,
        fps,
        max_tasks,
        start_date
    ):
    filtered_datasets = df[
        (df['total_episodes'] >= num_episodes) &
        (df['total_frames'] >= num_frames) & 
        (df['has_video'] == True) &
        (df['is_sim'] == False) &
        (df['robot_type'].isin(robot_set)) &
        (df['fps'].isin(fps)) &
        (df['total_tasks'] < max_tasks) &
        (df['creation_date'] >= start_date) &
        (~df['repo_id'].str.contains("test")) &
        (~df['repo_id'].str.contains("eval"))
    ]
    if filter_unlabeled_tasks:
        try:
            filtered_datasets['has_valid_tasks'] = filtered_datasets['tasks'].apply(filter_tasks)
            filtered_datasets = filtered_datasets[filtered_datasets['has_valid_tasks']]
        except Exception as e:
            print(f"Error while filtering tasks: {e}")
    if len(filtered_datasets) == 0:
        print("No dataset found with the specified filters")
        return 0, [], pd.DataFrame()
    return len(filtered_datasets), filtered_datasets["repo_id"].to_list(), filtered_datasets

def extract_data_from_current(current_dataset):
    min_eps = int(current_dataset['total_episodes'].min())
    min_frames = int(current_dataset['total_frames'].min())
    robot_types = list([str(el) for el in set(current_dataset['robot_type'].to_list())])
    robot_types.sort()
    fps_filter = list([int(el) for el in set(current_dataset['fps'].to_list())])
    task_count = int(current_dataset['total_tasks'].min())
    current_number_of_datasets = len(current_dataset)
    robot_fps = {}
    for robot in robot_types:
        robot_fps[str(robot)] = list([int(el) for el in set(current_dataset[current_dataset['robot_type'] == robot]['fps'].to_list())])
        robot_fps[robot].sort()
    robot_fps = json.dumps(robot_fps)
    return min_eps, min_frames, robot_types, fps_filter, task_count, current_number_of_datasets, robot_fps

def get_dataset_infos(current_dataset):
    datasets_info = []
    for _, row in current_dataset.iterrows():
        repo_id = row['repo_id']
        tasks = current_dataset[current_dataset['repo_id'] == repo_id]['tasks'].to_list()[0]
        tasks = json.loads(tasks)
        date_elems = current_dataset[current_dataset['repo_id'] == repo_id]["creation_date"].to_list()[0].split(" ")[0].split("-")
        creation_date = f"{date_elems[2]}/{date_elems[1]}/{date_elems[0]}"
        first_task = list(tasks.values())[0]
        datasets_info.append({
            'name': repo_id,
            'task_description': first_task if first_task else 'No task description available',
            'creation_date': creation_date
        })
    return datasets_info

def run_server(
    dataset: LeRobotDataset | IterableNamespace | None,
    episodes: list[int] | None,
    host: str,
    port: str,
    static_folder: Path,
    template_folder: Path,
):
    app = Flask(__name__, static_folder=static_folder.resolve(), template_folder=template_folder.resolve())
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0  # specifying not to cache

    csv_file = "./lerobot_datasets.csv"
    @app.route('/')
    def homepage():
        global full_dataset # CSV content
        global current_repo_id # Current repo_id being visualized for manual filtering
        global filtered_data # Filtered datasets after applying filters
        global current_dataset # Current datasets before filtering
        global map_repo_id_to_tasks
        full_dataset = pd.DataFrame()
        current_dataset = pd.DataFrame()
        filtered_data = pd.DataFrame()
        current_repo_id = None
        map_repo_id_to_tasks = {}
        csv_last_modified = os.path.getmtime(csv_file)
        last_modified = time.ctime(csv_last_modified)
        return render_template(
            'analyze_homepage.html',
            last_modified=last_modified
        )

    @app.route('/upload', methods=['POST'])
    def upload_file():
        global full_dataset
        global current_dataset
        global filtered_data

        if int(request.form['existing']) == 0:
            pasted_text = request.form.get('pasted_list', '').strip()
            if not pasted_text:
                print("No pasted dataset list provided.")
                return redirect(url_for('homepage'))
            import ast
            try:
                dataset_list = ast.literal_eval(pasted_text)
                if not isinstance(dataset_list, list):
                    raise ValueError("Pasted text is not a Python list.")
            except Exception as e:
                print(f"Could not parse pasted dataset list: {e}")
                return redirect(url_for('homepage'))

            # empty tasks column so that list_datasets.html can display them, we don't have access now
            keys = ['repo_id', 'tasks', 'creation_date', 'total_frames', 'total_episodes', 'fps', 'robot_type', 'has_video', 'is_sim', 'total_tasks']
            df = pd.DataFrame({'repo_id': dataset_list,
                            'tasks': [json.dumps({'0':'N/A'})]*len(dataset_list), 
                            'creation_date': ['N/A']*len(dataset_list), 
                            'total_frames': [0]*len(dataset_list),
                            'total_episodes': [0]*len(dataset_list),
                            'fps': [0]*len(dataset_list),
                            'robot_type': ['N/A']*len(dataset_list),
                            'has_video': [False]*len(dataset_list),
                            'is_sim': [False]*len(dataset_list),
                            'total_tasks': [0]*len(dataset_list)
                            })
            
            # to update for csv file
            file = csv_file
            if file:
                csv_df = pd.read_csv(file)
                # Check for tasks
                merged_df = df.merge(csv_df[keys], on='repo_id', how='left')
                for key in keys:
                    if key != 'repo_id':
                        df[key] = merged_df[key+'_y']
            full_dataset = df
            current_dataset = df
            filtered_data = df
            # go to filter page with the new list of datasets
            return redirect(url_for('list_datasets'))
        else:
            # to update for csv file
            file = csv_file
            if file == '' or not file.endswith('.csv') or not os.path.exists(file):
                print(f"File {file} does not exist")
                return redirect(url_for('homepage'))
            if file:
                df = pd.read_csv(file)
                # Remove nan robot type
                df = df.dropna(subset=['robot_type'])

                full_dataset = df
                current_dataset = df
                filtered_data = df

                dataset_infos = get_dataset_infos(current_dataset)
                min_eps, min_frames, robot_types, fps_filter, task_count, \
                    current_number_of_datasets, robot_fps = extract_data_from_current(current_dataset)
                return render_template('filter_dataset.html',
                                    min_frames=min_frames,
                                    min_eps=min_eps,
                                    robot_types=robot_types,
                                    fps_options=fps_filter,
                                    task_count=task_count,
                                    number_datasets=current_number_of_datasets,
                                    robot_fps_map=robot_fps,
                                    datasets=dataset_infos)

    @app.route('/submit', methods=['POST'])
    def submit_form():
        if int(request.form['finished']) == 0:
            selected_frames = int(request.form['frames'])
            selected_episodes = int(request.form['episodes'])
            selected_robot_type = request.form.getlist('robot_type')
            selected_fps = [int(float(el)) for el in request.form.getlist('fps')]
            selected_tasks = int(request.form['tasks'])
            start_date = request.form.get('startDate')

            global full_dataset
            global current_dataset
            try:
                total_datasets, repo_ids, filtered_datasets = filtering_metadata(
                    full_dataset,
                    selected_episodes,
                    selected_frames,
                    selected_robot_type,
                    False,
                    selected_fps,
                    selected_tasks,
                    start_date
                )
                if total_datasets == 0:
                    return jsonify({'datasets': [], 'totalDatasets': 0, 'error': 'No dataset found with the specified filters'})
                dataset_infos = get_dataset_infos(filtered_datasets)
                global filtered_data
                filtered_data = filtered_datasets
                current_dataset = filtered_datasets
                return jsonify({'datasets': dataset_infos, 'totalDatasets': total_datasets})
            except Exception as e:
                print(f"Error while filtering datasets: {e}")
                return jsonify({'datasets': [], 'totalDatasets': 0, 'error': str(e)})
        elif int(request.form['finished']) == 1:
            return redirect(url_for('list_datasets'))
        
    @app.route('/datasets')
    def list_datasets():
        global filtered_data
        if filtered_data.empty:
            return "No datasets available.", 404

        datasets_info = []
        for _, row in filtered_data.iterrows():
            repo_id = row['repo_id']
            tasks = filtered_data[filtered_data['repo_id'] == repo_id]['tasks'].to_list()[0]
            tasks = json.loads(tasks)
            first_task = list(tasks.values())[0]
            date_elems = current_dataset[current_dataset['repo_id'] == repo_id]["creation_date"].to_list()[0].split(" ")[0].split("-")
            creation_date = f"{date_elems[2]}/{date_elems[1]}/{date_elems[0]}"
            datasets_info.append({
                'name': repo_id,
                'task_description': first_task if first_task else 'No task description available',
                'creation_date': creation_date
            })
        return render_template('list_datasets.html', datasets=datasets_info)

    @app.route("/<string:dataset_namespace>/<string:dataset_name>/episode_<int:episode_id>")
    def show_episode(dataset_namespace, dataset_name, episode_id, dataset=dataset, episodes=episodes):
        repo_id = f"{dataset_namespace}/{dataset_name}"
        global current_repo_id
        current_repo_id = repo_id
        global filtered_data
        visualized_dataset = filtered_data[filtered_data['repo_id'] == repo_id]
        try:
            dataset = get_dataset_info(repo_id)
        except FileNotFoundError:
            return (
                "Make sure to convert your LeRobotDataset to v2 & above. See how to convert your dataset at https://github.com/huggingface/lerobot/pull/461",
                400,
            )

        episode_data_csv_str, columns, ignored_columns = get_episode_data(dataset, episode_id)
        dataset_info = {
            "repo_id": f"{dataset_namespace}/{dataset_name}",
            "num_samples":  visualized_dataset['total_frames'].to_list()[0],
            "num_episodes": visualized_dataset['total_episodes'].to_list()[0],
            "fps": visualized_dataset['fps'].to_list()[0],
        }
        video_keys = [key for key, ft in dataset.features.items() if ft["dtype"] == "video"]
        videos_info = [
            {
                "url": f"https://huggingface.co/datasets/{repo_id}/resolve/main/"
                + dataset.video_path.format(
                    episode_chunk=int(episode_id) // dataset.chunks_size,
                    video_key=video_key,
                    episode_index=episode_id,
                ),
                "filename": video_key,
            }
            for video_key in video_keys
        ]
        tasks = json.loads(visualized_dataset['tasks'].tolist()[0])
        tasks = list(tasks.values())[0]
        videos_info[0]["language_instruction"] = [tasks]

        if episodes is None:
            episodes = list(
                range(dataset.total_episodes)
            )
        global current_dataset
        remaining_data = len(filtered_data)
        return render_template(
            "manual_filter_dataset.html",
            episode_id=episode_id,
            episodes=episodes,
            dataset_info=dataset_info,
            videos_info=videos_info,
            episode_data_csv_str=episode_data_csv_str,
            columns=columns,
            ignored_columns=ignored_columns,
            number_datasets=len(current_dataset),
            number_remain_datasets=remaining_data,
        )

    @app.route("/change_desc", methods=['POST'])
    def change_desc():
        global current_repo_id
        global filtered_data
        global current_dataset
        global map_repo_id_to_tasks
        repo_id = current_repo_id
        validate = request.form.get('btnValidate')
        tasks = request.form.get('task_description')
        map_repo_id_to_tasks[repo_id] = tasks
        tasks = '{"0": "' + tasks + '" }'
        filtered_data.loc[filtered_data['repo_id'] == repo_id, 'tasks'] = tasks
        current_dataset.loc[current_dataset['repo_id'] == repo_id, 'tasks'] = tasks
        if validate:
            filtered_data = filtered_data[filtered_data['repo_id'] != repo_id]
            if len(filtered_data) > 0:
                next_dataset = filtered_data.iloc[0]
                dataset_namespace = next_dataset['repo_id'].split("/")[0]
                dataset_name = next_dataset['repo_id'].split("/")[1]
                return redirect(
                    url_for(
                        "show_episode",
                        dataset_namespace=dataset_namespace,
                        dataset_name=dataset_name,
                        episode_id=0,
                    )
                )
            else:
                all_repo_ids = current_dataset['repo_id'].to_list()
                return render_template(
                    'final_filtering.html', 
                    number_datasets=len(current_dataset),
                    repo_ids=all_repo_ids,
                    tasks_mapping=map_repo_id_to_tasks)
        return redirect(url_for('list_datasets'))

    @app.route("/filter", methods=['POST'])
    def filter_dataset():
        global current_repo_id
        repo_id = current_repo_id
        validate = request.form.get('btnValidate')
        remove = request.form.get('btnRemove')
        if remove:
            global current_dataset
            current_dataset = current_dataset[current_dataset['repo_id'] != repo_id]

        global filtered_data
        filtered_data = filtered_data[filtered_data['repo_id'] != repo_id]
        if len(filtered_data) > 0:
            next_dataset = filtered_data.iloc[0]
            dataset_namespace = next_dataset['repo_id'].split("/")[0]
            dataset_name = next_dataset['repo_id'].split("/")[1]

            return redirect(
                url_for(
                    "show_episode",
                    dataset_namespace=dataset_namespace,
                    dataset_name=dataset_name,
                    episode_id=0,
                )
            )
        else:
            all_repo_ids = current_dataset['repo_id'].to_list()
            return render_template(
                'final_filtering.html', 
                number_datasets=len(current_dataset),
                repo_ids=all_repo_ids,
                tasks_mapping=map_repo_id_to_tasks)
        
    @app.route('/final_filtering')
    def final_filtering():
        global current_dataset
        all_repo_ids = current_dataset['repo_id'].to_list()
        global map_repo_id_to_tasks
        return render_template(
            'final_filtering.html', 
            number_datasets=len(current_dataset),
            repo_ids=all_repo_ids,
            tasks_mapping=map_repo_id_to_tasks)
    
    @app.route('/download_csv')
    def download_csv():
        global current_dataset
        output = io.StringIO()
        current_dataset.to_csv(output)
        response = make_response(output.getvalue())
        response.headers['Content-Disposition'] = 'attachment; filename=current_dataset.csv'
        response.headers['Content-Type'] = 'text/csv'
        return response

    @app.route('/download_json', methods=['POST'])
    def download_json():
        map_to_id = request.form.get('map_to_id')
        if map_to_id:
            # Convert the string back to a dictionary
            map_to_id_dict = json.loads(map_to_id)

            # Create a JSON file in memory
            json_file = io.BytesIO()
            json_file.write(json.dumps(map_to_id_dict, indent=4).encode('utf-8'))
            json_file.seek(0)

            # Send the file as a downloadable attachment
            return send_file(json_file, mimetype='application/json', as_attachment=True, download_name='map_to_id.json')
        return jsonify({"error": "No data provided"}), 400
        
    app.run(host=host, port=port, debug=True)

def get_episode_data(dataset: LeRobotDataset | IterableNamespace, episode_index):
    """Get a csv str containing timeseries data of an episode (e.g. state and action).
    This file will be loaded by Dygraph javascript to plot data in real time."""
    columns = []

    selected_columns = [col for col, ft in dataset.features.items() if ft["dtype"] in ["float32", "int32"]]
    selected_columns.remove("timestamp")

    ignored_columns = []
    for column_name in selected_columns:
        shape = dataset.features[column_name]["shape"]
        shape_dim = len(shape)
        if shape_dim > 1:
            selected_columns.remove(column_name)
            ignored_columns.append(column_name)

    # init header of csv with state and action names
    header = ["timestamp"]

    for column_name in selected_columns:
        dim_state = (
            dataset.meta.shapes[column_name][0]
            if isinstance(dataset, LeRobotDataset)
            else dataset.features[column_name].shape[0]
        )

        if "names" in dataset.features[column_name] and dataset.features[column_name]["names"]:
            column_names = dataset.features[column_name]["names"]
            while not isinstance(column_names, list):
                column_names = list(column_names.values())[0]
        else:
            column_names = [f"{column_name}_{i}" for i in range(dim_state)]
        columns.append({"key": column_name, "value": column_names})

        header += column_names

    selected_columns.insert(0, "timestamp")

    repo_id = dataset.repo_id

    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/" + dataset.data_path.format(
        episode_chunk=int(episode_index) // dataset.chunks_size, episode_index=episode_index
    )
    df = pd.read_parquet(url)
    # only keep columns that are in df
    for col in selected_columns:
        if col not in df.columns:
            selected_columns.remove(col)
            ignored_columns.append(col)
    data = df[selected_columns]  # Select specific columns

    rows = np.hstack(
        (
            np.expand_dims(data["timestamp"], axis=1),
            *[np.vstack(data[col]) for col in selected_columns[1:]],
        )
    ).tolist()

    # Convert data to CSV string
    csv_buffer = StringIO()
    csv_writer = csv.writer(csv_buffer)
    # Write header
    csv_writer.writerow(header)
    # Write data rows
    csv_writer.writerows(rows)
    csv_string = csv_buffer.getvalue()

    return csv_string, columns, ignored_columns

def get_dataset_info(repo_id: str) -> IterableNamespace:
    response = requests.get(
        f"https://huggingface.co/datasets/{repo_id}/resolve/main/meta/info.json", timeout=5
    )
    response.raise_for_status()  # Raises an HTTPError for bad responses
    dataset_info = response.json()
    dataset_info["repo_id"] = repo_id
    return IterableNamespace(dataset_info)


def visualize_dataset_html(
    dataset: LeRobotDataset | None,
    episodes: list[int] | None = None,
    output_dir: Path | None = None,
    serve: bool = True,
    host: str = "127.0.0.1",
    port: int = 9090,
    force_override: bool = False,
) -> Path | None:
    init_logging()

    template_dir = Path(__file__).resolve().parent.parent / "templates"

    if output_dir is None:
        # Create a temporary directory that will be automatically cleaned up
        output_dir = tempfile.mkdtemp(prefix="lerobot_visualize_dataset_")

    output_dir = Path(output_dir)
    if output_dir.exists():
        if force_override:
            shutil.rmtree(output_dir)
        else:
            logging.info(f"Output directory already exists. Loading from it: '{output_dir}'")

    output_dir.mkdir(parents=True, exist_ok=True)

    static_dir = output_dir / "static"
    static_dir.mkdir(parents=True, exist_ok=True)

    if dataset is None:
        if serve:
            run_server(
                dataset=None,
                episodes=None,
                host=host,
                port=port,
                static_folder=static_dir,
                template_folder=template_dir,
            )
    else:
        # Create a simlink from the dataset video folder containing mp4 files to the output directory
        # so that the http server can get access to the mp4 files.
        if isinstance(dataset, LeRobotDataset):
            ln_videos_dir = static_dir / "videos"
            if not ln_videos_dir.exists():
                ln_videos_dir.symlink_to((dataset.root / "videos").resolve())

        if serve:
            run_server(dataset, episodes, host, port, static_dir, template_dir)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Name of hugging face repositery containing a LeRobotDataset dataset (e.g. `lerobot/pusht` for https://huggingface.co/datasets/lerobot/pusht).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory for a dataset stored locally (e.g. `--root data`). By default, the dataset will be loaded from hugging face cache folder, or downloaded from the hub if available.",
    )
    parser.add_argument(
        "--load-from-hf-hub",
        type=int,
        default=0,
        help="Load videos and parquet files from HF Hub rather than local system.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="*",
        default=None,
        help="Episode indices to visualize (e.g. `0 1 5 6` to load episodes of index 0, 1, 5 and 6). By default loads all episodes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory path to write html files and kickoff a web server. By default write them to 'outputs/visualize_dataset/REPO_ID'.",
    )
    parser.add_argument(
        "--serve",
        type=int,
        default=1,
        help="Launch web server.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Web host used by the http server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9090,
        help="Web port used by the http server.",
    )
    parser.add_argument(
        "--force-override",
        type=int,
        default=0,
        help="Delete the output directory if it exists already.",
    )

    args = parser.parse_args()
    kwargs = vars(args)
    repo_id = kwargs.pop("repo_id")
    load_from_hf_hub = kwargs.pop("load_from_hf_hub")
    root = kwargs.pop("root")

    dataset = None
    if repo_id:
        dataset = LeRobotDataset(repo_id, root=root) if not load_from_hf_hub else get_dataset_info(repo_id)

    visualize_dataset_html(dataset, **vars(args))


if __name__ == "__main__":
    main()

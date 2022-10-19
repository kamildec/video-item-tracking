#!/usr/bin/env python

import argparse
import io
import json
import logging
import os
import time
import zipfile
from enum import Enum
from typing import Any

import requests

PROD_API_URL = "https://demeter-cvat.prod.nomagic.io/api/v1"
STAGING_API_URL = "https://demeter-cvat.staging.nomagic.io/api/v1"


class COCOAnnotations:
    """Manager to create a joint annotation json file for all videos per dataset split (train/test). Annotation file
    consists of 4 sections: categories, videos, images, annotations.
    """

    def __init__(self) -> None:
        self._categories_list: list[dict[str, Any]] = []
        self._videos: list[dict[str, Any]] = []
        self._images: list[dict[str, Any]] = []
        self._annotations: list[dict[str, Any]] = []

        self._categories_names: set[str] = set()
        self._video_id_to_name: dict[int, str] = {}
        self._video_frame_to_image_id: dict[tuple[int, int], int] = {}  # Mapping (video_id, frame_id) -> image_id.

        self._next_category_id = 1
        self._next_image_id = 1
        self._next_annotation_id = 1

    def add_category(self, name: str) -> None:
        """Add category, if category with the same name does not exist yet.

        Args:
            name (str): Name of the category.
        """

        if name not in self._categories_names:
            self._categories_names.add(name)
            self._categories_list.append({"id": self._next_category_id, "name": name})
            self._next_category_id += 1

    def add_video(self, id: int, filename: str) -> None:
        """Add video entry.

        Args:
            id (int): Video ID (same as the ID of the corresponding cvat task).
            filename (str): Name of the video file or directory with images.
        """

        self._videos.append({"id": id, "file_name": filename})
        self._video_id_to_name[id] = filename

    def add_image(self, filename: str, frame_id: int, video_id: int, height: int, width: int) -> None:
        """Add image entry. Keeps track of the previous and next frame of this video.

        Args:
            filename (str): Image filename.
            frame_id (int): Frame number in the video.
            video_id (int): ID of the video frame belongs to.
            height (int): Frame height.
            width (int): Frame width.
        """

        prev_image_id = -1
        if frame_id > 1:
            self._images[-1]["next_image_id"] = self._next_image_id
            prev_image_id = self._images[-1]["id"]

        entry = {
            "file_name": os.path.join(self._video_id_to_name[video_id], "images", filename),
            "id": self._next_image_id,
            "frame_id": frame_id,
            "prev_images_id": prev_image_id,
            "next_image_id": -1,
            "video_id": video_id,
            "height": height,
            "width": width,
        }

        self._images.append(entry)
        self._video_frame_to_image_id[video_id, frame_id] = self._next_image_id
        self._next_image_id += 1

    def add_annotation(self, video_id: int, cvat_annotation: dict[str, Any], use_box_area: bool = True) -> None:
        """Add annotation entry. Annotation contains both segmentation mask polygon points and bbox. By default, saves
        bbox area under 'area' parameter and segmentation area under 'area_segmentation'.

        Args:
            video_id (int): ID of the video.
            cvat_annotation (dict[str, Any]): Annotation entry returned from CVAT COCO file.
            use_box_area (bool, optional): If True saves bbox area under 'area' otherwise saves segmentation area as
                'area'. Defaults to False.
        """

        entry = {
            "id": self._next_annotation_id,
            "category_id": cvat_annotation["category_id"],
            "image_id": self._video_frame_to_image_id[video_id, cvat_annotation["image_id"]],
            "track_id": int(cvat_annotation["attributes"]["id"] + 1),
            "bbox": cvat_annotation["bbox"],
            "conf": 1.0,
            "iscrowd": 0,
            "attributes": cvat_annotation["attributes"],
            "area": cvat_annotation["bbox"][2] * cvat_annotation["bbox"][3]
            if use_box_area
            else cvat_annotation["area"],
            "area_segmentation": cvat_annotation["area"],
        }

        self._annotations.append(entry)
        self._next_annotation_id += 1

    def as_dict(self) -> dict[str, Any]:
        """Export data to dictionary format."""

        return {
            "categories": self._categories_list,
            "videos": self._videos,
            "images": self._images,
            "annotations": self._annotations,
        }


class DatasetSplit(Enum):
    TRAIN = "train"
    TEST = "test"


class CVATDownloader:
    def __init__(self, save_path: str, username: str, password: str, cvat_api_url: str) -> None:
        """Utility class to download tasks from CVAT.

        Args:
            save_path (str): Path to save dataset.
            username (str): Username to log in to CVAT.
            password (str): Password to log in to CVAT.
            cvat_api_url (str): URL of the CVAT API.
        """

        self.api_url = cvat_api_url
        self.session = requests.Session()
        self.session.auth = (username, password)
        self._save_path = save_path
        self._tasks_url = self._get_request_url("tasks")
        self._annotations: dict[DatasetSplit, COCOAnnotations] = {
            DatasetSplit.TRAIN: COCOAnnotations(),
            DatasetSplit.TEST: COCOAnnotations(),
        }

    def _get_request_url(self, request: str) -> str:
        if request.startswith("/"):
            return os.path.join(self.api_url, request[1:])
        else:
            return os.path.join(self.api_url, request)

    def _get_tasks(self, task_prefix: str) -> list[dict[str, Any]]:
        """Returns json descriptions of all tasks with specified prefix."""

        tasks_response = self.session.get(self._tasks_url)
        tasks_response.raise_for_status()
        tasks_response_json = tasks_response.json()
        tasks_jsons = tasks_response_json["results"]

        next_task = tasks_response_json["next"]
        while next_task is not None:
            tasks_response = self.session.get(next_task)
            tasks_response.raise_for_status()
            tasks_response_json = tasks_response.json()
            tasks_jsons.extend(tasks_response_json["results"])
            next_task = tasks_response_json["next"]

        tasks: list[dict[str, Any]] = []
        for task_json in tasks_jsons:
            if task_json["name"].startswith(task_prefix):
                tasks.append(task_json)

        return tasks

    def _get_cvat_coco_dataset_per_task(
        self,
        task: dict[str, Any],
        split: DatasetSplit,
        annotations_only: bool = False,
    ) -> None:
        """Processes single CVAT task. Downloads annotations in COCO format (and optionally dataset) from CVAT and
        updates dataset annotation file.

        Args:
            task (dict[str, Any]): dictionary description of the CVAT task.
            split (DatasetSplit): Which annotation split to use.
            annotations_only (bool, optional): If True downloads only annotation, otherwise downloads images dataset
                as well. Defaults to False.
        """

        dump_cvat_url = self._get_request_url(
            f"/tasks/{task['id']}/{'annotations' if annotations_only else 'dataset'}?format=COCO%201.0&action=download"
        )
        logging.info(f"Downloading coco dataset from: {dump_cvat_url}")

        dump_cvat_response = self.session.get(dump_cvat_url)

        # When CVAT accepts the request with 202 status code, it does not send the dumped annotations because it is
        # still preparing them, we must wait and renew the request until the annotations are ready.
        while dump_cvat_response.status_code == 202:
            logging.info(f"Waiting for task '{task['name']}'.")
            time.sleep(0.05)
            dump_cvat_response = self.session.get(dump_cvat_url)

        dump_cvat_response.raise_for_status()
        dump_cvat_zipfile = zipfile.ZipFile(io.BytesIO(dump_cvat_response.content))
        dump_cvat_annotations: dict[str, Any] = json.loads(dump_cvat_zipfile.read("annotations/instances_default.json"))

        if not annotations_only:
            # Extract image files.
            images_path = os.path.join(self._save_path, split.value, task["name"])
            os.makedirs(images_path, exist_ok=True)

            for member_name in dump_cvat_zipfile.namelist():
                if member_name.startswith("images/"):
                    dump_cvat_zipfile.extract(member_name, images_path)

            logging.info(f"Extracted images to '{images_path}'.")

        # Extract annotations.
        for category in dump_cvat_annotations["categories"]:
            self._annotations[split].add_category(category["name"])

        self._annotations[split].add_video(task["id"], task["name"])

        for image in dump_cvat_annotations["images"]:
            self._annotations[split].add_image(
                filename=image["file_name"],
                frame_id=image["id"],
                video_id=task["id"],
                height=image["height"],
                width=image["width"],
            )

        for annotation in dump_cvat_annotations["annotations"]:
            self._annotations[split].add_annotation(task["id"], annotation)

        logging.info(f"Added annotations for task '{task['name']}' to {split.value} annotations.")

    def download_dataset(self, task_prefix: str, annotations_only: bool = False) -> None:
        """Processes all CVAT tasks with given prefix."""

        for task in self._get_tasks(task_prefix):
            split = DatasetSplit.TRAIN if task["id"] % 5 != 0 else DatasetSplit.TEST
            self._get_cvat_coco_dataset_per_task(task, split, annotations_only)

        train_annotations_path = os.path.join(self._save_path, "annotations", DatasetSplit.TRAIN.value + ".json")
        test_annotations_path = os.path.join(self._save_path, "annotations", DatasetSplit.TEST.value + ".json")

        os.makedirs(os.path.join(self._save_path, "annotations"), exist_ok=True)

        with open(train_annotations_path, "w") as f:
            json.dump(self._annotations[DatasetSplit.TRAIN].as_dict(), f, indent=4)

        logging.info(f"Saved train annotations to '{train_annotations_path}'.")

        with open(test_annotations_path, "w") as f:
            json.dump(self._annotations[DatasetSplit.TEST].as_dict(), f, indent=4)

        logging.info(f"Saved test annotations to '{test_annotations_path}'.")


def main(save_path: str, task_prefix: str, username: str, password: str, annotations_only: bool, prod: bool) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s.%(msecs)03d][%(levelname)s] %(message)s")

    cvat_api_url = PROD_API_URL if prod else STAGING_API_URL
    cvat_client = CVATDownloader(save_path=save_path, username=username, password=password, cvat_api_url=cvat_api_url)

    cvat_client.download_dataset(task_prefix, annotations_only=annotations_only)


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-prefix",
        type=str,
        required=True,
        help="Task's name prefix.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        required=True,
        help="Path to directory where to save dataset.",
    )
    parser.add_argument(
        "--username",
        type=str,
        required=True,
        help="Username to log in to CVAT.",
    )
    parser.add_argument(
        "--password",
        type=str,
        required=True,
        help="Password to log in to CVAT.",
    )
    parser.add_argument(
        "--annotations-only",
        default=False,
        action="store_true",
        help="Download only annotations (no images).",
    )
    parser.add_argument(
        "--prod",
        default=False,
        action="store_true",
        help=f"If passed, downloads from {PROD_API_URL}. Otherwise downloads from {STAGING_API_URL}.",
    )
    parser.set_defaults(annotations_only=False)

    return parser


if __name__ == "__main__":
    args = arg_parser().parse_args()
    main(**vars(args))

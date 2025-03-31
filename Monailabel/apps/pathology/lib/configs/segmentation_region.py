# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import multiprocessing
import os
from typing import Any, Dict, Optional, Union

import lib.infers
import lib.trainers
from monai.networks.nets import BasicUNet

from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer_v2 import InferTask
from monailabel.interfaces.tasks.train import TrainTask
# from monailabel.utils.others.generic import download_file, strtobool

logger = logging.getLogger(__name__)


class SegmentationRegion(TaskConfig):
    def init(self, name: str, model_dir: str, conf: Dict[str, str], planner: Any, **kwargs):
        super().init(name, model_dir, conf, planner, **kwargs)

        # Labels
        self.labels = {
            "Tumor": 1,
            "Stroma": 2,
            "Immune Cells": 3,
            "Necrosis": 4,
            "Other": 5,
        }
        self.label_colors = {
            "Tumor": (255, 0, 0),
            "Stroma": (255, 255, 0),
            "Immune Cells": (0, 255, 0),
            "Necrosis": (0, 0, 0),
            "Other": (0, 0, 255),
        }

        # Model Files disabled
        self.path = [
            os.path.join(self.model_dir, f"pretrained_{name}.pt"),  # pretrained
            os.path.join(self.model_dir, f"{name}.pt"),  # published
        ]


        # Network
        self.network = BasicUNet(
            spatial_dims=2,
            in_channels=3,
            out_channels=len(self.labels) + 1,
            features=(32, 64, 128, 256, 512, 32),
        )

    def infer(self) -> Union[InferTask, Dict[str, InferTask]]:
        preload = False
        roi_size = json.loads(self.conf.get("roi_size", "[1024, 1024]"))
        logger.info(f"Using Preload: {preload}; ROI Size: {roi_size}")

        task: InferTask = lib.infers.SegmentationRegion(
            path=self.path,
            network=self.network,
            labels=self.labels,
            preload=preload,
            roi_size=roi_size,
            config={
                "label_colors": self.label_colors,
                "max_workers": max(1, multiprocessing.cpu_count() // 2),
            },
        )
        return task

    def trainer(self) -> Optional[TrainTask]:
        output_dir = os.path.join(self.model_dir, self.name)
        load_path = self.path[0] if os.path.exists(self.path[0]) else self.path[1]

        task: TrainTask = lib.trainers.SegmentationRegion(
            model_dir=output_dir,
            network=self.network,
            load_path=load_path,
            publish_path=self.path[1],
            labels=self.labels,
            description="Train Region Segmentation Model",
            config={
                "max_epochs": 10,
                "train_batch_size": 16,
                "val_batch_size": 16,
            },
        )
        return task

import json
import logging
import multiprocessing
import os
from typing import Any, Dict, Optional, Union

import timm
import torch.nn as nn
import lib.infers
import lib.trainers
from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer_v2 import InferTask
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.utils.others.generic import download_file, strtobool

logger = logging.getLogger(__name__)


class SegmentationNuclei(TaskConfig):
    def init(self, name: str, model_dir: str, conf: Dict[str, str], planner: Any, **kwargs):
        super().init(name, model_dir, conf, planner, **kwargs)

        # Labels
        self.labels = {"Neoplastic cells": 1, "Inflammatory": 2, "Epithelial": 3, "Spindle-Shaped": 4}
        self.label_colors = {
            "Neoplastic cells": (255, 0, 0),
            "Inflammatory": (255, 255, 0),
            "Epithelial": (0, 0, 255),
            "Spindle-Shaped": (0, 255, 0),
        }

        # Model Paths
        self.path = [os.path.join(self.model_dir, f"pretrained_{name}.pth"), os.path.join(self.model_dir, f"{name}.pth")]

        # Download Pretrained Model (if needed)
        # if strtobool(self.conf.get("use_pretrained_model", "true")):
        #     url = f"{self.conf.get('pretrained_path', self.PRE_TRAINED_PATH)}/pathology_vit_segmentation.pth"
        #     download_file(url, self.path[0])

        # Load Pretrained ViT Model from timm
        self.network = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=len(self.labels) + 1)

        # Modify the output layer for segmentation
        self.network.head = nn.Linear(in_features=768, out_features=len(self.labels) + 1)

    def infer(self) -> Union[InferTask, Dict[str, InferTask]]:
        preload = strtobool(self.conf.get("preload", "false"))
        roi_size = json.loads(self.conf.get("roi_size", "[1024, 1024]"))
        logger.info(f"Using Preload: {preload}; ROI Size: {roi_size}")

        task: InferTask = lib.infers.SegmentationNuclei(
            path=self.path,
            network=self.network,
            labels=self.labels,
            preload=preload,
            roi_size=roi_size,
            config={"label_colors": self.label_colors, "max_workers": max(1, multiprocessing.cpu_count() // 2)},
        )
        return task

    def trainer(self) -> Optional[TrainTask]:
        output_dir = os.path.join(self.model_dir, self.name)
        load_path = self.path[0] if os.path.exists(self.path[0]) else self.path[1]

        task: TrainTask = lib.trainers.SegmentationNuclei(
            model_dir=output_dir,
            network=self.network,
            load_path=load_path,
            publish_path=self.path[1],
            labels=self.labels,
            description="Train Nuclei Segmentation Model using ViT",
            config={"max_epochs": 20, "train_batch_size": 8, "val_batch_size": 8},
        )
        return task

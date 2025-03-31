import json
import logging
import multiprocessing
import os
from typing import Any, Dict, Optional, Union

import lib.infers
import lib.trainers
import torch.nn as nn
import torch

from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer_v2 import InferTask
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.utils.others.generic import strtobool

logger = logging.getLogger(__name__)


# Define a simple segmentation network with random weights
class RandomSegNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=6):
        """
        A simple 2-layer convolutional network.
        out_channels should be number of labels + 1 (for background).
        """
        super(RandomSegNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        return x


class RandomSegmentation(TaskConfig):
    def init(self, name: str, model_dir: str, conf: Dict[str, str], planner: Any, **kwargs):
        super().init(name, model_dir, conf, planner, **kwargs)

        # Define labels and their colors
        self.labels = {
            "Neoplastic cells": 1,
            "Inflammatory": 2,
            "Connective/Soft tissue cells": 3,
            "Dead Cells": 4,
            "Epithelial": 5,
        }
        self.label_colors = {
            "Neoplastic cells": (255, 0, 0),
            "Inflammatory": (255, 255, 0),
            "Connective/Soft tissue cells": (0, 255, 0),
            "Dead Cells": (0, 0, 0),
            "Epithelial": (0, 0, 255),
        }

        # Optional flag for alternative labels (if needed)
        consep = strtobool(self.conf.get("consep", "false"))
        if consep:
            self.labels = {
                "Other": 1,
                "Inflammatory": 2,
                "Epithelial": 3,
                "Spindle-Shaped": 4,
            }
            self.label_colors = {
                "Other": (255, 0, 0),
                "Inflammatory": (255, 255, 0),
                "Epithelial": (0, 0, 255),
                "Spindle-Shaped": (0, 255, 0),
            }
        # Model Files are defined but we will use random weights so no download occurs.
        # self.path = [
        #     os.path.join(self.model_dir, f"pretrained_{name}{'_consep' if consep else ''}.pt"),
        #     os.path.join(self.model_dir, f"{name}{'_consep' if consep else ''}.pt"),
        # ]
        self.path = [
            os.path.join(self.model_dir, f"pretrained_{name}{''}.pt"),
            os.path.join(self.model_dir, f"{name}{''}.pt"),
        ]

        # Skip downloading pretrained weights for a randomly initialized model

        # Instantiate our new network with random weights.
        self.network = RandomSegNet(
            in_channels=3,
            out_channels=len(self.labels) + 1  # plus background channel
        )

    def infer(self) -> Union[InferTask, Dict[str, InferTask]]:
        # For random weights, no need to preload
        preload = False
        roi_size = json.loads(self.conf.get("roi_size", "[1024, 1024]"))
        logger.info(f"Using Preload: {preload}; ROI Size: {roi_size}")

        task: InferTask = lib.infers.RandomSegmentation(
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
        # Do not load any existing weights for random initialization.
        load_path = None

        task: TrainTask = lib.trainers.RandomSegmentation(
            model_dir=output_dir,
            network=self.network,
            load_path=load_path,
            publish_path=self.path[1],
            labels=self.labels,
            description="Train Nuclei Segmentation Model with Random Weights",
            config={
                "max_epochs": 10,
                "train_batch_size": 16,
                "val_batch_size": 16,
            },
        )
        return task

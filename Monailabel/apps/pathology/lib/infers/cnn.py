import logging
from typing import Any, Callable, Dict, Sequence

import numpy as np
from lib.transforms import LoadImagePatchd, PostFilterLabeld
from monai.transforms import Activationsd, AsDiscreted, ScaleIntensityRangeD, SqueezeDimd, Transposed

from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.tasks.infer.basic_infer import BasicInferTask
from monailabel.transform.post import FindContoursd
from monailabel.transform.writer import PolygonWriter

logger = logging.getLogger(__name__)


class RandomSegmentation(BasicInferTask):
    """
    Inference engine for nuclei segmentation using RandomSegNet with random weights.
    """

    def __init__(
        self,
        path,
        network=None,
        roi_size=(256, 256),
        type=InferType.SEGMENTATION,
        labels=None,
        dimension=2,
        description="A segmentation model for Nuclei with random weights",
        **kwargs,
    ):
        super().__init__(
            path=path,
            network=network,
            roi_size=roi_size,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
            load_strict=False,
            **kwargs,
        )

    def info(self) -> Dict[str, Any]:
        d = super().info()
        d["pathology"] = True
        return d

    def pre_transforms(self, data=None) -> Sequence[Callable]:
        return [
            LoadImagePatchd(keys="image", mode="RGB", dtype=np.uint8, padding=False),
            Transposed(keys="image", indices=[2, 0, 1]),
            ScaleIntensityRangeD(keys="image", a_min=0.0, a_max=255.0, b_min=-1.0, b_max=1.0),
        ]

    def post_transforms(self, data=None) -> Sequence[Callable]:
        return [
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys="pred", argmax=True),
            SqueezeDimd(keys="pred", dim=0),
            PostFilterLabeld(keys="pred"),
            FindContoursd(keys="pred", labels=self.labels, max_poly_area = 128 * 128),
        ]

    def writer(self, data, extension=None, dtype=None):
        writer = PolygonWriter(label=self.output_label_key, json=self.output_json_key)
        return writer(data)

import os
import sys
import numpy as np
import saverloader
import torch
from dotenv import load_dotenv
from pathlib import Path
from typing import Any, Dict, List, Literal
from typing_extensions import Literal
from nets.pips import Pips

import sly_functions
import supervisely as sly
from supervisely.nn.inference import PointTracking
from supervisely.nn.prediction_dto import PredictionPoint


root = (Path(__file__).parent / ".." / ".." / "..").resolve().absolute()
settings = root / "supervisely" / "serve" / "model_settings.yaml"


load_dotenv(root / "local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))


class PipsTracker(PointTracking):
    def load_on_device(
        self,
        model_dir: str,
        device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"] = "cpu",
    ):
        frames_per_iter = self.custom_inference_settings_dict.get("frames_per_iter", 8)
        stride = self.custom_inference_settings_dict.get("stride", 4)

        self.model = Pips(S=frames_per_iter, stride=stride).to(torch.device(device))
        if model_dir:
            _ = saverloader.load(str(model_dir), self.model, device=device)
        self.model.eval()
        self.device = device

    def predict(
        self,
        rgb_images: List[np.ndarray],
        settings: Dict[str, Any],
        start_object: PredictionPoint,
    ) -> List[PredictionPoint]:
        class_name = start_object.class_name
        h_resized = settings.get("h_resized", 360)
        w_resized = settings.get("w_resized", 640)
        frames_per_iter = settings.get("frames_per_iter", 8)

        rgbs = [torch.from_numpy(rgb_img).permute(2, 0, 1) for rgb_img in rgb_images]
        rgbs = torch.stack(rgbs, dim=0).unsqueeze(0)
        point = torch.tensor([[start_object.col, start_object.row]], dtype=float)

        with torch.no_grad():
            traj = sly_functions.run_model(
                self.model,
                rgbs,
                point,
                (h_resized, w_resized),
                frames_per_iter,
                device=self.device,
            )

        pred_points = [PredictionPoint(class_name, col=p[0], row=p[1]) for p in traj[1:]]
        return pred_points


if sly.is_debug_with_sly_net():
    model_dir = root / "reference_model"
else:
    model_dir = Path("/weights")  # path in Docker

pips = PipsTracker(model_dir=str(model_dir), custom_inference_settings=str(settings))

if sly.is_production():
    pips.serve()

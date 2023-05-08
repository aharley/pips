import functools

import numpy as np
import saverloader
from pathlib import Path
from typing import Any, Dict, List, Literal
from typing_extensions import Literal
from nets.pips import Pips

import sly_globals as g
import supervisely as sly
from supervisely.nn.inference import Tracking
from supervisely.nn.prediction_dto import PredictionPoint


root = (Path(__file__).parent / ".." / "..").resolve().absolute()

class PipsTracker(Tracking):
    def load_on_device(self, model_dir: str, device: Literal['cpu', 'cuda', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'] = "cpu"):
        frames_per_iter = self.custom_inference_settings_dict.get("frames_per_iter", 8)
        stride = self.custom_inference_settings_dict.get("stride", 4)
        
        self.model = Pips(S = frames_per_iter, stride = stride).to(torch.device(device))
        if model_dir:
            _ = saverloader.load(str(model_dir), self.model)
        self.model.eval()
    
    def predict(self, rgb_images: List[np.ndarray], settings: Dict[str, Any], start_object: PredictionPoint) -> List[PredictionPoint]:
        return super().predict(rgb_images, settings, start_object)


if sly.is_debug_with_sly_net():
    model_dir = root / "reference_model"
else:
    model_dir = Path("/weights")  # path in Docker

if sly.is_production():
    # pips.serve()
    pass
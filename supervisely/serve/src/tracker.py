import torch
from typing import Optional, List
import sly_functions

import supervisely as sly
from supervisely.geometry.geometry import Geometry


class TrackerContainer:
    def __init__(self, context, api, logger):
        self.api = api
        self.logger = logger
        self.frame_index = context["frameIndex"]
        self.frames_count = context["frames"]

        self.track_id = context["trackId"]
        self.video_id = context["videoId"]
        self.object_ids = list(context["objectIds"])
        self.figure_ids = list(context["figureIds"])
        self.direction = context["direction"]
        self.stop = len(self.object_ids) * self.frames_count

        self.geometries: List[Geometry] = []
        self.frames_indexes: List[int] = []
        self.frames: Optional[torch.Tensor] = None

        self.add_geometries()
        self.add_frames_indexes()
        self.load_frames()
        self.model = sly_functions.init_tracker(logger=self.logger)

        self.logger.info(f"TrackerController Initialized")

    def add_geometries(self):
        for figure_id in self.figure_ids:
            figure = self.api.video.figure.get_info_by_id(figure_id)
            geometry = sly.deserialize_geometry(figure.geometry_type, figure.geometry)
            self.geometries.append(geometry)

    def add_frames_indexes(self):
        total_frames = self.api.video.get_info_by_id(self.video_id).frames_count
        cur_index = self.frame_index

        while 0 <= cur_index < total_frames and len(self.frames_indexes) < self.frames_count + 1:
            self.frames_indexes.append(cur_index)
            cur_index += 1 if self.direction == "forward" else -1

    def load_frames(self):
        rgbs = []
        for frame_index in self.frames_indexes:
            img_rgb = self.api.video.frame.download_np(self.video_id, frame_index)
            rgbs.append(torch.from_numpy(img_rgb).permute(2, 0, 1))
        self._w, self._h = rgbs[0].shape[1:]
        self.frames = torch.stack(rgbs, dim=0).unsqueeze(0)

    def track(self):
        for pos, (obj_id, geometry) in enumerate(zip(self.object_ids, self.geometries), start=1):
            self.logger.info(f"Start process obj #{obj_id} - {geometry.geometry_name()}")
            points = sly_functions.geometry_to_np(geometry)
            points = torch.tensor(points, dtype=float)
            with torch.no_grad():
                trajs = sly_functions.run_model(self.model, self.frames, points)

            for i, new_points in enumerate(trajs[1:], start=1):
                cur_pos = i + (pos - 1) * self.frames_count
                new_points = sly_functions.check_bounds(new_points, self._w, self._h)
                frame_index = self.frames_indexes[i]
                new_figure = sly_functions.np_to_geometry(new_points, geometry.geometry_name())
                self.api.video.figure.create(
                    self.video_id,
                    obj_id,
                    frame_index,
                    new_figure.to_json(),
                    new_figure.geometry_name(),
                    self.track_id,
                )

                stop = self._notify(cur_pos)
                if stop:
                    self.logger("Task stoped by user.")
                    self._notify(self.stop)
                    return

        self._notify(self.stop)

    def _notify(self, pos: int):
        return self.api.video.notify_progress(
            self.track_id,
            self.video_id,
            min(self.frames_indexes),
            max(self.frames_indexes),
            pos,
            self.stop,
        )

import gymnasium.spaces as gyms
import numpy as np

from molmo_spaces.env.abstract_sensors import Sensor


class CameraSensor(Sensor):
    """Sensor for RGB camera images from MuJoCo."""

    def __init__(
        self,
        camera_name: str = "camera",
        img_resolution: tuple[int, int] = (480, 480),
        uuid: str | None = None,
    ) -> None:
        self.camera_name = camera_name
        self.img_resolution = img_resolution

        if uuid is None:
            uuid = f"camera_{camera_name}"

        # Define observation space for RGB images
        width, height = img_resolution
        observation_space = gyms.Box(low=0, high=255, shape=(height, width, 3), dtype=np.uint8)
        super().__init__(uuid=uuid, observation_space=observation_space)

    def get_observation(self, env, task, batch_index: int = 0, *args, **kwargs) -> np.ndarray:
        """Get camera image from environment rendering."""

        # Use camera-specific frame access for multi-camera support
        # if hasattr(env, 'render_rgb_frame') and callable(env.render_rgb_frame):
        frame = env.render_rgb_frame(self.camera_name)

        if frame is not None:
            return frame

        # Return black image if no rendering available
        width, height = self.img_resolution
        return np.zeros((height, width, 3), dtype=np.uint8)


class DepthSensor(Sensor):
    """Sensor for depth images from MuJoCo.

    Returns raw metric depth in meters as float32. Encoding to RGB for video storage
    happens at save time. See molmo_spaces.utils.depth_utils for encoding/decoding functions.
    """

    def __init__(
        self,
        camera_name: str = "camera",
        img_resolution: tuple[int, int] = (480, 480),
        uuid: str | None = None,
    ) -> None:
        self.camera_name = camera_name
        self.img_resolution = img_resolution

        if uuid is None:
            uuid = f"depth_{camera_name}"

        # Define observation space for raw depth (float32 in meters)
        width, height = img_resolution
        observation_space = gyms.Box(low=0.0, high=10.0, shape=(height, width), dtype=np.float32)
        super().__init__(uuid=uuid, observation_space=observation_space)

    def get_observation(self, env, task, batch_index: int = 0, *args, **kwargs) -> np.ndarray:
        """Get depth image from environment rendering."""
        # Use camera-specific frame access for multi-camera support
        if hasattr(env, "render_depth_frame") and callable(env.render_depth_frame):
            frame = env.render_depth_frame(self.camera_name)
            if frame is not None:
                return frame

        # Fallback to default camera for backward compatibility
        if hasattr(env, "depth_frame") and env.depth_frame is not None:
            return env.depth_frame

        # Return zero depth if no rendering available
        width, height = self.img_resolution
        return np.zeros((height, width), dtype=np.float32)


class ProximityDepthBufferSensor(Sensor):
    """Returns a stack of sub-stepped depth frames captured during the policy step.

    Frames live on env._proximity_depth_frames[camera_name]; populated by
    task.step() at the configured proximity_sensor_period_ms. Output shape:
    (n_substeps, H, W) float32 meters where (H, W) is the SPAD output resolution
    (default 8x8). The renderer produces frames at the global camera resolution;
    we area-average them down here to the SPAD output dimensions.
    """

    def __init__(
        self,
        camera_name: str = "camera",
        img_resolution: tuple[int, int] = (8, 8),
        max_substeps: int = 8,
        uuid: str | None = None,
    ) -> None:
        self.camera_name = camera_name
        self.img_resolution = img_resolution  # (width, height) of SPAD output
        self.max_substeps = max_substeps

        if uuid is None:
            uuid = f"proximity_{camera_name}"

        width, height = img_resolution
        observation_space = gyms.Box(
            low=0.0,
            high=10.0,
            shape=(max_substeps, height, width),
            dtype=np.float32,
        )
        super().__init__(uuid=uuid, observation_space=observation_space)

    def _downsample(self, frame: np.ndarray) -> np.ndarray:
        out_w, out_h = self.img_resolution
        if frame.shape == (out_h, out_w):
            return frame.astype(np.float32)
        import cv2

        return cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA).astype(np.float32)

    def get_observation(self, env, task, batch_index: int = 0, *args, **kwargs) -> np.ndarray:
        frames = getattr(env, "_proximity_depth_frames", {}).get(self.camera_name, [])
        if not frames and hasattr(env, "record_proximity_depths"):
            # Post-reset call: no sim sub-step has run yet so the buffer is empty.
            # Render once now so the first observation is real depth — an all-zero
            # frame reads downstream as "contact at 0 m", which is worse than stale.
            env.record_proximity_depths([self.camera_name])
            frames = getattr(env, "_proximity_depth_frames", {}).get(self.camera_name, [])
        out_w, out_h = self.img_resolution
        out = np.zeros((self.max_substeps, out_h, out_w), dtype=np.float32)
        if not frames:
            return out
        downsampled = [self._downsample(f) for f in frames[: self.max_substeps]]
        # Left-pad by repeating the earliest frame (never zeros: a zero substep reads
        # as a 0 m return). Normal steps deliver exactly max_substeps frames, so the
        # padding only triggers on the post-reset observation.
        while len(downsampled) < self.max_substeps:
            downsampled.insert(0, downsampled[0])
        out[:] = np.stack(downsampled, axis=0)
        return out


_TURBO_LUT: np.ndarray | None = None


def _get_turbo_lut() -> np.ndarray:
    """256x3 uint8 RGB lookup table for the 'turbo' colormap (built once, cached)."""
    global _TURBO_LUT
    if _TURBO_LUT is None:
        ramp = np.arange(256, dtype=np.uint8).reshape(256, 1)
        try:
            import cv2

            bgr = cv2.applyColorMap(ramp, cv2.COLORMAP_TURBO).reshape(256, 3)
            _TURBO_LUT = np.ascontiguousarray(bgr[:, ::-1])  # BGR -> RGB
        except Exception:
            from matplotlib import cm

            _TURBO_LUT = (cm.turbo(np.linspace(0.0, 1.0, 256))[:, :3] * 255).astype(np.uint8)
    return _TURBO_LUT


def depth_to_turbo_rgb(depth: np.ndarray, near: float, far: float) -> np.ndarray:
    """Map a metric-depth (H, W) float array to an (H, W, 3) uint8 turbo RGB image.

    Depth is clipped to [near, far] so coloring is stable across frames/sensors; near
    surfaces map to the warm (red) end of turbo, far/empty pixels to the cool (blue) end.
    """
    lut = _get_turbo_lut()
    d = np.asarray(depth, dtype=np.float32)
    span = max(float(far) - float(near), 1e-6)
    norm = np.clip((float(far) - d) / span, 0.0, 1.0)  # near -> 1 (warm), far -> 0 (cool)
    idx = (norm * 255.0).astype(np.uint8)
    return lut[idx]


class ProximityVizRGBSensor(CameraSensor):
    """High-res RGB visualization of a proximity (SPAD) sensor's view.

    Only added when exp_config.viz_sensor_rgb is True. Subclasses CameraSensor so the
    save pipeline treats it exactly like any RGB camera (saved as an mp4, dropped before
    batching for memory). Unlike CameraSensor it renders through the dedicated
    proximity-viz renderer (cosmetic skin hidden, same MJCF camera/fovy as the native 8x8
    SPAD render -- only the resolution differs) rather than the shared global renderer.
    """

    def get_observation(self, env, task, batch_index: int = 0, *args, **kwargs) -> np.ndarray:
        rgb, _ = env.render_proximity_viz(self.camera_name)
        return rgb


class ProximityVizDepthSensor(CameraSensor):
    """High-res TURBO-colormapped depth visualization of a proximity (SPAD) sensor's view.

    Only added when exp_config.viz_sensor_rgb is True. Renders the same view as the native
    8x8 SPAD (cosmetic skin hidden, same MJCF camera/fovy) at viz_sensor_resolution, then
    maps the metric depth to a 3-channel turbo RGB image (near = warm, far = cool) so it is
    directly viewable. Subclasses CameraSensor so the save pipeline writes it as a normal
    RGB mp4 (NOT a lossless depth-encoded video); its uuid must therefore not end in
    ``_depth``. Shares the per-step render cache with ProximityVizRGBSensor (one scene
    render serves both viz sensors).
    """

    def __init__(
        self,
        camera_name: str = "camera",
        img_resolution: tuple[int, int] = (640, 480),
        uuid: str | None = None,
        depth_range: tuple[float, float] = (0.05, 4.0),
    ) -> None:
        super().__init__(camera_name=camera_name, img_resolution=img_resolution, uuid=uuid)
        self.depth_range = depth_range

    def get_observation(self, env, task, batch_index: int = 0, *args, **kwargs) -> np.ndarray:
        _, depth = env.render_proximity_viz(self.camera_name)
        near, far = self.depth_range
        return depth_to_turbo_rgb(depth, near, far)


class SegmentationSensor(Sensor):
    """Sensor for segmentation images from MuJoCo, outputs video-compatible arrays."""

    def __init__(
        self,
        camera_name: str = "camera",
        img_resolution: tuple[int, int] = (480, 480),
        uuid: str | None = None,
    ) -> None:
        self.camera_name = camera_name
        self.img_resolution = img_resolution

        if uuid is None:
            uuid = f"segmentation_{camera_name}"

        # Define observation space for uint8 images with channel dimension
        width, height = img_resolution
        observation_space = gyms.Box(low=0, high=255, shape=(height, width, 1), dtype=np.uint8)
        super().__init__(uuid=uuid, observation_space=observation_space)

    def get_observation(self, env, task, batch_index: int = 0, *args, **kwargs) -> np.ndarray:
        """Get segmentation image from environment rendering."""
        # Use camera-specific frame access for multi-camera support
        if hasattr(env, "segmentation_frame") and callable(env.segmentation_frame):
            frame = env.segmentation_frame(self.camera_name)
            if frame is not None:
                return frame

        # Fallback to default camera for backward compatibility
        if hasattr(env, "segmentation_frame") and env.segmentation_frame is not None:
            return env.segmentation_frame

        # Return zero segmentation if no rendering available
        width, height = self.img_resolution
        return np.zeros((height, width, 1), dtype=np.uint8)


class CameraParameterSensor(Sensor):
    """Sensor for camera parameters (intrinsics and extrinsics)."""

    def __init__(
        self,
        camera_name: str = "camera",
        uuid: str | None = None,
        img_resolution: tuple[int, int] = (480, 480),
    ) -> None:
        self.img_resolution = img_resolution
        self.camera_name = camera_name

        if uuid is None:
            uuid = f"camera_params_{camera_name}"

        observation_space = gyms.Dict(
            {
                "extrinsic_cv": gyms.Box(low=-np.inf, high=np.inf, shape=(3, 4), dtype=np.float32),
                "cam2world_gl": gyms.Box(low=-np.inf, high=np.inf, shape=(4, 4), dtype=np.float32),
                "intrinsic_cv": gyms.Box(low=-np.inf, high=np.inf, shape=(3, 3), dtype=np.float32),
            }
        )
        super().__init__(uuid=uuid, observation_space=observation_space)

    def get_observation(self, env, task, batch_index: int = 0, *args, **kwargs) -> dict:
        """Get camera parameters for a specific environment."""
        camera = env.camera_manager.registry[self.camera_name]
        world2cam = camera.get_pose()
        # Create extrinsic_cv (Computer Vision convention - world2cam)
        extrinsic_cv = np.linalg.inv(world2cam)[:3, :]  # 3x4 matrix
        cam2world_gl = world2cam

        height, width = self.img_resolution
        fovy_degrees = camera.fov

        # Convert field of view to focal length
        focal_length = (height / 2.0) / np.tan(np.radians(fovy_degrees / 2.0))

        # Create intrinsic matrix (assuming square pixels and centered principal point)
        intrinsic_cv = np.array(
            [[focal_length, 0, width / 2.0], [0, focal_length, height / 2.0], [0, 0, 1]],
            dtype=np.float32,
        )

        # Ensure consistent structure and ordering
        data = {
            "cam2world_gl": cam2world_gl.tolist(),
            "extrinsic_cv": extrinsic_cv.tolist(),
            "intrinsic_cv": intrinsic_cv.tolist(),
        }
        return data


# # Legacy sensors from other project (keeping for reference)
# class AgentsCameraParametersSensor(Sensor):
#     def __init__(
#         self,
#         uuid: str = "agent_camera_params",
#         str_max_len: Union[str, int] = 2000,
#     ) -> None:
#         assert isinstance(str_max_len, int)
#         self.str_max_len = str_max_len
#         observation_space = self._get_observation_space()
#         super().__init__(uuid=uuid, observation_space=observation_space)

#     def _get_observation_space(self) -> gyms.MultiDiscrete:
#         return gyms.Discrete(self.str_max_len)

#     def get_observation(self, env, task, *args, **kwargs) -> np.ndarray:
#         # Legacy implementation - would need adaptation for molmo-spaces
#         agent_parameter_sensors = {}
#         for which_cam in ["front", "left", "right", "down"]:
#             params = round_floats_in_dict(
#                 task.controller.camera_registry[which_cam]["camera_parameters"]
#             )
#             if params is not None and "camera_intrinsic" in params:
#                 del params[
#                     "camera_intrinsic"
#                 ]  # alternately, make it json-friendly. this seems fine though
#             agent_parameter_sensors[which_cam] = params
#         param_string = json.dumps(agent_parameter_sensors)
#         # Convert string to bytes array for gym compatibility
#         byte_array = np.zeros(self.str_max_len, dtype=np.uint8)
#         encoded = param_string.encode('utf-8')[:self.str_max_len]
#         byte_array[:len(encoded)] = list(encoded)
#         return byte_array


# class RawRGBCameraSensor(Sensor):
#     def __init__(self, uuid: str, height: int, width: int, which_camera: str):
#         self.height = height
#         self.width = width
#         self.which_camera = which_camera

#         observation_space = gyms.Box(
#             low=0, high=255,
#             shape=(height, width, 3),
#             dtype=np.uint8
#         )
#         super().__init__(uuid=uuid, observation_space=observation_space)

#     def get_observation(self, env, task, *args, **kwargs) -> Any:
#         # Legacy implementation - would need adaptation for molmo-spaces
#         frame = env.camera_registry[self.which_camera]["rgb"].copy()
#         if frame.shape[0] != self.height or frame.shape[1] != self.width:
#             import platform
#             if platform.system() != "Darwin":
#                 raise NotImplementedError(
#                     "Resizing the raw frames is a temp hack to get the warped and raw frames at "
#                     "the same time for visual comparison. If you are actually generating data, "
#                     "do not just bypass this check, fix get_core_sensors to actually be "
#                     "what you want."
#                 )
#             import cv2
#             frame = cv2.resize(frame, (self.width, self.height))
#         return frame

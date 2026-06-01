import math
import os

import numpy as np
import torch
from pyquaternion import Quaternion

from lib.scene import BoundingBox, LiDARSensor
from lib.utils.console_utils import *
from lib.utils.general_utils import matrix_to_quaternion
from t4_devkit import T4Devkit
from tqdm import tqdm


def _pointcloud_to_range_image(xyzs, intensities, H, W, inc_bottom, inc_top, max_depth=120.0):
    """Convert raw point cloud (in sensor frame) to range image.

    Args:
        xyzs: (N, 3) array of points in sensor coordinate frame.
        intensities: (N,) array of intensity values.
        H: Range image height (number of beams).
        W: Range image width (number of azimuth bins).
        inc_bottom: Bottom inclination angle in radians.
        inc_top: Top inclination angle in radians.
        max_depth: Maximum depth to include.

    Returns:
        range_image: (H, W, 2) array with [depth, intensity].
    """
    azimuth_left, azimuth_right = np.pi, -np.pi
    h_res = (azimuth_right - azimuth_left) / W
    v_res = (inc_bottom - inc_top) / H

    x, y, z = xyzs[:, 0], xyzs[:, 1], xyzs[:, 2]
    dists = np.linalg.norm(xyzs, axis=1)

    # Vectorized filtering
    valid = (dists <= max_depth) & (dists >= 0.1)
    x, y, z, dists = x[valid], y[valid], z[valid], dists[valid]
    valid_intensities = intensities[valid]

    # Vectorized projection
    azimuth = np.arctan2(y, x)
    inclination = np.arctan2(z, np.sqrt(x ** 2 + y ** 2))

    w_idx = np.round((azimuth - azimuth_left) / h_res).astype(np.int32)
    h_idx = np.round((inclination - inc_top) / v_res).astype(np.int32)

    # Bounds filtering
    in_bounds = (w_idx >= 0) & (w_idx < W) & (h_idx >= 0) & (h_idx < H)
    w_idx, h_idx = w_idx[in_bounds], h_idx[in_bounds]
    dists, valid_intensities = dists[in_bounds], valid_intensities[in_bounds]

    # Sort by distance (descending) so closer points overwrite farther ones
    order = np.argsort(-dists)
    w_idx, h_idx = w_idx[order], h_idx[order]
    dists, valid_intensities = dists[order], valid_intensities[order]

    range_map = np.zeros((H, W), dtype=np.float64)
    intensity_map = np.zeros((H, W), dtype=np.float64)
    range_map[h_idx, w_idx] = dists
    intensity_map[h_idx, w_idx] = valid_intensities

    range_image = np.stack([range_map, intensity_map], axis=-1)
    return range_image


def _get_sensor2ego(t4, channel):
    """Get sensor-to-ego transform for a given channel.

    Args:
        t4: T4Devkit instance.
        channel: Sensor channel name.

    Returns:
        sensor2ego: (4, 4) numpy array.
    """
    # Find the calibrated sensor for this channel
    for cs in t4.calibrated_sensor:
        sensor = t4.get("sensor", cs.sensor_token)
        if sensor.channel == channel:
            # Build 4x4 transform from translation + rotation
            q = Quaternion(cs.rotation)
            rotation_matrix = q.rotation_matrix
            sensor2ego = np.eye(4, dtype=np.float64)
            sensor2ego[:3, :3] = rotation_matrix
            sensor2ego[:3, 3] = np.array(cs.translation)
            return sensor2ego

    raise ValueError(f"Could not find calibrated sensor for channel: {channel}")


def _get_ego2world(t4, ego_pose_token):
    """Get ego-to-world transform from an ego pose token.

    Args:
        t4: T4Devkit instance.
        ego_pose_token: Token string for the ego pose.

    Returns:
        ego2world: (4, 4) numpy array.
    """
    ego_pose = t4.get("ego_pose", ego_pose_token)
    q = Quaternion(ego_pose.rotation)
    rotation_matrix = q.rotation_matrix
    ego2world = np.eye(4, dtype=np.float64)
    ego2world[:3, :3] = rotation_matrix
    ego2world[:3, 3] = np.array(ego_pose.translation)
    return ego2world


def load_t4_raw(base_dir, args):
    """Load T4 dataset and convert to LiDAR-RT format.

    Args:
        base_dir: Path to the T4 dataset root directory.
        args: Configuration arguments. Expected fields:
            - frame_length: [start_frame, end_frame]
            - data_type: "T4"
            - lidar_channel: LiDAR channel name (e.g. "LIDAR_TOP")
            - topic_mapping: dict mapping channel -> ROS topic (for rosbag)
            - range_image_width: Width of generated range image (default: 1024)
            - range_image_height: Height of generated range image (default: 64)
            - inc_bottom: Bottom inclination in degrees (default: -25.0)
            - inc_top: Top inclination in degrees (default: 15.0)
            - max_depth: Maximum depth in meters (default: 120.0)

    Returns:
        lidar: LiDARSensor instance.
        bboxes: Dict[str, BoundingBox] mapping object IDs to bounding boxes.
    """
    # Configuration
    lidar_channel = getattr(args, "lidar_channel", "LIDAR_TOP")
    topic_mapping = getattr(args, "topic_mapping", None)
    use_rosbag = getattr(args, "use_rosbag", False)
    W = getattr(args, "range_image_width", 1024)
    H = getattr(args, "range_image_height", 64)
    inc_bottom = math.radians(getattr(args, "inc_bottom", -25.0))
    inc_top = math.radians(getattr(args, "inc_top", 15.0))
    max_depth = getattr(args, "max_depth", 120.0)

    # Initialize T4Devkit
    t4 = T4Devkit(base_dir, use_rosbag=use_rosbag, topic_mapping=topic_mapping)

    # Get sensor calibration (sensor2ego)
    sensor2ego = _get_sensor2ego(t4, lidar_channel)

    # Find all sample_data entries for the target LiDAR channel, sorted by timestamp
    lidar_sample_datas = []
    for sd in t4.sample_data:
        if sd.channel == lidar_channel and sd.is_key_frame:
            lidar_sample_datas.append(sd)
    lidar_sample_datas.sort(key=lambda sd: sd.timestamp)

    frames = args.frame_length
    if frames[1] >= len(lidar_sample_datas):
        frames = [frames[0], len(lidar_sample_datas) - 1]
        print(yellow(f"Warning: Clamped frame range to [0, {frames[1]}]"))

    # Initialize LiDARSensor
    lidar = LiDARSensor(
        sensor2ego=sensor2ego,
        name=lidar_channel,
        inclination_bounds=(inc_bottom, inc_top),
        data_type=args.data_type,
    )

    # Cache directory for range images
    cache_dir = os.path.join(base_dir, "cache", lidar_channel)
    os.makedirs(cache_dir, exist_ok=True)

    # Process each frame
    for frame in tqdm(range(frames[0], frames[1] + 1)):
        sd = lidar_sample_datas[frame]

        # Get ego pose
        ego2world = _get_ego2world(t4, sd.ego_pose_token)

        # Check cache
        cache_path = os.path.join(cache_dir, f"range_image_frame_{frame}.pt")
        if os.path.exists(cache_path):
            cached = torch.load(cache_path, weights_only=True)
            range_image_r1 = cached["r1"]
            range_image_r2 = cached["r2"]
        else:
            # Get point cloud from rosbag (per-sensor, not concatenated)
            pc = t4.get_lidar_pointcloud(sd.token)
            # pc.points is (4, N) array: [x, y, z, intensity]
            points = pc.points.T  # (N, 4)
            xyzs = points[:, :3]  # sensor frame coordinates
            intensities = points[:, 3]

            # Convert point cloud to range image
            range_image_r1 = _pointcloud_to_range_image(
                xyzs, intensities, H, W, inc_bottom, inc_top, max_depth
            )
            range_image_r1[range_image_r1 == -1] = 0
            range_image_r2 = np.zeros_like(range_image_r1)

            # Convert to tensors and cache
            range_image_r1 = torch.from_numpy(range_image_r1).float()
            range_image_r2 = torch.from_numpy(range_image_r2).float()
            torch.save({"r1": range_image_r1, "r2": range_image_r2}, cache_path)

        lidar.add_frame(frame, ego2world, range_image_r1, range_image_r2)

    # Load bounding boxes from T4 annotations
    bboxes = _load_t4_bboxes(t4, lidar_sample_datas, frames, args)

    return lidar, bboxes


def _load_t4_bboxes(t4, lidar_sample_datas, frames, args):
    """Load 3D bounding boxes from T4 annotations.

    Args:
        t4: T4Devkit instance.
        lidar_sample_datas: List of sample_data sorted by timestamp.
        frames: [start_frame, end_frame].
        args: Configuration arguments.

    Returns:
        bboxes: Dict[str, BoundingBox].
    """
    bboxes = {}
    vehicle_categories = {"car", "truck", "bus", "bicycle", "motorcycle", "trailer"}

    for frame in range(frames[0], frames[1] + 1):
        sd = lidar_sample_datas[frame]

        # Use get_box3ds API to get annotations for this sample_data
        boxes = t4.get_box3ds(sd.token)

        for box in boxes:
            # Filter by vehicle categories
            label_name = box.semantic_label.name
            category_name = label_name.split(".")[-1] if "." in label_name else label_name
            if category_name not in vehicle_categories:
                continue

            object_id = box.uuid if box.uuid else str(id(box))
            size = np.array(box.size)  # (width, length, height)

            if object_id not in bboxes:
                bboxes[object_id] = BoundingBox(1, object_id, torch.tensor(size).float())

            # Box3D position and rotation are in world frame
            position = np.array(box.position)
            rotation_matrix = box.rotation.rotation_matrix

            pos_t = torch.from_numpy(position).float().cuda()
            R_t = torch.from_numpy(rotation_matrix).float().cuda()

            quaternion = matrix_to_quaternion(R_t)
            quaternion = quaternion / torch.norm(quaternion)
            quaternion = quaternion.unsqueeze(0)

            dT = torch.zeros(3).float().cuda()
            dR = torch.eye(3).float().cuda()
            bboxes[object_id].frame[frame] = (pos_t, quaternion, dT, dR)

    return bboxes

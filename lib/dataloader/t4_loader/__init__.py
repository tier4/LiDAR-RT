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
    """Get sensor-to-ego transform for a given channel from T4 DB."""
    for cs in t4.calibrated_sensor:
        sensor = t4.get("sensor", cs.sensor_token)
        if sensor.channel == channel:
            q = Quaternion(cs.rotation)
            rotation_matrix = q.rotation_matrix
            sensor2ego = np.eye(4, dtype=np.float64)
            sensor2ego[:3, :3] = rotation_matrix
            sensor2ego[:3, 3] = np.array(cs.translation)
            return sensor2ego
    raise ValueError(f"Could not find calibrated sensor for channel: {channel}")


def _build_sensor2ego(translation, rotation_wxyz):
    """Build sensor2ego 4x4 matrix from translation and quaternion (wxyz)."""
    q = Quaternion(rotation_wxyz)
    sensor2ego = np.eye(4, dtype=np.float64)
    sensor2ego[:3, :3] = q.rotation_matrix
    sensor2ego[:3, 3] = np.array(translation)
    return sensor2ego


def _get_ego2world(t4, ego_pose_token):
    """Get ego-to-world transform from an ego pose token."""
    ego_pose = t4.get("ego_pose", ego_pose_token)
    q = Quaternion(ego_pose.rotation)
    rotation_matrix = q.rotation_matrix
    ego2world = np.eye(4, dtype=np.float64)
    ego2world[:3, :3] = rotation_matrix
    ego2world[:3, 3] = np.array(ego_pose.translation)
    return ego2world


def load_t4_raw(base_dir, args):
    """Load T4 dataset with multi-LiDAR support.

    If args.lidar_sensors is defined, loads each sensor individually from
    rosbag via t4-devkit. Otherwise falls back to single-channel loading.

    Returns:
        lidars: dict[str, LiDARSensor] — one sensor per channel.
        bboxes: dict[str, BoundingBox] — shared across all sensors.
    """
    lidar_sensors_cfg = getattr(args, "lidar_sensors", None)

    if lidar_sensors_cfg:
        return _load_t4_multi_lidar(base_dir, args, lidar_sensors_cfg)
    else:
        # Fallback: single-channel mode (backward compat)
        lidar, bboxes = _load_t4_single_lidar(base_dir, args)
        sensor_name = getattr(args, "lidar_channel", "LIDAR_CONCAT")
        return {sensor_name: lidar}, bboxes


def _load_t4_multi_lidar(base_dir, args, lidar_sensors_cfg):
    """Load multiple LiDAR sensors from rosbag.

    Uses LIDAR_CONCAT sample_data for timestamps and ego poses,
    then reads each sensor's pandar_packets from rosbag at matching timestamps.
    """
    from t4_devkit.rosbag import Rosbag2Reader, TopicMapping

    # Load T4 DB for timestamps, ego poses, and bounding boxes
    t4 = T4Devkit(base_dir, use_rosbag=False)

    # Get LIDAR_CONCAT sample_data for timestamps and ego poses
    lidar_sample_datas = []
    for sd in t4.sample_data:
        if sd.channel == "LIDAR_CONCAT" and sd.is_key_frame:
            lidar_sample_datas.append(sd)
    lidar_sample_datas.sort(key=lambda sd: sd.timestamp)

    frames = list(args.frame_length)
    if frames[1] >= len(lidar_sample_datas):
        frames = [frames[0], len(lidar_sample_datas) - 1]
        print(yellow(f"Warning: Clamped frame range to [0, {frames[1]}]"))

    # Collect timestamps and ego poses from LIDAR_CONCAT sample_data
    frame_timestamps = {}  # frame_id -> timestamp_us
    frame_ego2world = {}   # frame_id -> (4, 4) numpy array
    for frame in range(frames[0], frames[1] + 1):
        sd = lidar_sample_datas[frame]
        frame_timestamps[frame] = sd.timestamp
        frame_ego2world[frame] = _get_ego2world(t4, sd.ego_pose_token)

    # Open rosbag reader with all sensor topics
    bag_dir = os.path.join(base_dir, "input_bag")
    topic_mappings = [
        TopicMapping(channel=s["name"], topic=s["topic"])
        for s in lidar_sensors_cfg
    ]
    reader = Rosbag2Reader(bag_dir, topic_mapping=topic_mappings)

    # Load each sensor
    lidars = {}
    for sensor_cfg in lidar_sensors_cfg:
        sensor_name = sensor_cfg["name"]
        topic = sensor_cfg["topic"]
        H = sensor_cfg.get("range_image_height", 64)
        W = sensor_cfg.get("range_image_width", 1800)
        inc_bottom = math.radians(sensor_cfg.get("inc_bottom", -25.0))
        inc_top = math.radians(sensor_cfg.get("inc_top", 15.0))
        max_depth = sensor_cfg.get("max_depth", 200.0)

        # Build sensor2ego transform
        s2e_trans = sensor_cfg.get("sensor2ego_translation", [0, 0, 0])
        s2e_rot = sensor_cfg.get("sensor2ego_rotation", [1, 0, 0, 0])
        sensor2ego = _build_sensor2ego(s2e_trans, s2e_rot)

        lidar = LiDARSensor(
            sensor2ego=sensor2ego,
            name=sensor_name,
            inclination_bounds=(inc_bottom, inc_top),
            data_type="T4",
        )

        # Cache directory
        cache_dir = os.path.join(base_dir, "cache", sensor_name)
        os.makedirs(cache_dir, exist_ok=True)

        print(f"  Loading sensor: {sensor_name} ({topic})")
        for frame in tqdm(range(frames[0], frames[1] + 1), desc=f"  {sensor_name}"):
            ego2world = frame_ego2world[frame]
            timestamp_us = frame_timestamps[frame]

            cache_path = os.path.join(cache_dir, f"range_image_frame_{frame}.pt")
            if os.path.exists(cache_path):
                cached = torch.load(cache_path, weights_only=True)
                range_image_r1 = cached["r1"]
                range_image_r2 = cached["r2"]
            else:
                # Read point cloud from rosbag via pandar decoder
                try:
                    pc = reader.get_pointcloud(sensor_name, timestamp_us)
                    points = pc.points.T  # (N, 4)
                    xyzs = points[:, :3]
                    intensities = points[:, 3]
                except (KeyError, ValueError) as e:
                    print(yellow(f"    Warning: frame {frame} skipped for {sensor_name}: {e}"))
                    xyzs = np.zeros((0, 3))
                    intensities = np.zeros(0)

                range_image_r1 = _pointcloud_to_range_image(
                    xyzs, intensities, H, W, inc_bottom, inc_top, max_depth
                )
                range_image_r1[range_image_r1 == -1] = 0
                range_image_r2 = np.zeros_like(range_image_r1)

                range_image_r1 = torch.from_numpy(range_image_r1).float()
                range_image_r2 = torch.from_numpy(range_image_r2).float()
                torch.save({"r1": range_image_r1, "r2": range_image_r2}, cache_path)

            lidar.add_frame(frame, ego2world, range_image_r1, range_image_r2)

        lidars[sensor_name] = lidar

    reader.close()

    # Load bounding boxes (shared across all sensors)
    bboxes = _load_t4_bboxes(t4, lidar_sample_datas, frames, args)

    return lidars, bboxes


def _load_t4_single_lidar(base_dir, args):
    """Load T4 dataset with single LiDAR channel (backward compat)."""
    lidar_channel = getattr(args, "lidar_channel", "LIDAR_TOP")
    topic_mapping = getattr(args, "topic_mapping", None)
    use_rosbag = getattr(args, "use_rosbag", False)
    W = getattr(args, "range_image_width", 1024)
    H = getattr(args, "range_image_height", 64)
    inc_bottom = math.radians(getattr(args, "inc_bottom", -25.0))
    inc_top = math.radians(getattr(args, "inc_top", 15.0))
    max_depth = getattr(args, "max_depth", 120.0)

    t4 = T4Devkit(base_dir, use_rosbag=use_rosbag, topic_mapping=topic_mapping)

    sensor2ego = _get_sensor2ego(t4, lidar_channel)

    lidar_sample_datas = []
    for sd in t4.sample_data:
        if sd.channel == lidar_channel and sd.is_key_frame:
            lidar_sample_datas.append(sd)
    lidar_sample_datas.sort(key=lambda sd: sd.timestamp)

    frames = list(args.frame_length)
    if frames[1] >= len(lidar_sample_datas):
        frames = [frames[0], len(lidar_sample_datas) - 1]
        print(yellow(f"Warning: Clamped frame range to [0, {frames[1]}]"))

    lidar = LiDARSensor(
        sensor2ego=sensor2ego,
        name=lidar_channel,
        inclination_bounds=(inc_bottom, inc_top),
        data_type=args.data_type,
    )

    cache_dir = os.path.join(base_dir, "cache", lidar_channel)
    os.makedirs(cache_dir, exist_ok=True)

    for frame in tqdm(range(frames[0], frames[1] + 1)):
        sd = lidar_sample_datas[frame]
        ego2world = _get_ego2world(t4, sd.ego_pose_token)

        cache_path = os.path.join(cache_dir, f"range_image_frame_{frame}.pt")
        if os.path.exists(cache_path):
            cached = torch.load(cache_path, weights_only=True)
            range_image_r1 = cached["r1"]
            range_image_r2 = cached["r2"]
        else:
            pc = t4.get_lidar_pointcloud(sd.token)
            points = pc.points.T  # (N, 4)
            xyzs = points[:, :3]
            intensities = points[:, 3]

            range_image_r1 = _pointcloud_to_range_image(
                xyzs, intensities, H, W, inc_bottom, inc_top, max_depth
            )
            range_image_r1[range_image_r1 == -1] = 0
            range_image_r2 = np.zeros_like(range_image_r1)

            range_image_r1 = torch.from_numpy(range_image_r1).float()
            range_image_r2 = torch.from_numpy(range_image_r2).float()
            torch.save({"r1": range_image_r1, "r2": range_image_r2}, cache_path)

        lidar.add_frame(frame, ego2world, range_image_r1, range_image_r2)

    bboxes = _load_t4_bboxes(t4, lidar_sample_datas, frames, args)
    return lidar, bboxes


def _load_t4_bboxes(t4, lidar_sample_datas, frames, args):
    """Load 3D bounding boxes from T4 annotations."""
    bboxes = {}
    vehicle_categories = {"car", "truck", "bus", "bicycle", "motorcycle", "trailer"}

    for frame in range(frames[0], frames[1] + 1):
        sd = lidar_sample_datas[frame]
        boxes = t4.get_box3ds(sd.token)

        for box in boxes:
            label_name = box.semantic_label.name
            category_name = label_name.split(".")[-1] if "." in label_name else label_name
            if category_name not in vehicle_categories:
                continue

            object_id = box.uuid if box.uuid else str(id(box))
            size = np.array(box.size)  # (width, length, height)

            if object_id not in bboxes:
                bboxes[object_id] = BoundingBox(1, object_id, torch.tensor(size).float())

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

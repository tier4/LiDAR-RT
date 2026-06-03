import csv
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

# Directory containing Hesai angle correction CSV files
_HESAI_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "hesai")

# Beam table cache: sensor_type -> list of elevation angles in radians (top to bottom)
_BEAM_TABLE_CACHE = {}

# Ego vehicle crop box (base_link frame = rear axle center)
# JPN Taxi Gen2 (aip_xx1_gen2)
_EGO_CROP_BOX = {
    "min_x": -0.85,  "max_x": 3.55,
    "min_y": -0.8475, "max_y": 0.8475,
    "min_z": 0.0,     "max_z": 2.5,
}
# Side mirror crop box
_MIRROR_CROP_BOX = {
    "min_x": 2.84,  "max_x": 3.16,
    "min_y": -1.03, "max_y": 1.03,
    "min_z": 0.86,  "max_z": 1.22,
}


def _is_inside_ego(xyzs_ego):
    """Return boolean mask: True for points inside ego vehicle (to be removed).

    Checks against the vehicle body box and the mirror box.
    Args:
        xyzs_ego: (N, 3) numpy array in ego (base_link) frame.
    Returns:
        (N,) boolean array, True = inside ego, should be masked out.
    """
    x, y, z = xyzs_ego[:, 0], xyzs_ego[:, 1], xyzs_ego[:, 2]
    b = _EGO_CROP_BOX
    in_body = (
        (x >= b["min_x"]) & (x <= b["max_x"]) &
        (y >= b["min_y"]) & (y <= b["max_y"]) &
        (z >= b["min_z"]) & (z <= b["max_z"])
    )
    m = _MIRROR_CROP_BOX
    in_mirror = (
        (x >= m["min_x"]) & (x <= m["max_x"]) &
        (y >= m["min_y"]) & (y <= m["max_y"]) &
        (z >= m["min_z"]) & (z <= m["max_z"])
    )
    return in_body | in_mirror


def _origin_inside_box(origin, box):
    """Check if a single origin point is inside an AABB."""
    return (box["min_x"] <= origin[0] <= box["max_x"] and
            box["min_y"] <= origin[1] <= box["max_y"] and
            box["min_z"] <= origin[2] <= box["max_z"])


def _ray_intersects_ego(ray_origins, ray_dirs, sensor_origin_ego):
    """Check if rays intersect the ego vehicle bounding box (body + mirrors).

    Only masks rays from sensors OUTSIDE the ego bbox. If the sensor origin is
    inside the bbox (e.g., roof-mounted), no rays are masked for that box.

    Uses slab method for ray-AABB intersection.
    Args:
        ray_origins: (N, 3) numpy array, ray origin in ego frame.
        ray_dirs: (N, 3) numpy array, ray direction (need not be unit).
        sensor_origin_ego: (3,) numpy array, sensor position in ego frame.
    Returns:
        (N,) boolean array, True = ray intersects ego bbox.
    """
    def _ray_aabb(origins, dirs, box):
        """Slab method: returns True if ray hits the AABB at t > 0 from outside."""
        bmin = np.array([box["min_x"], box["min_y"], box["min_z"]])
        bmax = np.array([box["max_x"], box["max_y"], box["max_z"]])

        inv_dir = np.where(np.abs(dirs) > 1e-12, 1.0 / dirs, np.sign(dirs) * 1e12)
        t1 = (bmin - origins) * inv_dir
        t2 = (bmax - origins) * inv_dir

        tmin = np.minimum(t1, t2).max(axis=1)  # entry
        tmax = np.maximum(t1, t2).min(axis=1)  # exit

        # Only count hits where entry is at t > 0 (ray enters from outside)
        return (tmax > np.maximum(tmin, 0.0)) & (tmin > 0.0)

    hits = np.zeros(len(ray_origins), dtype=bool)
    if not _origin_inside_box(sensor_origin_ego, _EGO_CROP_BOX):
        hits |= _ray_aabb(ray_origins, ray_dirs, _EGO_CROP_BOX)
    if not _origin_inside_box(sensor_origin_ego, _MIRROR_CROP_BOX):
        hits |= _ray_aabb(ray_origins, ray_dirs, _MIRROR_CROP_BOX)
    return hits


def _compute_ego_mask(sensor2ego, inclination_bounds, H, W):
    """Compute per-pixel ego intersection mask for a sensor's range image.

    Args:
        sensor2ego: (4, 4) numpy array.
        inclination_bounds: list of per-beam radians (bottom-to-top) or tuple of (min, max).
        H, W: range image dimensions.
    Returns:
        (H, W) boolean numpy array, True = ray intersects ego (should be masked).
    """
    # Build ray directions in sensor frame (same logic as LiDARSensor.get_range_rays)
    x = (np.arange(W, 0, -1, dtype=np.float64)) / float(W)
    azimuth = x * 2.0 * np.pi - np.pi  # (W,)

    if isinstance(inclination_bounds, (list,)) and len(inclination_bounds) > 2:
        # Per-beam angles (bottom-to-top), flip to top-to-bottom for row order
        beam_angles = np.array(inclination_bounds[::-1], dtype=np.float64)  # (H,)
    else:
        inc_min, inc_max = inclination_bounds[0], inclination_bounds[1]
        row_idx = (np.arange(H, 0, -1, dtype=np.float64)) / float(H)
        beam_angles = row_idx * (inc_max - inc_min) + inc_min  # (H,)

    # Meshgrid: (H, W)
    elev, az = np.meshgrid(beam_angles, azimuth, indexing="ij")

    # Ray directions in sensor frame
    rays_x = np.cos(elev) * np.cos(az)
    rays_y = np.cos(elev) * np.sin(az)
    rays_z = np.sin(elev)
    rays_sensor = np.stack([rays_x, rays_y, rays_z], axis=-1)  # (H, W, 3)

    # Transform to ego frame
    R = sensor2ego[:3, :3]
    t = sensor2ego[:3, 3]
    rays_ego = rays_sensor @ R.T  # (H, W, 3) direction in ego frame
    origins_ego = np.broadcast_to(t, (H, W, 3))  # sensor origin in ego frame

    # Flatten and test
    rays_flat = rays_ego.reshape(-1, 3)
    origins_flat = origins_ego.reshape(-1, 3)
    hits = _ray_intersects_ego(origins_flat, rays_flat, t)

    return hits.reshape(H, W)


def _load_beam_table(sensor_type):
    """Load beam elevation table from Hesai angle correction CSV.

    Returns:
        beam_elevations: numpy array of elevation angles in radians,
                         sorted top (positive) to bottom (negative).
    """
    if sensor_type in _BEAM_TABLE_CACHE:
        return _BEAM_TABLE_CACHE[sensor_type]

    csv_map = {
        "OT128": "OT128_Angle-Correction-File-1.csv",
        "XT32": "XT32_Angle_Correction_File-1.csv",
        "XT16": "XT16_Angle_Correction_File-1.csv",
    }
    filename = csv_map.get(sensor_type)
    if filename is None:
        return None

    csv_path = os.path.join(_HESAI_DATA_DIR, filename)
    if not os.path.exists(csv_path):
        print(yellow(f"Warning: Beam table not found: {csv_path}"))
        return None

    elevations_deg = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            elevations_deg.append(float(row["Elevation"]))

    # Sort top (most positive) to bottom (most negative) — this is row order in range image
    elevations_deg.sort(reverse=True)
    beam_elevations = np.radians(np.array(elevations_deg))

    _BEAM_TABLE_CACHE[sensor_type] = beam_elevations
    return beam_elevations


def _pointcloud_to_range_image_with_beam_table(xyzs, intensities, beam_elevations, W, max_depth=120.0):
    """Convert point cloud to range image using actual beam elevation angles.

    Each row in the output corresponds to a physical beam, eliminating empty
    rows caused by non-uniform beam spacing projected onto a uniform grid.

    Args:
        xyzs: (N, 3) array of points in sensor coordinate frame.
        intensities: (N,) array of intensity values.
        beam_elevations: (H,) array of elevation angles in radians, sorted top to bottom.
        W: Range image width (number of azimuth bins).
        max_depth: Maximum depth to include.

    Returns:
        range_image: (H, W, 2) array with [depth, intensity].
    """
    H = len(beam_elevations)
    azimuth_left, azimuth_right = np.pi, -np.pi
    h_res = (azimuth_right - azimuth_left) / W

    x, y, z = xyzs[:, 0], xyzs[:, 1], xyzs[:, 2]
    dists = np.linalg.norm(xyzs, axis=1)

    valid = (dists <= max_depth) & (dists >= 0.1)
    x, y, z, dists = x[valid], y[valid], z[valid], dists[valid]
    valid_intensities = intensities[valid]

    azimuth = np.arctan2(y, x)
    inclination = np.arctan2(z, np.sqrt(x ** 2 + y ** 2))

    w_idx = np.round((azimuth - azimuth_left) / h_res).astype(np.int32)

    # Find nearest beam for each point using searchsorted on ascending array
    beam_asc = beam_elevations[::-1].copy()  # ascending order
    insert_idx = np.searchsorted(beam_asc, inclination)
    insert_idx = np.clip(insert_idx, 1, len(beam_asc) - 1)
    left_diff = np.abs(inclination - beam_asc[insert_idx - 1])
    right_diff = np.abs(inclination - beam_asc[insert_idx])
    nearest_asc = np.where(left_diff <= right_diff, insert_idx - 1, insert_idx)
    h_idx = (H - 1 - nearest_asc).astype(np.int32)  # map back to descending order

    in_bounds = (w_idx >= 0) & (w_idx < W) & (h_idx >= 0) & (h_idx < H)
    w_idx, h_idx = w_idx[in_bounds], h_idx[in_bounds]
    dists, valid_intensities = dists[in_bounds], valid_intensities[in_bounds]

    order = np.argsort(-dists)
    w_idx, h_idx = w_idx[order], h_idx[order]
    dists, valid_intensities = dists[order], valid_intensities[order]

    range_map = np.zeros((H, W), dtype=np.float64)
    intensity_map = np.zeros((H, W), dtype=np.float64)
    range_map[h_idx, w_idx] = dists
    intensity_map[h_idx, w_idx] = valid_intensities

    range_image = np.stack([range_map, intensity_map], axis=-1)
    return range_image


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
    sensor2ego transforms are read from /tf_static via t4-devkit.
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
        TopicMapping(
            channel=s["name"],
            topic=s["topic"],
            sensor_type=s.get("sensor_type"),
            frame_id=s.get("frame_id"),
        )
        for s in lidar_sensors_cfg
    ]
    reader = Rosbag2Reader(bag_dir, topic_mapping=topic_mappings)

    # Load each sensor
    lidars = {}
    for sensor_cfg in lidar_sensors_cfg:
        sensor_name = sensor_cfg["name"]
        topic = sensor_cfg["topic"]
        frame_id = sensor_cfg.get("frame_id", sensor_name)
        sensor_type = sensor_cfg.get("sensor_type")
        W = sensor_cfg.get("range_image_width", 1800)
        max_depth = sensor_cfg.get("max_depth", 200.0)

        # Load beam elevation table if available
        beam_table = _load_beam_table(sensor_type) if sensor_type else None

        if beam_table is not None:
            H = len(beam_table)
            # inclination_bounds as list (bottom-to-top, radians) for LiDARSensor
            inclination_bounds = beam_table[::-1].tolist()  # reverse: bottom to top
            print(f"  {sensor_name}: using {sensor_type} beam table ({H} beams, "
                  f"[{math.degrees(inclination_bounds[0]):.1f}°, {math.degrees(inclination_bounds[-1]):.1f}°])")
        else:
            H = sensor_cfg.get("range_image_height", 64)
            inc_bottom = math.radians(sensor_cfg.get("inc_bottom", -25.0))
            inc_top = math.radians(sensor_cfg.get("inc_top", 15.0))
            inclination_bounds = (inc_bottom, inc_top)

        # Get sensor2ego from /tf_static in rosbag
        sensor2ego = reader.get_sensor2ego(frame_id)
        ego2sensor = np.linalg.inv(sensor2ego)
        print(f"  {sensor_name}: sensor2ego t=[{sensor2ego[0,3]:.3f}, {sensor2ego[1,3]:.3f}, {sensor2ego[2,3]:.3f}]")

        # Compute ego vehicle ray intersection mask (static per sensor)
        ego_ray_mask = _compute_ego_mask(sensor2ego, inclination_bounds, H, W)
        ego_masked_pixels = ego_ray_mask.sum()
        if ego_masked_pixels > 0:
            print(f"  {sensor_name}: ego mask blocks {ego_masked_pixels}/{H*W} pixels "
                  f"({ego_masked_pixels/(H*W)*100:.1f}%)")

        lidar = LiDARSensor(
            sensor2ego=sensor2ego,
            name=sensor_name,
            inclination_bounds=inclination_bounds,
            data_type="T4",
        )

        # Cache directory (v5: beam-table + ego ray masking)
        cache_dir = os.path.join(base_dir, "cache_v5", sensor_name)
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
                # Read point cloud from rosbag (auto-transformed to ego frame)
                try:
                    pc = reader.get_pointcloud(sensor_name, timestamp_us)
                    points = pc.points.T  # (N, 4)
                    # Filter out points inside ego vehicle (base_link frame)
                    xyzs_ego = points[:, :3]
                    ego_mask = ~_is_inside_ego(xyzs_ego)
                    points = points[ego_mask]
                    # Transform ego frame → sensor frame for range image creation
                    xyzs_ego = points[:, :3]
                    xyzs_h = np.hstack([xyzs_ego, np.ones((len(xyzs_ego), 1))])
                    xyzs_sensor = (ego2sensor @ xyzs_h.T).T[:, :3]
                    intensities = points[:, 3]
                except (KeyError, ValueError) as e:
                    print(yellow(f"    Warning: frame {frame} skipped for {sensor_name}: {e}"))
                    xyzs_sensor = np.zeros((0, 3))
                    intensities = np.zeros(0)

                if beam_table is not None:
                    range_image_r1 = _pointcloud_to_range_image_with_beam_table(
                        xyzs_sensor, intensities, beam_table, W, max_depth
                    )
                else:
                    range_image_r1 = _pointcloud_to_range_image(
                        xyzs_sensor, intensities, H, W,
                        inclination_bounds[0], inclination_bounds[1], max_depth
                    )
                range_image_r1[range_image_r1 == -1] = 0
                # Zero out pixels where rays pass through ego vehicle
                range_image_r1[ego_ray_mask] = 0
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

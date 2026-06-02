import os

import numpy
import torch
from lib.dataloader import kitti_loader, waymo_loader, t4_loader
from lib.dataloader.gs_loader import SceneLidar
from lib.utils.console_utils import *


def load_scene(data_dir, args, test=False):
    if getattr(args, "data_type", None) == "T4":
        print(blue("\n====== [Loading] T4 Dataset ======"))
        # T4 loader already returns dict[str, LiDARSensor]
        lidars_dict, bboxes = t4_loader.load_t4_raw(data_dir, args)
    elif "waymo" in data_dir:
        print(blue("\n====== [Loading] Waymo Open Dataset ======"))
        lidar, bboxes = waymo_loader.load_waymo_raw(data_dir, args)
        lidars_dict = {"waymo_top": lidar}
    elif "kitti" in data_dir:
        print(blue("\n====== [Loading] KITTI Dataset ======"))
        lidar, bboxes = kitti_loader.load_kitti_raw(data_dir, args)
        lidars_dict = {"kitti": lidar}
    else:
        raise ValueError("Error: invalid dataset")

    print(blue("------------"))
    scene = SceneLidar(args, (lidars_dict, bboxes), test=test)
    return scene

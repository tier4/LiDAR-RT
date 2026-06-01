import os

import numpy
import torch
from lib.dataloader import kitti_loader, waymo_loader, t4_loader
from lib.dataloader.gs_loader import SceneLidar
from lib.utils.console_utils import *


def load_scene(data_dir, args, test=False):
    if getattr(args, "data_type", None) == "T4":
        print(blue("\n====== [Loading] T4 Dataset ======"))
        lidars, bboxes = t4_loader.load_t4_raw(data_dir, args)
    elif "waymo" in data_dir:
        print(blue("\n====== [Loading] Waymo Open Dataset ======"))
        lidars, bboxes = waymo_loader.load_waymo_raw(data_dir, args)
    elif "kitti" in data_dir:
        print(blue("\n====== [Loading] KITTI Dataset ======"))
        lidars, bboxes = kitti_loader.load_kitti_raw(data_dir, args)
    else:
        raise ValueError("Error: invalid dataset")

    print(blue("------------"))
    scene = SceneLidar(args, (lidars, bboxes), test=test)
    return scene

import os
import sys

from easydict import EasyDict

cfg = EasyDict()

"""
Path Settings
"""
cfg.CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
cfg.LIB_DIR = os.path.dirname(cfg.CONFIG_DIR)
cfg.ROOT_DIR = os.path.dirname(cfg.LIB_DIR)
cfg.ASSETS_DIR = os.path.join(cfg.ROOT_DIR, "assets")
cfg.GUI_DIR = os.path.join(cfg.ROOT_DIR, "gui")
cfg.DLL_DIR = os.path.join(cfg.LIB_DIR, "DobotDLL")
cfg.UTILS_DIR = os.path.join(cfg.LIB_DIR, "utils")


def add_path():
    """システムのファイルパスを設定するための関数"""

    for key, value in cfg.items():
        if "DIR" in key:
            sys.path.insert(0, value)


add_path()
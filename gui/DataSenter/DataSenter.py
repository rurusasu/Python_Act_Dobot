import sys, os

sys.path.append(".")
sys.path.append("..")
sys.path.append("../../")

from glob import glob

import json
import numpy as np
from PIL import Image

from src.config.config import cfg

"""
保存する時のファイル名
画像名_channel_size_w_size_h
"""

# ファイル名を取得
files = [
    p
    # for p in glob(cfg.TEST_DIR + os.sep + "**" + os.sep + "*.png", recursive=True)
    for p in glob(cfg.TEST_DIR + os.sep + "**")
    # if os.path.isfile(p)
]

# 取得したファイルリストから画像情報を読み出す
data = dict()
# datas = []
for f in files:
    # TESTディレクトリ内のディレクトリを検索
    if os.path.isdir(f):
        datas = []
        img_path_list = glob(f + os.sep + "*.png", recursive=True)
        for path in img_path_list:
            # img = np.array(Image.open(path))
            img = Image.open(path)  # 画像読み出し

            # もし、読み出した画像が "RGBA" の場合
            if img.mode == "RGBA":
                img = img.convert("RGB")  # RGBA -> RGB
                img.save(path)  # 上書き保存

            mode = img.mode
            img = np.array(img)  # pillow -> np
            name = os.path.basename(path)
            path = path
            if len(img.shape) < 3:
                ch = 1
                w, h = img.shape
            # elif img.shape[2] == 4:

            else:
                w, h, ch = img.shape

            datas.append(
                {
                    "name": name,
                    "path": path,
                    "mode": mode,
                    "channel": str(ch),
                    "width": str(w),
                    "height": str(h),
                }
            )
        parent_dir_name = os.path.basename(f.rstrip(os.sep))
        data[parent_dir_name] = datas


# 辞書オブジェクトをJSONファイルへ出力
save_path = cfg.TEST_DIR + os.sep + "data.json"
with open(save_path, mode="wt", encoding="utf_8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)


import datetime as dt
import os
from typing import Literal, Union

import cv2
import numpy as np

from utils.base_utils import makedir


def create_dir_name_date(root_pth: str, dir_name: str) -> str:
    """日付の名前が着いたディレクトリを作成するための関数

    Args:
        root_pth (str): ルートディレクトリ
        dir_name (str): 日付に追加する文字列

    Returns:
        str: 作成したディレクトリ名を含めた絶対パス
    """
    # 現在時刻を取得
    dt_now = dt.datetime.now()
    # フォルダ名用にyyyymmddの文字列を取得する
    today = dt_now.strftime("%Y%m%d%H%M")

    file_pth = os.path.join(root_pth, today+"_"+dir_name)
    makedir(file_pth)
    return file_pth



def save_img(img: np.ndarray, dir_name: str, file_name: Union[str, None] = None, ext: Literal["png", "jpg", "bmp"] = "png") -> int:
    """
    画像保存用の関数．

    Args:
        img (np.ndarray): 保存用の画像．
        dir_name (str): 画像の保存先のディレクトリ．
        ext (Literal[, optional): 保存する際の拡張子. Defaults to "png".

    Returns:
        int: 保存のフラグ
    """
    # 画像が ndarray 配列か確認
    if type(img) != np.ndarray:
        return -1
    else:
        dst = img.copy()
        dst = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR) # RGB -> BGR
    # ディレクトリが存在するか確認
    if not os.path.isdir(dir_name):
        return -2
    # 画像保存用の拡張子の選択
    if ext == "png": ext = ".png"
    elif ext == "jpg": ext = ".jpg"
    elif ext == "bmp": ext = ".bmp"
    else: return -3

    # 現在時刻を取得
    dt_now = dt.datetime.now()
    # フォルダ名用にyyyymmddの文字列を取得する
    today = dt_now.strftime("%Y%m%d%H%M")
    if type(file_name) == str:
        filepth = os.path.join(dir_name, today+"_"+file_name+ext)
    else:
        filepth = os.path.join(dir_name, today+ext)

    cv2.imwrite(filepth, dst)

import sys, os

sys.path.append(".")
sys.path.append("..")
sys.path.append("../../")

import cv2
import numpy as np
from PIL import Image


def rgb_to_srgb(img: np.ndarray, max_value: int = 255):
    """
    RGB画像->sRGB画像に変換する関数
    以下のサイトを参考に作成
    https://mikio.hatenablog.com/entry/2018/09/10/213756

    Args:
        img (np.ndarray):
            RGB画像
        max_value (int optional):
            画素値の最大値

    Return:
        dst (np.ndarray):
            sRGBに変換された画像
    """

    # 入力画像がRGB画像でない場合
    if img.shape[2] != 3:
        raise ValueError("Channel Error: {}".format(img.shape[2]))

    dst = img.copy()
    w, h, ch = dst.shape

    for i in range(0, w):
        for j in range(0, h):
            for k in range(0, 3):
                dst[i][j][k] = _rgb_to_srgb(dst[i][j][k], quantum_max=max_value)

    return dst


def _rgb_to_srgb(value, quantum_max=1):
    if value <= 0.0031308:
        return value * 12.92
    value = float(value) / quantum_max
    value = (value ** (1.0 / 2.4)) * 1.055 - 0.055
    return value * quantum_max


def srgb_to_rgb(img: np.ndarray, max_value: int = 255):
    """sRGB画像->RGB画像に変換する関数

    Args:
        img (np.ndarray):
            sRGB画像
        max_value (int optional):
            画素値の最大値

    Return:
        dst (np.ndarray):
            RGBに変換された画像
    """

    # 入力画像がRGB画像でない場合
    if img.shape[2] != 4:
        raise ValueError("Channel Error: {}".format(img.shape[2]))

    dst = img.copy()
    w, h, ch = dst.shape

    for i in range(0, w):
        for j in range(0, h):
            for k in range(0, 3):
                dst[i][j][k] = _srgb_to_rgb(dst[i][j][k], quantum_max=max_value)

    return dst


def _srgb_to_rgb(value, quantum_max=1.0):
    value = float(value) / quantum_max
    if value <= 0.04045:
        return value / 12.92
    value = ((value + 0.055) / 1.055) ** 2.4
    return value * quantum_max


def AutoGrayScale(img, clearly: bool = False, calc: str = "cv2") -> np.ndarray:
    """入力画像を自動的にグレースケール画像に変換する関数

    Args:
        img (np.ndarray):
            変換前の画像
        clearly (bool optional):
            ガウシアンフィルタを用いて入力画像のノイズを除去する．
            * True: 適用する
            * False: 適用しない: default
        calc (str, optional):
            グレースケール変換を行うための関数を指定.
            Defaults to "cv2".

    Returns:
        dst (np.ndarray):
            変換後の画像データ(Errorが発生した場合: None)

    """

    dst = img.copy()
    try:
        # ガウスフィルタをかけてノイズを除去する
        if clearly:
            dst = cv2.GaussianBlur(dst, (5, 5), 0)
        # 入力画像がRGBの時
        if len(dst.shape) > 2:
            dst = _GrayScale(dst, calc)
    except Exception as e:
        print("GrayScaleError:", e)
        return None
    else:
        return dst


def _GrayScale(img: np.ndarray, calc: str = "cv2") -> np.ndarray:
    """入力画像をグレースケール画像に変換する関数

    Args:
        img (np.ndarray):
            変換前の画像
        calc (str, optional):
            グレースケールを行うための関数を指定
            "cv2": cv2.cvtColor で変換

    Return:
        dst (np.ndarray):
            グレースケール化後の画像(Errorが発生した場合: None)
    """
    if calc == "cv2":
        dst = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return dst
    else:
        return None


if __name__ == "__main__":
    # from DobotFunction.Camera import WebCam_OnOff, Snapshot, Preview
    # _, cam = WebCam_OnOff(0)
    # _, img = Snapshot(cam)
    # Preview(img, preview="plt")
    import json
    from PIL import Image
    from matplotlib import pyplot as plt

    from src.config.config import cfg

    # テスト画像の保存先
    save_path = cfg.GRAY_IMG_DIR
    clac_type = "cv2"

    # テストデータロード
    json_path = cfg.TEST_DIR + os.sep + "data.json"
    with open(json_path, mode="rt", encoding="utf_8") as f:
        datas = json.load(f)

    # テスト
    for data in datas["org_img"]:
        for key, value in data.items():
            if key == "path":
                # 画像を開いて numpy 配列に変換
                img = np.array(Image.open(value))
                # rgb -> srgb
                img = rgb_to_srgb(img)
                # グレースケール化
                # img = AutoGrayScale(img, calc=clac_type)

                img = Image.fromarray(img)
                img.save(save_path + os.sep + name + ".png")
            elif key == "name":
                name = value.rstrip(".png") + "_gray_" + clac_type

    # plt.imshow(img, cmap="gray")
    # plt.show()


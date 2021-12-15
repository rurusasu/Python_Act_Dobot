import sys
import os
from typing import List, Literal, Tuple, Union

sys.path.append(".")
sys.path.append("../")
sys.path.append("../../")

import cv2
import numpy as np

CalcCOGMode = {
    "image": 0,
    "outline": 1,
}
# 輪郭情報
RetrievalMode = {
    "LIST": cv2.RETR_LIST,
    "EXTERNAL": cv2.RETR_EXTERNAL,
    "CCOMP": cv2.RETR_CCOMP,
    "TREE": cv2.RETR_TREE,
}
# 輪郭の中間点情報
ApproximateMode = {
    "Keep": cv2.CHAIN_APPROX_NONE,
    "Not-Keep": cv2.CHAIN_APPROX_SIMPLE,
}


def CenterOfGravity(
    rgb_img: np.ndarray,
    bin_img: np.ndarray,
    Retrieval: Literal["LIST", "EXTERNAL", "CCOMP", "TREE"] = "TREE",
    Approximate: Literal["Keep", "Not-Keep"] = "Keep",
    min_area=100,
    cal_Method: Literal["image", "outline"] = "image",
    orientation: bool = False,
    drawing_figure: bool = True,
) -> Tuple[Union[List[float], None], np.ndarray]:
    """
    オブジェクトの図心を計算する関数

    Args:
        rgb_img (np.ndarray): 計算された重心位置を重ねて表示するRGB画像
        bin_img (np.ndarray): 重心計算対象の二値画像．
        Retrieval (Literal["LIST", "EXTERNAL", "CCOMP", "TREE"], optional):
            2値画像の画素値が 255 の部分と 0 の部分を分離した際に，その親子関係を保持するか指定．
            Defaults to "TREE".
            * "LIST": 輪郭の親子関係を無視する(親子関係が同等に扱われるので、単なる輪郭として解釈される)．
            * "EXTERNAL": 最外の輪郭を検出する
            * "CCOMP": 2つの階層に分類する(物体の外側の輪郭を階層1、物体内側の穴などの輪郭を階層2として分類).
            * "TREE": 全階層情報を保持する
        Approximate (Literal["Keep", "Not-Keep"], optional):
            輪郭の近似方法．Defaults to "Keep"．
            * "Keep": 中間点も保持する。
            * cv2.CHAIN_APPROX_SIMPLE: 中間点は保持しない。
        min_area (int): 領域が占める面積の閾値を指定
        cal_Method (Literal["image", "outline"] optional):
            重心位置の計算対象．Defaults to "image".
            * "image": 画像から重心を計算
            * "outline": オブジェクトの輪郭から重心を計算
        orientation (bool, optional): オブジェクトの輪郭情報に基づいて姿勢を推定する関数．`cal_Method = 1` の場合のみ適用可能．Default to False.
        drawing_figure (bool optional): 輪郭線と重心位置が示された図を描画する。default to True
    Return:
        G (Union[List[float], None]): G=[x, y, angle], オブジェクトの重心座標と，そのオブジェクトの2D平面での回転角度．
        dst (np.ndarray): 重心位置が描画された二値画像
    """
    # ------------ #
    # 初期値設定 #
    # ------------ #
    angle = None
    # 親子関係の保持設定
    if Retrieval in RetrievalMode:
        Retrieval = RetrievalMode[Retrieval]
    else:
        raise ValueError("The `RetrievalMode` is invalid.")
    if Approximate in ApproximateMode:
        Approximate = ApproximateMode[Approximate]
    else:
        raise ValueError("The `Approximate` is invalid.")

    # 入力が2値画像以外の場合
    if (type(bin_img) is not np.ndarray) or (len(bin_img.shape) != 2):
        raise ValueError("入力画像が不正です！")

    # dst_rgb = rgb_img.copy()
    # dst_bin = bin_img.copy()
    # 画像をもとに重心を求める場合
    if cal_Method == 0:
        M = cv2.moments(bin_img, False)

    # 輪郭から重心を求める場合
    else:
        contours = _ExtractContours(
            bin_img=bin_img,
            RetrievalMode=RetrievalMode,
            ApproximateMode=ApproximateMode,
            min_area=min_area,
        )

        # 等高線の描画（Contour line drawing）
        if drawing_figure:
            bin_img = __drawing_edge(bin_img, contours)

        maxCont = contours[0]
        for c in contours:
            if len(maxCont) < len(c):
                maxCont = c

        M = cv2.moments(maxCont)

        if orientation:
            # オブジェクトの輪郭情報から回転角度を計算する．
            # REF: https://seinzumtode.hatenadiary.jp/entry/20171121/1511241157
            # OpenCV-Doc: http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html
            for i, cnt in enumerate(contours):
                # ellipse = cv2.fitEllipse(cnt)
                # dst_bin = cv2.ellipse(dst_bin, ellipse, (255, 0, 0), 2)
                rect = cv2.minAreaRect(cnt)
                # 角度計算の補正
                # REF: https://stackoverflow.com/questions/15956124/minarearect-angles-unsure-about-the-angle-returned
                _, (w, h), angle = rect
                if w > h:
                    angle += 90

                # 外接矩形4点を求める
                # box = cv2.boxPoints(ellipse)
                box = cv2.boxPoints(rect)

                box = np.int0(box)
                rgb_img = cv2.drawContours(rgb_img, [box], 0, (0, 0, 255), 2)
                # 小数点以下2桁に丸め
                angle = round(angle, 2)
    if int(M["m00"]) == 0:
        return None, rgb_img

    try:
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    except ZeroDivisionError:
        return None, rgb_img

    # 重心位置を円で表示
    # 変数: img, 中心座標, 半径, 色
    cv2.circle(rgb_img, center=(cx, cy), radius=10, color=100, thickness=2)

    if drawing_figure:
        cv2.imshow("Convert", rgb_img)  # 画像を出力

    return [cx, cy, angle], rgb_img


def _ExtractContours(
    bin_img,
    RetrievalMode=cv2.RETR_EXTERNAL,
    ApproximateMode=cv2.CHAIN_APPROX_NONE,
    min_area=100,
) -> np.ndarray:
    """
    画像に含まれるオブジェクトの輪郭(contours)を抽出する関数。
    黒い背景（暗い色）から白い物体（明るい色）の輪郭を検出すると仮定。

    Args:
        bin_img (np.ndarray): 二値画像
        RetrievalMode (optional): 輪郭の階層情報
            * cv2.RETR_LIST: 輪郭の親子関係を無視する(親子関係が同等に扱われるので、単なる輪郭として解釈される)。
            * cv2.RETR_EXTERNAL: 最も外側の輪郭だけを検出するモード
            * cv2.RETR_CCOMP: 2レベルの階層に分類する(物体の外側の輪郭を階層1、物体内側の穴などの輪郭を階層2として分類)。
            * cv2.RETR_TREE: 全階層情報を保持する。
        ApproximateMode (optional): 輪郭の近似方法
            * cv2.CHAIN_APPROX_NONE: 中間点も保持する。
            * cv2.CHAIN_APPROX_SIMPLE: 中間点は保持しない。
        min_area (int): 領域が占める面積の閾値を指定

    Returns:
        approx (list[int]): 近似した輪郭情報
    """
    # 輪郭検出（Detection contours）
    # contours: 輪郭線の画素位置の numpy 配列
    contours, hierarchy = cv2.findContours(bin_img, RetrievalMode, ApproximateMode)
    # 小さい輪郭は誤検出として削除する
    contours = list(filter(lambda x: cv2.contourArea(x) > min_area, contours))
    # 輪郭近似（Contour approximation）
    approx = __approx_contour(contours)

    return approx


def __approx_contour(contours: list):
    """
    輪郭線の直線近似を行う関数

    Arg:
        contours (list[int]): 画像から抽出した輪郭情報
    Return:
        approx (list[int]): 近似した輪郭情報
    """
    approx = []
    for i in range(len(contours)):
        cnt = contours[i]
        epsilon = 0.001 * cv2.arcLength(cnt, True)  # 実際の輪郭と近似輪郭の最大距離を表し、近似の精度を表すパラメータ
        approx.append(cv2.approxPolyDP(cnt, epsilon, True))
    return approx


def __drawing_edge(src: np.ndarray, contours: list) -> np.ndarray:
    """
    入力されたimgに抽出した輪郭線を描く関数

    Args:
        src (np.ndarray): 輪郭線を描く元の画像データ
        contours (list[int]): 画像から抽出した輪郭情報
    Return:
        dst (np.ndarray): 輪郭情報を描画した画像
    """
    dst = cv2.drawContours(src, contours, -1, color=(255, 0, 0), thickness=1)
    return dst


if __name__ == "__main__":
    from PIL import Image
    from matplotlib import pyplot as plt

    from src.config.config import cfg
    from ImageProcessing.GrayScale import AutoGrayScale
    from ImageProcessing.Binarization import GlobalThreshold

    img_path = cfg.TEST_IMG_ORG_DIR + os.sep + "lena.png"
    img = np.array(Image.open(img_path))  # ロード
    img = AutoGrayScale(img, clearly=True)  # RGB -> Gray
    img = GlobalThreshold(img)  # Gray -> Binary
    # ----- #
    # テスト #
    # ----  #
    """
    cvt = [cv2.RETR_LIST, cv2.RETR_EXTERNAL, cv2.RETR_TREE]
    for i, j in enumerate(cvt):
        dst = ExtractContours(img, RetrievalMode=j)

        plt.figure(i)
        plt.imshow(dst, cmap = "gray")
    """
    dst, _ = _ExtractContours(img)
    plt.imshow(dst, cmap="gray")
    plt.show()

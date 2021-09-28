import sys
import os

sys.path.append(".")
sys.path.append("../")
sys.path.append("../../")

import cv2
import numpy as np


def CenterOfGravity(
    bin_img: np.ndarray,
    RetrievalMode=cv2.RETR_EXTERNAL,
    ApproximateMode=cv2.CHAIN_APPROX_NONE,
    min_area=100,
    cal_Method: int=0,
    drawing_figure: bool=True
) -> list:
    """
    オブジェクトの図心を計算する関数

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
        cal_Method (int optional): 重心計算を行う方法を選択する
            * 0: 画像から重心を計算
            * 1: オブジェクトの輪郭から重心を計算
        drawing_figure (bool optional): 輪郭線と重心位置が示された図を描画する。default to True
    Return:
        G(list): G=[x, y], オブジェクトの重心座標
    """

    # 入力が2値画像以外の場合
    if (type(bin_img) is not np.ndarray) or (len(bin_img.shape) != 2):
        raise ValueError('入力画像が不正です！')

    dst = bin_img.copy()
    # 画像をもとに重心を求める場合
    if cal_Method == 0:
        M = cv2.moments(dst, False)

    # 輪郭から重心を求める場合
    else:
        contours = _ExtractContours(
            bin_img=dst,
            RetrievalMode=RetrievalMode,
            ApproximateMode=ApproximateMode,
            min_area=min_area
            )


        # 等高線の描画（Contour line drawing）
        if drawing_figure:
            dst = __drawing_edge(dst, contours)

        maxCont = contours[0]
        for c in contours:
            if len(maxCont) < len(c):
                maxCont = c

        M = cv2.moments(maxCont)
    if int(M["m00"]) == 0:
        return None

    try:
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    except ZeroDivisionError:
        return None

    if drawing_figure:
        cv2.circle(dst, (cx, cy), 4, 100, 2, 4)  # 重心位置を円で表示
        cv2.imshow('Convert', dst)  # 画像として出力

    return [cx, cy]


def _ExtractContours(
    bin_img,
    RetrievalMode=cv2.RETR_EXTERNAL,
    ApproximateMode=cv2.CHAIN_APPROX_NONE,
    min_area=100,
) -> np.ndarray:
    """
    画像に含まれるオブジェクトの輪郭を抽出する関数。
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
    contours, hierarchy = cv2.findContours(bin_img, RetrievalMode, ApproximateMode)
    # 小さい輪郭は誤検出として削除する
    contrours = list(filter(lambda x: cv2.contourArea(x) > 100, contours))
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
        epsilon = 0.001 * cv2.arcLength(cnt, True) # 実際の輪郭と近似輪郭の最大距離を表し、近似の精度を表すパラメータ
        approx.append(cv2.approxPolyDP(cnt, epsilon, True))
    return approx


def __drawing_edge(
    src: np.ndarray,
    contours: list
    ) -> np.ndarray:
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
    img = np.array(Image.open(img_path)) # ロード
    img = AutoGrayScale(img, clearly=True) # RGB -> Gray
    img = GlobalThreshold(img) # Gray -> Binary
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

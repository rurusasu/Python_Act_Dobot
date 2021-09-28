import os
import sys

sys.path.append(".")
sys.path.append("..")
sys.path.append("../../")

import cv2
import numpy as np


def gaussian(src: np.ndarray, kernel: tuple = (3, 3), sigma: tuple = (0, 0)):
    """ガウシアンフィルタによる平滑化処理を行うための関数

    Parameters
    ----------
    src (ndarray):
        入力画像
    kernel (tuple):
        カーネルサイズ (x, y)
        default: (3, 3)
    sigma (tuple):
        ガウス分布の分散 sigma の値
        X軸、Y軸方向に対して別々の値を指定することができる。
        両軸とも 0 の場合はカーネルサイズに合わせて sigma は自動的に計算される。
        default (0, 0)

    Return
    ------
    dst (ndarray):
        出力画像
    """
    new_img = src.copy()
    dst = cv2.GaussianBlur(new_img, kernel, sigmaX=sigma[0], sigmaY=sigma[1])

    return dst


def laplacian(src: np.ndarray, bit=cv2.CV_64F, ksize=3):
    """
    ラプラシアンフィルタによる空間フィルタリングを行う関数

    Parameters
    ----------
    src : OpenCV型
        入力画像
    bit
        出力画像のビット深度
    ksize
        カーネルサイズ

    Returns
    -------
    dst : OpenCV型
        出力画像
    """
    new_img = src.copy()
    dst = cv2.Laplacian(new_img, bit, ksize)

    return dst


def prewitt(src, dx=1, dy=1):
    """
    プレヴィットフィルタによる空間フィルタリングを行う関数

    Parameters
    ----------
    src : OpenCV型
        入力画像
    dx
        x軸方向微分の次数
    dy
        y軸方向微分の次数
    ksize
        カーネルサイズ

    (dx, dy) = (1, 0) : 横方向の輪郭検出

    (dx, dy) = (1, 0) : 縦方向の輪郭検出

    (dx, dy) = (1, 1) : 斜め右上方向の輪郭検出

    Returns
    -------
    dst : OpenCV型
        出力画像
    """
    new_img = src.copy()

    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

    if dx == 1 and dy == 0:
        dst = cv2.filter2D(new_img, -1, kernelx)
    elif dx == 0 and dy == 1:
        dst = cv2.filter2D(new_img, -1, kernely)
    elif dx == 1 and dy == 1:
        dst_x = cv2.filter2D(new_img, -1, kernelx)
        dst_y = cv2.filter2D(new_img, -1, kernely)
        dst = dst_x + dst_y
    else:
        print("dx, dy は 0 もしくは 1 を指定してください。")
        dst = None

    return dst


def sobel(src, bit=cv2.CV_64F, dx: int = 1, dy: int = 1, ksize=3):
    """
    ソーベルフィルタによる空間フィルタリングを行う関数

    Parameters
    ----------
    src (ndarray配列):
        入力画像
    bit
        出力画像のビット深度
    dx (int):
        x軸方向微分の次数 default = 1
    dy (int)
        y軸方向微分の次数 default = 1
    ksize
        カーネルサイズ

    (dx, dy) = (1, 0) : 横方向の輪郭検出

    (dx, dy) = (1, 0) : 縦方向の輪郭検出

    (dx, dy) = (1, 1) : 斜め右上方向の輪郭検出

    Returns
    -------
    dst : OpenCV型
        出力画像
    """
    if (dx == 0) and (dy == 0):
        raise ValueError(
            "The differential direction dx={}, dy={} is incorrect!".format(dx, dy)
        )

    new_img = src.copy()
    dst = cv2.Sobel(new_img, bit, dx, dy, ksize)

    return dst


if __name__ == "__main__":
    import json
    from PIL import Image
    from matplotlib import pyplot as plt

    from src.config.config import cfg

    # テスト画像の保存先
    save_path = cfg.SMOOTHING_IMG_DIR
    func = "gaussinan"

    # テストデータロード
    json_path = cfg.TEST_DIR + os.sep + "data.json"
    with open(json_path, mode="rt", encoding="utf_8") as f:
        datas = json.load(f)

    # テスト
    for data in datas["noise_img"]:
        for key, value in data.items():
            if key == "path":
                # 画像を開いて numpy 配列に変換
                img = np.array(Image.open(value))

                # gaussinan filter test
                if func == "gaussinan":
                    img = gaussian(img, sigma=(1, 1))
                # laplacian filter test
                elif func == "laplacian":
                    img = laplacian(img)
                # prewitt filter test
                elif func == "prewitt":
                    pre_xy = prewitt(img)
                # sobel filter test
                elif func == "sobel":
                    img = sobel(img)

                if type(img) == np.ndarray:

                    # 上段の画像の表示設定
                    plt.subplot(2, 1, 1)  # 引数はそれぞれ，全体の行数，全体の列数，設定対象のIndex
                    plt.imshow(img)
                    plt.axis("off")

                    # 下段のヒストグラムの設定
                    plt.subplot(2, 1, 2)
                    if len(img.shape) == 2:
                        color = ["k"]
                    elif len(img.shape) == 3:
                        color = ["r", "g", "b"]
                    for (i, col) in enumerate(color):
                        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
                        hist = np.sqrt(hist)
                        plt.plot(hist, color=col)
                    plt.show()

                    img = Image.fromarray(img)
                    img.save(save_path + os.sep + name + ".png")
            elif key == "name":
                name = value.rstrip(".png") + "_" + func

    """
    # オリジナル画像を表示
    plt.subplot(3, 3, 1), plt.imshow(img, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    # Laplacian filtering を使用した画像を表示
    plt.subplot(3, 3, 2), plt.imshow(lap3, cmap='gray')
    plt.title('Laplacian, ksize=3'), plt.xticks([]), plt.yticks([])
    plt.subplot(3, 3, 3), plt.imshow(lap5, cmap='gray')
    plt.title('Laplacian, ksize=5'), plt.xticks([]), plt.yticks([])
    # Prewitt filtering を使用した画像を表示
    plt.subplot(3, 3, 4), plt.imshow(pre_xy, cmap='gray')
    plt.title('Prewitt dx=1, dy=1'), plt.xticks([]), plt.yticks([])
    plt.subplot(3, 3, 5), plt.imshow(pre_x, cmap='gray')
    plt.title('Prewitt dx=1, dy=0'), plt.xticks([]), plt.yticks([])
    plt.subplot(3, 3, 6), plt.imshow(pre_y, cmap='gray')
    plt.title('Prewitt dx=0, dy=1'), plt.xticks([]), plt.yticks([])
    # Sobel filtering を使用した画像を表示
    plt.subplot(3, 3, 7), plt.imshow(sob_xy, cmap='gray')
    plt.title('Sobel dx=1, dy=1'), plt.xticks([]), plt.yticks([])
    plt.subplot(3, 3, 8), plt.imshow(sob_x, cmap='gray')
    plt.title('Sobel dx=1, dy=0'), plt.xticks([]), plt.yticks([])
    plt.subplot(3, 3, 9), plt.imshow(sob_y, cmap='gray')
    plt.title('Sobel dx=0, dy=1'), plt.xticks([]), plt.yticks([])
    """

    # オリジナル画像
    # plt.subplot(1, 3, 1), plt.imshow(img)
    # plt.title("Original"), plt.xticks([]), plt.yticks([])
    # Gaussian filtering を使用した画像を表示
    # plt.subplot(1, 3, 2), plt.imshow(gau)
    # plt.title("Gaussian"), plt.xticks([]), plt.yticks([])

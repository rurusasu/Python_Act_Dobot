import os
import sys

sys.path.append(".")
sys.path.append("..")
sys.path.append("../../")

import numpy as np
import cv2


def canny(src, thresh1=100, thresh2=200):
    """
    Cannyアルゴリズムによって、画像から輪郭を取り出すアルゴリズム

    Parameters
    ----------
    src : OpenCV型
        入力画像
    thresh1
        最小閾値(Hysteresis Thresholding処理で使用)
    thresh2
        最大閾値(Hysteresis Thresholding処理で使用)

    Returns
    -------
    dst : OpenCV型
        出力画像

    """
    new_img = src.copy()
    dst = cv2.Canny(new_img, thresh1, thresh2)

    return dst


def LoG(src, ksize=(3, 3), sigmaX=1.3, l_ksize=3):
    """
    ガウシアンフィルタで画像を平滑化してノイズを除去した後、ラプラシアンフィルタで輪郭を取り出す

    Parameters
    ----------
    src : OpenCV型
        入力画像
    ksize : tuple
        ガウシアンフィルタのカーネルサイズ
    sigmaX
        ガウス分布のσ
    l_ksize
        ラプラシアンフィルタのカーネルサイズ

    Returns
    -------
    dst : OpenCV型
        出力画像
    """
    new_img = src.copy()
    dst = cv2.GaussianBlur(new_img, ksize, sigmaX)
    dst = laplacian(dst, ksize=l_ksize)

    return dst


if __name__ == "__main__":
    import json
    from PIL import Image
    from matplotlib import pyplot as plt

    from src.config.config import cfg

    # テスト画像の保存先
    save_path = cfg.EDGE_IMG_DIR
    func = "canny"

    # テストデータロード
    json_path = cfg.TEST_DIR + os.sep + "data.json"
    with open(json_path, mode="rt", encoding="utf_8") as f:
        datas = json.load(f)

    # テスト
    for data in datas["gray_img"]:
        for key, value in data.items():
            if key == "path":
                # 画像を開いて numpy 配列に変換
                img = np.array(Image.open(value))
                if func == "canny":  # canny edge detector test
                    img = canny(img)
                elif func == "log":  # LoG filter test
                    img = LoG(img)
                if type(img) == np.ndarray:
                    img = Image.fromarray(img)
                    img.save(save_path + os.sep + name + ".png")
            elif key == "name":
                name = value.rstrip(".png") + func

    """
    # 画像をarrayに変換
    im_list = np.asarray(img)
    # 貼り付け
    plt.imshow(im_list, cmap="gray")
    """

    """
    # オリジナル画像を表示
    plt.subplot(1, 3, 1), plt.imshow(img, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    # Cannyエッジ検出器を使用した画像を表示
    plt.subplot(1, 3, 2), plt.imshow(can, cmap='gray')
    plt.title('Canny'), plt.xticks([]), plt.yticks([])
    # LoG filtering を使用した画像を表示
    plt.subplot(1, 3, 3), plt.imshow(log, cmap='gray')
    plt.title('LoG'), plt.xticks([]), plt.yticks([])
    """

    # 表示
    plt.show()

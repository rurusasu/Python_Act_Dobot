import sys, os

sys.path.append(".")
sys.path.append("..")
sys.path.append("../../")

import cv2
import numpy as np

def Contrast_cvt(src, cvt_type):
    """
    濃度の変換方法を選択する関数

    Parameter
    ---------
    src : OpenCV型
        変換前の画像
    cvt_type : string
        濃度の変換方法
        ・線形濃度変換
        ・非線形濃度変換
        ・ヒストグラム平坦化

    Return
    ------
    dst : OpenCV型
        変換後の画像
    """
    a = 0.7
    gamma = 0.5

    new_img = src.copy()

    if cvt_type == '線形濃度変換':  # 線形濃度変換を行う
        new_img = _LUT_curve(__curve_1, a, new_img)
    elif cvt_type == '非線形濃度変換':  # ガンマ補正を行う
        new_img = _LUT_curve(__curve_5, gamma, new_img)
    elif cvt_type == 'ヒストグラム平坦化':  # ヒストグラム平坦化を行う
        if len(new_img.shape) == 2:
        #if color_type == 'glay':  # グレー画像について変換
            new_img = _GlayHist(new_img)
        elif len(new_img.shape) == 3:
        #elif color_type == 'RGB':  # rgb画像について変換
            new_img = _RGBHist(new_img)

    return new_img

def _LUT_curve(f, a: float, rgb_img: np.ndarray) -> np.ndarray:
    """
    Look Up Tableを LUT[input][0] = output という256行の配列として作る。
    例: LUT[0][0] = 0, LUT[127][0] = 160, LUT[255][0] = 255

    Args:
        f (function): 濃度変換に使用する曲線の関数
        a (float): 曲線の計算に使用される変数
        rgb_img (np.ndarray): 変換前の画像

    Return:
        dst (np.ndarray): 変換後の画像

    """
    LUT = np.arange(256, dtype='uint8').reshape(-1, 1)
    LUT = np.array([f(a, x).astype('uint8') for x in LUT])

    dst = cv2.LUT(rgb_img, LUT)

    return dst


def __curve_1(a, x):
    y = a * x
    return y

def __curve_2(a, x):
    y = x + a
    return y

def __curve_3(a, x):
    y = a * (x - 127.0) + 127.0
    return y

def __curve_4(a, x):
    zmin, zmax = 20.0, 220.0
    y = a * (x - zmin) / (zmax - zmin)
    return y

def __curve_5(gamma, x):
    y = 255*(x/255)**(1/gamma)
    return y


def _GlayHist(glay_img: np.ndarray, clip_limit: int = 3, grid: tuple = (8, 8), thresh: int = 225) -> np.ndarray:
    """
    グレー画像に対して適応的ヒストグラム平坦化(Clahe)を行う関数

    Args:
        glay_img (np.ndarray):
            変換前の画像
        clip_limit (int):
            コントラストの強調制限
            範囲(0~255)
            上限値を超える画素はその他のビンに均等に分配され，その後にヒストグラム平坦化を適用します.
        grid (tuple):
            タイルサイズ
            適応的ヒストグラム平坦化では, 画像をタイルサイズの小領域に分割し, 領域ごとにヒストグラム平坦化を行う.
        thresh (int):
            白色の領域を調整する閾値

    Return:
        dst (np.ndarray):
            変換後の画像
    """
    dst = glay_img.copy()
    clahe = cv2.createCLAHE(cliplimit=clip_limit, tileGridSize=grid)
    dst = clahe.apply(dst)
    #dst = new_img.copy()
    dst[dst > thresh] = 255
    return dst


def _RGBHist(rgb_img: np.ndarray, clip_limit: int = 3, grid: tuple=(8, 8), thresh: int=225):
    """
    rgb画像に対して適応的ヒストグラム平坦化(Clahe)を行う関数

    Args:
    rgb_img (np.ndarray):
        変換前の画像
    clip_limit (int):
        コントラストの強調制限
        範囲(0~255)
        上限値を超える画素はその他のビンに均等に分配され，その後にヒストグラム平坦化を適用します.
    grid (tuple):
        タイルサイズ
        適応的ヒストグラム平坦化では, 画像をタイルサイズの小領域に分割し, 領域ごとにヒストグラム平坦化を行う.
    thresh (int):
        白色の領域を調整する閾値

    Return:
        dst (np.ndarray):
            変換後の画像
    """
    dst = rgb_img.copy()
    clahe = cv2.createCLAHE(cliplimit=clip_limit, tileGridSize=grid)
    r, g, b = cv2.split(dst)

    # r, g, bそれぞれで変換を行う
    dst_R = clahe.apply(r)
    dst_G = clahe.apply(g)
    dst_B = clahe.apply(b)

    th_R, th_G, th_B = dst_R.copy(), dst_G.copy(), dst_B.copy
    th_R[dst_R > thresh] = 255
    th_G[dst_G > thresh] = 255
    th_B[dst_B > thresh] = 255

    dst = cv2.merge((th_R, th_G, th_B))
    dst = dst[:, :, ::-1]

    return dst


if __name__ == "__main__":
    import json
    from PIL import Image
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    from src.config.config import cfg

    # テスト画像の保存先
    save_path = cfg.CONTRAST_IMG_DIR
    cvt_type = "線形濃度変換"

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
                # コントラスト変更
                img = Contrast_cvt(img, cvt_type=cvt_type)

                img = Image.fromarray(img)
                img.save(save_path + os.sep + name + ".png")
            elif key == "name":
                name = value.rstrip(".png") + "_" + cvt_type

    a=0.7
    N = 1000
    x = np.linspace(-5, 5, N)

    y1 = __curve_1(a, x)
    y2 = __curve_2(a, x)
    y3 = __curve_3(a, x)
    y4 = __curve_4(a, x)
    y5 = __curve_5(a, x)

    fig = plt.figure()
    ax1 = fig.add_subplot(231, title="curve_1", xlabel="Number", ylabel="result")
    ax1.plot(x, y1)
    ax2 = fig.add_subplot(232, title="curve_2", xlabel="Number", ylabel="result")
    ax2.plot(x, y2)
    ax3 = fig.add_subplot(233, title="curve_3", xlabel="Number", ylabel="result")
    ax3.plot(x, y3)
    ax4 = fig.add_subplot(234, title="curve_4", xlabel="Number", ylabel="result")
    ax4.plot(x, y4)
    ax5 = fig.add_subplot(235, title="curve_5", xlabel="Number", ylabel="result")
    ax5.plot(x, y5)

    ax6 = fig.add_subplot(236, title="curve", xlabel="Number", ylabel="result")
    ax6.plot(x, y1, label="curve1")
    ax6.plot(x, y2, label="curve2")
    ax6.plot(x, y3, label="curve3")
    ax6.plot(x, y4, label="curve4")
    ax6.plot(x, y5, label="curve5")
    plt.legend()

    plt.show()

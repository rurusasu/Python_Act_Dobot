import os
import sys

sys.path.append('.')
sys.path.append('..')

import cv2
import numpy as np

from kernels import gaussian

def gausian_pyr(src: np.ndarray, kernel: tuple=(3, 3), sigma: tuple=(1, 1), level: int=1):
    """ガウシアンフィルタを用いて画像ピラミッドを作成する関数。
    level は、ピラミッドの高さを表す。すなわち、画像に対してガウシアンフィルタを適用する回数を表す。

    Parameters
    ----------
    src (ndarray):
        入力画像
    kernel (tuple):
        カーネルサイズ (x, y)
        default (3, 3)
    sigma (tuple):
        ガウス分布の分散 sigma の値
        X軸、Y軸方向に対して別々の値を指定することができる。
        両軸とも 1 以上の値にする必要がある(1 の場合は level を上げても画素値は最初にフィルタリングした値から変化しない)。
        default (1, 1)
    lebel (int):
        作成するピラミッドのレベル
        default 1

    Return
    ------
    dst (ndarray)
        出力画像
    """
    new_img = src.copy()

    if (sigma[0] == 0) or (sigma[1] == 0):
        raise ValueError('Invaild Parameter')
    else:
        sigma = np.array(sigma)

    for lvl in range(level):
        if lvl == 0:
            dst = gaussian(new_img, kernel, sigma)
            dsts = dst
        elif lvl > 0:
            sigma = sigma*(lvl+1)
            dst = gaussian(dst, kernel, sigma)
            dsts = np.stack([dsts, dst])
        else:
            raise ValueError('Invaild Parameter')

    return dsts


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from PIL import Image

    from config.config import cfg

    idx = 8  # load image number
    pyramid_level=2 # 画像ピラミッドの高さ

    # ディレクトリ設定
    # linemod
    #base_dir = os.path.join(cfg.LINEMOD_DIR, object_name, 'data')
    #object_name = "ape"
    #img_path = os.path.join(base_dir, 'color{}.jpg'.format(idx))

    # Test_img
    img_path = os.path.join(cfg.TEST_IMAGE_DIR, 'image{}.jpg'.format(idx))

    # image read
    img = Image.open(img_path)
    # img = img.convert('L')  # gray scale
    img = np.array(img)
    x, y = img.shape[0], img.shape[1]

    # gaussian pyramid test
    #g_p = gausian_pyr(img, level=pyramid_level)
    g_p = gausian_pyr(img, sigma=(10, 10), level=pyramid_level)


        # オリジナル画像
    plt.subplot(1, 3, 1), plt.imshow(img)
    plt.title('Original x:{}, y:{}'.format(x, y))
    plt.xticks([]), plt.yticks([])
    # Gaussian filtering を使用した画像を表示
    plt.subplot(1, 3, 2), plt.imshow(g_p[0])
    plt.title('x:{}, y:{}, sigma'.format(g_p[0].shape[0], g_p[0].shape[1]))
    plt.xticks([]), plt.yticks([])

    if pyramid_level == 2:
        plt.subplot(1, 3, 3), plt.imshow(g_p[1])
    plt.title('x:{}, y:{}, sigma'.format(g_p[1].shape[0], g_p[1].shape[1]))
    plt.xticks([]), plt.yticks([])

    # 表示
    plt.show()
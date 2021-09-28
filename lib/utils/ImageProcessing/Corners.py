import os
import sys

sys.path.append('.')
sys.path.append('..')

import cv2
import numpy as np

def Harris(img: np.ndarray, blockSize: int =2, ksize: int =3, k: float =0.04):
    """Harrisコーナー検出を行う関数

    Parames
    -------
    img (ndarrya):
        検出対象画像
    blockSize (int):
        コーナー検出の際に考慮する隣接領域のサイズ
        default: 2
    ksize (int):
        Sobelフィルタのカーネルサイズ
        default: 3
    k (float):
        フリーパラメータ
        default 0.04
    """
    gray = img.copy()

    # 画像がグレースケールでない場合
    if len(gray.shape) != 2:
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)

    gray= np.float32(gray)
    dst = cv2.cornerHarris(gray, blockSize, ksize, k)
    # コーナーを強調表示(膨張)する。この処理は重要ではない。
    dst = cv2.dilate(dst, None)

    return dst


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from PIL import Image

    from config.config import cfg

    idx = 7 # load image number

    # ディレクトリ設定
    ## linemod
    #base_dir = os.path.join(cfg.LINEMOD_DIR, object_name, 'data')
    #object_name = "ape"
    #img_path = os.path.join(base_dir, 'color{}.jpg'.format(idx))

    ## Test_img
    img_path = os.path.join(cfg.TEST_IMAGE_DIR, 'image{}.jpg'.format(idx))

    # image read
    img = Image.open(img_path)
    #img = img.convert('L')  # gray scale
    img = np.array(img)
    x, y = img.shape[0], img.shape[1]

    # Harris corner test
    dst = Harris(img)

    plt.subplot(1, 2, 1), plt.imshow(dst, cmap='gray')
    plt.title('Detected corners'), plt.xticks([]), plt.yticks([])

    plt.subplot(1, 2, 2)
    if len(img.shape) != 2:
        img[dst>0.01*dst.max()] = [255, 0, 255]
        plt.imshow(img)
    else:
        img[dst>0.01*dst.max()] = 255
        plt.imshow(img, cmap='gray')

    plt.title('Retult width:{}, Height:{}'.format(x, y)), plt.xticks([]), plt.yticks([])

    plt.show()


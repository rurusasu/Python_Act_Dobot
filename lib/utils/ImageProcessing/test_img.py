import sys, os

sys.path.append(".")
sys.path.append("..")
sys.path.append("../../")

import numpy as np
import random
from PIL import Image

from src.config.config import cfg


def Gradation(size=256):
    img = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(i)
        img.append(row)

    img = np.array(img)
    img = Image.fromarray(img, mode="L")

    return img


def Noisy(
    WIDTH=640, HEIGHT=480, BLOCK_SIZE: int = 1, stride: int = 1, img_type: str = "RGB"
):
    """RGBノイズ画像作成処理
    """

    # 画像サイズの端数を切り捨てる
    # w = WIDTH
    # h = HEIGHT
    w = int(WIDTH / (BLOCK_SIZE + stride)) * (BLOCK_SIZE + stride)
    h = int(HEIGHT / (BLOCK_SIZE + stride)) * (BLOCK_SIZE + stride)
    # 画像インスタンスを作成

    if img_type == "RGB":
        img = Image.new(img_type, (w, h), (0, 0, 0))

        # 乱数をピクセル値とする
        for i in range(0, w, (BLOCK_SIZE + stride)):
            for j in range(0, h, (BLOCK_SIZE + stride)):
                r = int(random.random() * 256)
                g = int(random.random() * 256)
                b = int(random.random() * 256)

                for k in range(BLOCK_SIZE):
                    for l in range(BLOCK_SIZE):
                        img.putpixel((i + l, j + k), (r, g, b))

    return img


def SaltAndPepper(img: np.ndarray, sample_num: int = 1000):
    """画像に「ごま塩ノイズ」をかける関数

    Args:
        img (np.ndarray):
            ノイズをかける元画像
        sample_num (int optional):
            ノイズをかける座標のサンプル数
            default: 1000

    Return:
        dst (np.ndarray):
            ノイズがかかった画像(Errorが発生した場合: None)
    """
    dst = img.copy()
    try:
        w, h, ch = dst.shape
        # 白色のノイズをかける
        pts_x = np.random.randint(0, h - 1, sample_num)
        pts_y = np.random.randint(0, w - 1, sample_num)
        dst[(pts_y, pts_x)] = (255, 255, 255)
        # 黒色のノイズをかける
        pts_x = np.random.randint(0, h - 1, 1000)
        pts_y = np.random.randint(0, w - 1, 1000)
        dst[(pts_y, pts_x)] = (0, 0, 0)
    except Exception as e:
        print("SaltAndPepper Function Error:", e)
        return None
    else:
        return dst


if __name__ == "__main__":
    import json
    from PIL import Image
    from matplotlib import pyplot as plt

    from src.config.config import cfg

    # テスト画像の保存先
    save_path = cfg.NOISE_IMG_DIR
    sample_num = 5000

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
                # ノイズを付加
                img = SaltAndPepper(img)
                if type(img) == np.ndarray:
                    img = Image.fromarray(img)
                    img.save(save_path + os.sep + name + ".png")
            elif key == "name":
                name = value.rstrip(".png") + "_salt_and_pepper_" + str(sample_num)

    """
    img_path = cfg.TEST_IMG_ORG_DIR + os.sep + "lena.png"

    img = Image.open(img_path)
    img = np.array(img)
    # noise = Noisy(BLOCK_SIZE=30, stride=2)
    # noise = np.array(noise)
    # plt.imshow(noise)
    save_path = cfg.TEST_IMG_ORG_DIR + os.sep + "SaltAndPepper.png"
    img = SaltAndPepper(img)
    img = Image.fromarray(img)
    img.save(save_path)
    # img = np.array(img) + np.array(noise)
    plt.imshow(img)
    plt.show()
    """

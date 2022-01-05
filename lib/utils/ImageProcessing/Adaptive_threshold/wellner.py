import numpy as np

def Wellner_threshold(src: np.ndarray) -> np.ndarray:
    if type(src) != np.ndarray:
        raise("An undefined value was assigned to `binary_type`.")
    new_img = src.copy()

    if len(src.shape) == 2:
        pass
    elif len(src.shape) == 3:
        # グレースケール化
        new_img = new_img.convert('L')
    else:
        raise("Invalid input size for src.")

    (h, w) = new_img.shape
    # 画素値を1行に整列
    new_img = new_img.reshape((-1, 1))

    # もし画像の幅が8で割り切れない場合
    if w % 8 != 0:
        raise("`w` is not divisible by `8`.")
    else: s = int(w / 8)

    t = 15
    output = np.empty(0)
    #移動平均(Moving average)を計算するためのlistを作成
    MA_list = np.zeros((s))

    for i, v in enumerate(new_img):
        # list内の移動平均を計算する
        MA = MA_list.sum() / s
        MA_list_2 = MA_list[:-1].copy()
        MA_list = np.insert(MA_list_2, 0, v)
        #---------------------
        # 二値化処理する
        #---------------------
        if v < MA * ((100-t) / 100):
            v = 255
        else:
            v = 0

        v = np.array(v)
        output = np.r_[output, v]
    # print('処理が終了しました。')
    output = output.reshape((h, w)).astype(np.uint8)
    return output

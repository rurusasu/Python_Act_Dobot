from typing import Union, Tuple

import cv2
import numpy as np

def Bradley_threshold(src: np.ndarray, kernel_size: Union[int, Tuple[int, int]]=5, T: int=0.30):
    if kernel_size == None:
        raise("It is invalid to assign None to `kernel_size`.")
    if isinstance(kernel_size, int):
        # int 型の場合の処理
        k = int(kernel_size)/2
    elif isinstance(kernel_size, tuple):
        # tuple 型の場合の処理
        k = int(max(kernel_size/2))
    else:
        raise("An unknown type was assigned to `kernel_size`.")

    if len(src.shape) == 3:
        new_img = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    else:
        new_img = src.copy()
    res = np.zeros_like(new_img)

    # 積分画像作成
    int_img = cv2.integral(new_img)

    (h, w) = new_img.shape
    s2 = k
    for col in range(w):
        for row in range(h):
            y0 = int(max(row-s2, 0))
            y1 = int(min(row+s2, h-1))
            x0 = int(max(col-s2, 0))
            x1 = int(min(col+s2, w-1))
            count = (y1-y0)*(x1-x0)
            sum_ = -1
            if count == 0:
                if x0 == x1 and y0 == y1:
                    sum_ = int_img[y0, x0]
                if x1 == x0 and y0 != y1:
                    sum_ = int_img[y1, x1] - int_img[y0, x1]
                if y1 == y0 and x1 != x0:
                    sum_ = int_img[y1, x1] - int_img[y1, x0]
            else:
                sum_ = int(int_img[y1, x1]) - int(int_img[y0, x1]) - int(int_img[y1, x0]) + int(int_img[y0, x0])
                if sum_ < 0: sum_ = -1 * sum_

            # mat[row,col] = sum_/count
            mean = sum_/count

            if new_img[row, col] < mean * (100-T)/100:
            # if new_img[row, col] < mean * T:
                res[row, col] = 0
            else:
                res[row, col] = 255

    return res


def get_int_img(src: np.ndarray) -> np.ndarray:
    """積分画像を作成するための関数

    Args:
        src (np.ndarray): 入力画像

    Returns:
        int_img[np.ndarray]: 積分画像
    """
    h, w = src.shape
    #integral img
    int_img = np.zeros_like(src, dtype=np.uint32)
    for col in range(w):
        for row in range(h):
            int_img[row,col] = src[0:row+1,0:col+1].sum()
    return int_img

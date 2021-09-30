import sys
import time
from typing import Union, Literal

sys.path.append(".")
sys.path.append("..")

import cv2
import numpy as np
import matplotlib.pyplot as plt

from lib.utils.ImageProcessing.Binarization import (
    GlobalThreshold,
    AdaptiveThreshold,
    TwoThreshold,
)
from lib.utils.ImageProcessing.Contrast import Contrast_cvt
from lib.utils.ImageProcessing.CenterOfGravity import CenterOfGravity
from lib.utils.ImageProcessing.GrayScale import AutoGrayScale


def DeviceNameToNum(device_name: str) -> int:
    """
    接続するWebCameraを選択する関数

    Parameter
    ---------
    device_name : int
        使用したいデバイス名を指定

    Return
    ------
    device_num : int
        名前が一致したデバイスに割り当てられた番号を返す
    """

    if device_name == "TOSHIBA_Web_Camera-HD":
        device_num = 0
    elif device_name == "Logicool_HD_Webcam_C270":
        device_num = 1
    else:
        device_num = None

    return device_num


def WebCam_OnOff(device_num: int, cam: Union[cv2.VideoCapture, None] = None):
    """
    WebCameraを読み込む関数

    Args:
        device_num (int): カメラデバイスを番号で指定
            0:PC内臓カメラ
            1:外部カメラ
        cam (Union[cv2.VideoCapture, None], optional): 接続しているカメラ情報. Defaults to None.

    Returns:
        response(int): 動作終了を表すフラグ
            0: connect
            1: release
            2: NotFound
        capture(cv2.VideoCapture): 接続したデバイス情報を返す
            cv2.VideoCapture: connect
            None: release or NotFound
    """
    if cam is None:  # カメラが接続されていないとき
        cam = cv2.VideoCapture(device_num, cv2.CAP_DSHOW)
        # カメラに接続できなかった場合
        if not cam.isOpened():
            return 2, None
        # 接続できた場合
        else:
            return 0, cam

    else:  # カメラに接続されていたとき
        cam.release()
        cv2.destroyAllWindows()
        return 1, None


def Snapshot(cam: cv2.VideoCapture) -> np.ndarray:
    """WebCameraでスナップショットを撮影する関数

    Arg:
        cam(cv2.VideoCapture): 接続しているカメラ情報

    Return:
        response(int):
            3: 撮影できました。
            4: 撮影できませんでした。
        img(np.ndarray): 撮影した画像
            np.ndarray: 撮影成功
            None: 撮影失敗
    """
    ret, img = cam.read()  # 静止画像をGET
    # 静止画が撮影できた場合
    if ret:
        dst = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB
        # dst = np.array(dst)
        # dst = img
        return 3, dst
    # 撮影できなかった場合
    else:
        return 4, None


def scale_box(src: np.ndarray, width: int, height: int):
    """
    アスペクト比を固定して、指定した大きさに収まるようリサイズする。

    Args:
        src (np.ndarray):
            入力画像
        width (int):
            サイズ変更後の画像幅
        height (int):
            サイズ変更後の画像高さ

    Return:
        dst (np.ndarray):
            サイズ変更後の画像
    """
    scale = max(width / src.shape[1], height / src.shape[0])
    return cv2.resize(src, dsize=None, fx=scale, fy=scale)


def Preview(img: np.ndarray = None, window_name: str = "frame", preview: str = "cv2"):
    """
    webカメラの画像を表示する関数

    Parameters
    ----------
    img : ndarray型
        画像のピクセル値配列
        default : None
    window_name : str
        画像を表示する時のウインドウ名
        default : "frame"
    preview : str
        画像をウインドウ上に表示するときに使用するパッケージ名
        OpenCV の imshow を使用する場合 : "cv2" (default)
        Matplotlib の plt.show を使用する場合: "plt"

    Returns
    -------
    response : int
        画像表示の可否を返す
        1: 表示できた。
        -1: 表示できない。
    """
    # 画像が入力されている場合
    if type(img) is np.ndarray:
        # 画像を OpenCV でウインドウ上に表示する
        if preview == "cv2":
            cv2.imshow(window_name, img)
            return 1
        # 画像を Matplotlib で ウインドウ上に表示する
        elif preview == "plt":
            # グレースケール画像の場合
            if len(img.shape) == 2:
                plt.imshow(img, cmap="gray")
            # RGB画像の場合
            elif len(img.shape) == 3:
                plt.imshow(img)
            plt.show()
            return 1
        # 表示に使うパッケージの選択が不適切な場合
        else:
            return -1
    # 画像が入力されていない場合
    else:
        return -1

def Color_cvt(src: np.ndarray, color_type: str):
    """画像の色空間を変換する関数
        変換には OpenCV の関数を使用する。

        Args:
            src (np.ndarray): 変換前の画像
            color_type (str): 変更後の色空間
            * Gray: グレースケール
            * HSV: HDV空間

        Return:
            det (np.ndarray): 変換後の画像
    """
    dst = src.copy()
    try:
        if color_type == 'Gray':
            #dst = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
            dst = AutoGrayScale(dst, clearly=True)
        elif color_type == 'HSV':
            dst = cv2.cvtColor(dst, cv2.COLOR_RGB2HSV)
        else:
            raise ValueError("選択された変換は存在しません。")
    except Exception as e:
        print(e)
    else:
        return dst


def SnapshotCvt(
    cam: cv2.VideoCapture,
    Color_Space: Literal["RGB", "Gray"]="RGB",
    Color_Density: Literal["なし", "線形濃度変換", "ヒストグラム平坦化"]="なし",
    Binarization: Literal["なし", "Global", "Otsu", "Adaptive", "Two"]="なし",
    LowerThreshold: int = 10,
    UpperThreshold: int = 150,
    AdaptiveThreshold_type: Literal["Mean", "Gaussian", "Wellner"]="Mean",
    AdaptiveThreshold_BlockSize: int=11,
    AdaptiveThreshold_Constant: int=2,
    color: int=4):
    dst_org = dst_bin = None
    response, img = Snapshot(cam)
    if response != 3:
        return 4 # WebCam_NotGetImage

    dst_org = img.copy()

    # ---------------------------
    # 撮影した画像を変換する。
    # ---------------------------
    # 色空間変換
    if Color_Space != "RGB":
        img = Color_cvt(img, Color_Space)
    # 濃度変換
    if Color_Density != "なし":
        img = Contrast_cvt(img, Color_Density)
    # 二値化処理
    if Binarization != "なし":  # 二値化処理
        if Binarization == "Global":  # 大域的二値化処理
            img = GlobalThreshold(img, threshold=LowerThreshold)
        elif Binarization == "Otsu":  # 大津の二値化処理
            img = GlobalThreshold(img, Type="Otsu")
        elif Binarization == "Adaptive":
            img = AdaptiveThreshold(
                img=img,
                method=str(AdaptiveThreshold_type),
                block_size=AdaptiveThreshold_BlockSize,
                C=AdaptiveThreshold_Constant,
            )
        elif Binarization == "Two":  # 2つの閾値を用いた二値化処理
            # ピックアップする色を番号に変換

            img = TwoThreshold(
                img=img,
                LowerThreshold=LowerThreshold,
                UpperThreshold=UpperThreshold,
                PickupColor=color,
            )

    return dst_org, img


def Contours(
    img: np.ndarray,
    CalcCOG: Literal["画像から重心を計算", "輪郭から重心を計算"],
    Retrieval: Literal["親子関係を無視する", "最外の輪郭を検出する", "2つの階層に分類する", "全階層情報を保持する"],
    Approximate: Literal["中間点を保持する", "中間点を保持しない"],
    drawing_figure: bool = True
    ):
        """スナップショットの撮影からオブジェクトの重心位置計算までの一連の画像処理を行う関数。

        Args:
            img(np.ndarray): 接続しているカメラ情報
            CalcCOG(Literal["画像から重心を計算", "輪郭から重心を計算"]): 重心位置の計算対象を指定．
            Retrieval(Literal["親子関係を無視する", "最外の輪郭を検出する", "2つの階層に分類する", "全階層情報を保持する"]): 2値画像の画素値が 255 の部分と 0 の部分を分離した際に，その親子関係を保持するか指定．
            Approximate(Literal["中間点を保持する", "中間点を保持しない"]): 輪郭の中間点を保持するか指定．
            drawing_figure(bool): 図を描画する．Default to True.
        Returns:
            COG(list): COG=[x, y], オブジェクトの重心位置
        """
        COG = []
        CalcCOGMode = {
            "画像から重心を計算": 0,
            "輪郭から重心を計算": 1,
        }
        # 輪郭情報
        RetrievalMode = {
            "親子関係を無視する": cv2.RETR_LIST,
            "最外の輪郭を検出する": cv2.RETR_EXTERNAL,
            "2つの階層に分類する": cv2.RETR_CCOMP,
            "全階層情報を保持する": cv2.RETR_TREE,
        }
        # 輪郭の中間点情報
        ApproximateMode = {
            "中間点を保持する": cv2.CHAIN_APPROX_NONE,
            "中間点を保持しない": cv2.CHAIN_APPROX_SIMPLE,
        }


        if type(img) != np.ndarray:
            raise TypeError("入力はnumpy配列を使用してください．")
        elif img.max == 0:
            raise ValueError("画像にオブジェクトが映っていません．")
        try:
            COG, img = CenterOfGravity(
                bin_img=img,
                RetrievalMode=RetrievalMode[Retrieval],
                ApproximateMode=ApproximateMode[Approximate],
                min_area=100,
                cal_Method=CalcCOGMode[CalcCOG],
                drawing_figure=drawing_figure
            )
        except Exception as e:
            print("Gravity center position calculation error")
        finally:
            return COG, img



if __name__ == "__main__":
    response, cam = WebCam_OnOff(device_num=0)
    if response == 1:
        response, img = Snapshot(cam=cam)
        if response == 1:
            Preview(img, preview="plt")

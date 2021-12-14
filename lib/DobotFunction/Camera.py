import sys
from typing import Tuple, Union, List, Literal

sys.path.append(".")
sys.path.append("..")
sys.path.append("../../")

import cv2
import numpy as np
import matplotlib.pyplot as plt
import threading

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
        # バッファサイズを小さくすることによる高速化
        # REF: https://qiita.com/iwatake2222/items/b8c442a9ec0406883950
        cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
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


# bufferless VideoCapture
class VideoCaptureWrapper:
    def __init__(self, device_num: int, cam: Union[cv2.VideoCapture, None] = None):
        """
        WebCameraを読み込むクラス
        参考: [opencvのキャプチャデバイス（カメラ）から最新のフレームを取得する方法](https://stackoverflow.com/questions/43665208/how-to-get-the-latest-frame-from-capture-device-camera-in-opencv)

        Args:
            device_num (int): カメラデバイスを番号で指定
                0:PC内臓カメラ
                1:外部カメラ
            cam (Union[cv2.VideoCapture, None], optional): 接続しているカメラ情報. Defaults to None.
        """
        if cam is None:  # カメラが接続されていないとき
            self.cam = cv2.VideoCapture(device_num)
            # バッファサイズを小さくすることによる高速化
            # REF: https://qiita.com/iwatake2222/items/b8c442a9ec0406883950
            self.cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # カメラに接続できなかった場合
            if not self.cam.isOpened():
                self.err_num = 2
            # 接続できた場合
            else:
                self.t = threading.Thread(target=self._reader)
                self.t.daemon = True
                self.t.start()
                self.err_num = 0

    def isError(self) -> int:
        """カメラ接続/解放時のエラーを返す関数．

        Returns:
            response(int): 動作終了を表すフラグ
                0: connect
                2: NotFound
        """
        return self.err_num

    def release(self) -> Tuple[int, None]:
        """カメラを解放する関数

        Returns:
            response(int): 動作終了を表すフラグ
                1: release
        """
        self.cam.release()
        return 1, None

    # grab frames as soon as they are available
    def _reader(self):
        while True:
            ret = self.cam.grab()
            if not ret:
                break

    # retrieve latest frame
    def read(self):
        ret, frame = self.cam.retrieve()
        return ret, frame


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
    """
    画像の色空間を変換する関数。変換には OpenCV の関数を使用する。
    入力された画像は、コピーされず直接変換される点に注意。

    Args:
        src (np.ndarray): 変換前の画像
        color_type (str): 変更後の色空間
        * Gray: グレースケール
        * HSV: HDV空間

    Return:
        det (np.ndarray): 変換後の画像
    """
    try:
        if color_type == "Gray":
            # dst = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
            dst = AutoGrayScale(src, clearly=True)
        elif color_type == "HSV":
            dst = cv2.cvtColor(src, cv2.COLOR_RGB2HSV)
        else:
            raise ValueError("選択された変換は存在しません。")
    except Exception as e:
        print(f"Color convert error: {e}")
    else:
        return dst


def ImageCvt(
    img: np.ndarray,
    Color_Space: Literal["RGB", "Gray"] = "RGB",
    Color_Density: Literal["None", "Linear", "Non-Linear", "Histogram-Flatten"] = "None",
    Binarization: Literal["None", "Global", "Otsu", "Adaptive", "Two"] = "None",
    LowerThreshold: int = 10,
    UpperThreshold: int = 150,
    AdaptiveThreshold_type: Literal["Mean", "Gaussian", "Wellner"] = "Mean",
    AdaptiveThreshold_BlockSize: int = 11,
    AdaptiveThreshold_Constant: int = 2,
    color: int = 4,
    background_color: Literal[0, 1] = 0
) -> Tuple[int, np.ndarray, np.ndarray, Union[None, float], Union[None, float]]:
    """
    入力画像に対して指定の処理を施す関数．

    Args:
        img (np.ndarray): 変換前の画像データ．
        Color_Space (Literal["RGB", "Gray"], optional):
            画像の階調．Defaults to "RGB".
        Color_Density (Literal["None", "Linear", "Non-Linear", "Histogram-Flatten"], optional):
            画像の濃度変換.
            Defaults to "None".
        Binarization (Literal["None", "Global", "Otsu", "Adaptive", "Two"], optional):
            画像の二値化.
            Defaults to "None".
        LowerThreshold (int, optional): [description]. Defaults to 10.
        UpperThreshold (int, optional): [description]. Defaults to 150.
        AdaptiveThreshold_type (Literal["Mean", "Gaussian", "Wellner"], optional):
            小領域中での閾値の計算方法．Defaults to "Mean".
            * Mean: 近傍領域の中央値を閾値とする。
            * Gaussian: 近傍領域の重み付け平均値を閾値とする。
                        重みの値はGaussian分布になるように計算。
            * Wellner:
        AdaptiveThreshold_BlockSize (int, optional):
            閾値計算に使用する近傍領域のサイズ．ただし1より大きい奇数．Defaults to 11.
        AdaptiveThreshold_Constant (int, optional): [description]. Defaults to 2.
        color (int, optional): 計算された閾値から引く定数．Defaults to 4.
        background_color (Literal[0, 1], optional): 背景の色．Defaults to 0.
            * 0: 背景が黒．
            * 1: 背景が白．

    Returns:
        Tuple[int, np.ndarray, np.ndarray]: 返り値．
            * err (int): エラーフラグ．4: 撮影エラー, 5: 画像処理成功
            * dst_org (np.ndarray): 撮影されたオリジナル画像．
            * dst_bin (np.ndarray): 画像処理された画像．
            * l_th (None|float): 下側の閾値．二値化処理を使用しなかった場合はNone．
            * u_th (None|float): 上側の閾値．2つの二値化処理以外を指定した場合はNone．
    """
    l_th = u_th = None
    dst = img.copy()

    # ------------------ #
    # 撮影した画像を変換する #
    # ------------------ #
    # 色空間変換
    if Color_Space != "RGB":
        dst = Color_cvt(dst, Color_Space)
    # 濃度変換
    if Color_Density != "None":
        dst = Contrast_cvt(dst, Color_Density)
    # 二値化処理
    if Binarization == "None":
        return 5, img, dst
    else:
        # 大域的二値化処理
        if Binarization == "Global":
            if Color_Space == "RGB":
                r, g, b = cv2.split(img)
                if color == 0:  # Red
                    l_th, dst = GlobalThreshold(r, threshold=LowerThreshold)
                elif color == 1:  # Green
                    l_th, dst = GlobalThreshold(g, threshold=LowerThreshold)
                elif color == 2:
                    l_th, dst = GlobalThreshold(b, threshold=LowerThreshold)
                else:
                    pass
            else:
                l_th, dst = GlobalThreshold(dst, threshold=LowerThreshold)
        # 大津の二値化処理
        elif Binarization == "Otsu":
            if Color_Space == "RGB":
                r, g, b = cv2.split(img)
                if color == 0:
                    l_th, dst = GlobalThreshold(r, Type="Otsu")
                elif color == 1:
                    l_th, dst = GlobalThreshold(g, Type="Otsu")
                elif color == 2:
                    l_th, dst = GlobalThreshold(b, Type="Otsu")
                else:
                    pass
            else:
                l_th, dst = GlobalThreshold(dst, Type="Otsu")
        # 適応的二値化処理
        elif Binarization == "Adaptive":
            if Color_Space == "RGB":
                r, g, b = cv2.split(img)
                if color == 0:
                    dst = AdaptiveThreshold(
                        img=r,
                        method=str(AdaptiveThreshold_type),
                        block_size=AdaptiveThreshold_BlockSize,
                        C=AdaptiveThreshold_Constant,
                    )
                elif color == 1:
                    dst = AdaptiveThreshold(
                        img=g,
                        method=str(AdaptiveThreshold_type),
                        block_size=AdaptiveThreshold_BlockSize,
                        C=AdaptiveThreshold_Constant,
                    )
                elif color == 2:
                    dst = AdaptiveThreshold(
                        img=b,
                        method=str(AdaptiveThreshold_type),
                        block_size=AdaptiveThreshold_BlockSize,
                        C=AdaptiveThreshold_Constant,
                    )
                else:
                    pass
            else:
                dst = AdaptiveThreshold(
                    img=dst,
                    method=str(AdaptiveThreshold_type),
                    block_size=AdaptiveThreshold_BlockSize,
                    C=AdaptiveThreshold_Constant,
                )
        # 2つの閾値を用いた二値化処理
        elif Binarization == "Two" and color != 5:
            dst = TwoThreshold(
                img=dst,
                LowerThreshold=LowerThreshold,
                UpperThreshold=UpperThreshold,
                PickupColor=color,
            )
            l_th = LowerThreshold
            u_th = UpperThreshold

        if background_color == 1:
            dst = cv2.bitwise_not(dst)
    return 5, dst, l_th, u_th


def SnapshotCvt(
    cam: cv2.VideoCapture,
    Color_Space: Literal["RGB", "Gray"] = "RGB",
    Color_Density: Literal["None", "Linear", "Non-Linear", "Histogram-Flatten"] = "None",
    Binarization: Literal["None", "Global", "Otsu", "Adaptive", "Two"] = "None",
    LowerThreshold: int = 10,
    UpperThreshold: int = 150,
    AdaptiveThreshold_type: Literal["Mean", "Gaussian", "Wellner"] = "Mean",
    AdaptiveThreshold_BlockSize: int = 11,
    AdaptiveThreshold_Constant: int = 2,
    color: int = 4,
    background_color: Literal[0, 1] = 0
) -> Tuple[int, np.ndarray, np.ndarray, Union[None, float], Union[None, float]]:
    """
    スナップショットを撮影し，二値化処理を行う関数．

    Args:
        cam (cv2.VideoCapture):
            接続しているカメラ情報．
        Color_Space (Literal["RGB", "Gray"], optional):
            画像の階調．Defaults to "RGB".
        Color_Density (Literal["None", "Linear", "Non-Linear", "Histogram-Flatten"], optional):
            画像の濃度変換.
            Defaults to "None".
        Binarization (Literal["None", "Global", "Otsu", "Adaptive", "Two"], optional):
            画像の二値化.
            Defaults to "None".
        LowerThreshold (int, optional): [description]. Defaults to 10.
        UpperThreshold (int, optional): [description]. Defaults to 150.
        AdaptiveThreshold_type (Literal["Mean", "Gaussian", "Wellner"], optional):
            小領域中での閾値の計算方法．Defaults to "Mean".
            * Mean: 近傍領域の中央値を閾値とする．
            * Gaussian: 近傍領域の重み付け平均値を閾値とする．
                        重みの値はGaussian分布になるように計算．
            * Wellner:
        AdaptiveThreshold_BlockSize (int, optional):
            閾値計算に使用する近傍領域のサイズ．ただし1より大きい奇数．Defaults to 11.
        AdaptiveThreshold_Constant (int, optional): [description]. Defaults to 2.
        color (int, optional): 計算された閾値から引く定数．Defaults to 4.
        background_color (Literal[0, 1], optional): 背景の色．Defaults to 0.
            * 0: 背景が黒．
            * 1: 背景が白．

    Returns:
        Tuple[int, np.ndarray, np.ndarray]: 返り値．
            * err (int): エラーフラグ．4: 撮影エラー, 5: 画像処理成功
            * dst_org (np.ndarray): 撮影されたオリジナル画像．
            * dst_bin (np.ndarray): 画像処理された画像．
            * l_th (None|float): 下側の閾値．二値化処理を使用しなかった場合はNone．
            * u_th (None|float): 上側の閾値．2つの二値化処理以外を指定した場合はNone．
    """
    l_th = u_th = None
    err, dst_org = Snapshot(cam)

    if err != 3:
        return 4, [], []  # WebCam_NotGetImage

    err, dst_bin, l_th, u_th = ImageCvt(
        dst_org,
        Color_Space=Color_Space,
        Color_Density=Color_Density,
        Binarization=Binarization,
        LowerThreshold=LowerThreshold,
        UpperThreshold=UpperThreshold,
        AdaptiveThreshold_type=AdaptiveThreshold_type,
        AdaptiveThreshold_BlockSize=AdaptiveThreshold_BlockSize,
        AdaptiveThreshold_Constant=AdaptiveThreshold_Constant,
        color=color,
        background_color=background_color
    )

    return err, dst_org, dst_bin, l_th, u_th


def Contours(
    rgb_img: np.ndarray,
    bin_img: np.ndarray,
    CalcCOG: Literal["image", "outline"] = "image",
    Retrieval: Literal["LIST", "EXTERNAL", "CCOMP", "TREE"] = "TREE",
    Approximate: Literal["Keep", "Not-Keep"] ="Keep",
    orientation: bool = False,
    drawing_figure: bool = True,
) -> Tuple[Union[List[float], None], np.ndarray]:
    """スナップショットの撮影からオブジェクトの重心位置計算までの一連の画像処理を行う関数。

    Args:
        rgb_img (np.ndarray): 計算された重心位置を重ねて表示するRGB画像
        bin_img (np.ndarray): 重心計算対象の二値画像．
        CalcCOG (Literal["image", "outline"], optional):
            重心位置の計算対象を指定．Default to "image".
        Retrieval (Literal["LIST", "EXTERNAL", "CCOMP", "TREE"], optional):
            2値画像の画素値が 255 の部分と 0 の部分を分離した際に，その親子関係を保持するか指定．
            Default to "TREE".
            * "LIST": 親子関係を無視する
            * "EXTERNAL": 最外の輪郭を検出する
            * "CCOMP": 2つの階層に分類する
            * "TREE": 全階層情報を保持する
        Approximate (Literal["Keep", "Not-Keep"], optional):
            輪郭の中間点を保持するか指定．Default to "Keep".
        orientation (bool, optional):
            オブジェクトの輪郭情報に基づいて姿勢を推定する関数．
            `CalcCOG = "outline"` の場合のみ適用可能．Default to False.
        drawing_figure (bool, optional): 図を描画する．Default to True.
    Returns:
        COG (List[float]): COG=[x, y, angle]
            オブジェクトの重心座標と，そのオブジェクトの2D平面での回転角度．
        rgb_img (np.ndarray): [W, H, C] の rgb 画像．
    """
    COG = None
    CalcCOGMode = {
        "image": 0,
        "outline": 1,
    }
    # 輪郭情報
    RetrievalMode = {
        "LIST": cv2.RETR_LIST,
        "EXTERNAL": cv2.RETR_EXTERNAL,
        "CCOMP": cv2.RETR_CCOMP,
        "TREE": cv2.RETR_TREE,
    }
    # 輪郭の中間点情報
    ApproximateMode = {
        "Keep": cv2.CHAIN_APPROX_NONE,
        "Not-Keep": cv2.CHAIN_APPROX_SIMPLE,
    }

    if type(bin_img) != np.ndarray:
        raise TypeError("入力はnumpy配列を使用してください．")
    elif bin_img.max == 0:
        raise ValueError("画像にオブジェクトが映っていません．")
    try:
        COG, rgb_img = CenterOfGravity(
            rgb_img=rgb_img,
            bin_img=bin_img,
            RetrievalMode=RetrievalMode[Retrieval],
            ApproximateMode=ApproximateMode[Approximate],
            min_area=100,
            cal_Method=CalcCOGMode[CalcCOG],
            orientation=orientation,
            drawing_figure=drawing_figure,
        )
    except Exception as e:
        print(f"Gravity center position calculation error: {e}")
    finally:
        return COG, rgb_img


if __name__ == "__main__":
    response, cam = WebCam_OnOff(device_num=0)
    if response == 1:
        response, img = Snapshot(cam=cam)
        if response == 1:
            Preview(img, preview="plt")
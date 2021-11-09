import sys
from typing import Tuple, Union, List, Literal

sys.path.append(".")
sys.path.append("..")

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

    def isError(self):
        return self.err_num

    def release(self):
        self.cam.release()
        cv2.destroyAllWindows()
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
    Color_Density: Literal["なし", "線形濃度変換", "ヒストグラム平坦化"] = "なし",
    Binarization: Literal["なし", "Global", "Otsu", "Adaptive", "Two"] = "なし",
    LowerThreshold: int = 10,
    UpperThreshold: int = 150,
    AdaptiveThreshold_type: Literal["Mean", "Gaussian", "Wellner"] = "Mean",
    AdaptiveThreshold_BlockSize: int = 11,
    AdaptiveThreshold_Constant: int = 2,
    color: int = 4,
) -> Tuple[int, np.ndarray, np.ndarray]:
    dst = img.copy()

    # ---------------------------
    # 撮影した画像を変換する。
    # ---------------------------
    # 色空間変換
    if Color_Space != "RGB":
        dst = Color_cvt(dst, Color_Space)
    # 濃度変換
    if Color_Density != "なし":
        dst = Contrast_cvt(dst, Color_Density)
    # 二値化処理
    if Binarization != "なし":  # 二値化処理
        if Binarization == "Global":  # 大域的二値化処理
            dst = GlobalThreshold(dst, threshold=LowerThreshold)
        elif Binarization == "Otsu":  # 大津の二値化処理
            dst = GlobalThreshold(dst, Type="Otsu")
        elif Binarization == "Adaptive":
            dst = AdaptiveThreshold(
                img=dst,
                method=str(AdaptiveThreshold_type),
                block_size=AdaptiveThreshold_BlockSize,
                C=AdaptiveThreshold_Constant,
            )
        elif Binarization == "Two":  # 2つの閾値を用いた二値化処理
            # ピックアップする色を番号に変換

            dst = TwoThreshold(
                img=dst,
                LowerThreshold=LowerThreshold,
                UpperThreshold=UpperThreshold,
                PickupColor=color,
            )

    return 5, img, dst


def SnapshotCvt(
    cam: cv2.VideoCapture,
    Color_Space: Literal["RGB", "Gray"] = "RGB",
    Color_Density: Literal["なし", "線形濃度変換", "非線形濃度変換", "ヒストグラム平坦化"] = "なし",
    Binarization: Literal["なし", "Global", "Otsu", "Adaptive", "Two"] = "なし",
    LowerThreshold: int = 10,
    UpperThreshold: int = 150,
    AdaptiveThreshold_type: Literal["Mean", "Gaussian", "Wellner"] = "Mean",
    AdaptiveThreshold_BlockSize: int = 11,
    AdaptiveThreshold_Constant: int = 2,
    color: int = 4,
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    スナップショットを撮影し，二値化処理を行う関数．

    Args:
        cam (cv2.VideoCapture): 接続しているカメラ情報
        Color_Space (Literal[, optional): 画像の階調．["RGB", "Gray"]． Defaults to "RGB".
        Color_Density (Literal[, optional): 画像の濃度変換.
        * ["線形濃度変換", "非線形濃度変換", "ヒストグラム平坦化"]．
        * Defaults to "なし".
        Binarization (Literal[, optional): 画像の二値化. Defaults to "なし".
        LowerThreshold (int, optional): [description]. Defaults to 10.
        UpperThreshold (int, optional): [description]. Defaults to 150.
        AdaptiveThreshold_type (Literal[, optional): [description]. Defaults to "Mean".
        AdaptiveThreshold_BlockSize (int, optional): [description]. Defaults to 11.
        AdaptiveThreshold_Constant (int, optional): [description]. Defaults to 2.
        color (int, optional): [description]. Defaults to 4.

    Returns:
        Tuple[int, np.ndarray, np.ndarray]: 返り値
        * err (int): エラーフラグ．4: 撮影エラー, 5: 画像処理成功
        * dst_org (np.ndarray): 撮影されたオリジナル画像
        * dst_bin (np.ndarray): 画像処理された画像
    """
    dst_org = dst_bin = None
    err, dst_org = Snapshot(cam)

    if err != 3:
        return 4, [], []  # WebCam_NotGetImage

    err, _, dst_bin = ImageCvt(
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
    )

    return err, dst_org, dst_bin


def Contours(
    rgb_img: np.ndarray,
    bin_img: np.ndarray,
    CalcCOG: Literal["画像から重心を計算", "輪郭から重心を計算"],
    Retrieval: Literal["親子関係を無視する", "最外の輪郭を検出する", "2つの階層に分類する", "全階層情報を保持する"],
    Approximate: Literal["中間点を保持する", "中間点を保持しない"],
    orientation: bool = False,
    drawing_figure: bool = True,
) -> Tuple[Union[List[float], None], np.ndarray]:
    """スナップショットの撮影からオブジェクトの重心位置計算までの一連の画像処理を行う関数。

    Args:
        rgb_img (np.ndarray): 計算された重心位置を重ねて表示するRGB画像
        bin_img (np.ndarray): 重心計算対象の二値画像．
        CalcCOG (Literal["画像から重心を計算", "輪郭から重心を計算"]): 重心位置の計算対象を指定．
        Retrieval (Literal["親子関係を無視する", "最外の輪郭を検出する", "2つの階層に分類する", "全階層情報を保持する"]): 2値画像の画素値が 255 の部分と 0 の部分を分離した際に，その親子関係を保持するか指定．
        Approximate (Literal["中間点を保持する", "中間点を保持しない"]): 輪郭の中間点を保持するか指定．
        orientation (bool, optional): オブジェクトの輪郭情報に基づいて姿勢を推定する関数．
        `cal_Method = 1` の場合のみ適用可能．Default to False.
        drawing_figure (bool, optional): 図を描画する．Default to True.
    Returns:
        COG(List[float]): COG=[x, y, angle], オブジェクトの重心座標と，そのオブジェクトの2D平面での回転角度．
        rgb_img (np.ndarray): [W, H, C] の rgb画像．
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

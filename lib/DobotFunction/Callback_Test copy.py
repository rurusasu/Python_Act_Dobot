import sys
from queue import Queue
from typing import Any, Callable, Dict, Iterable, List, Literal, Mapping, Union
from threading import Thread, Timer

sys.path.append("../../")

import cv2

from lib.DobotDLL import DobotDllType as dType
from lib.DobotFunction.Camera import SnapshotCvt, Contours

ptpMoveModeDict = {
    "JumpCoordinate": dType.PTPMode.PTPJUMPXYZMode,
    "MoveJCoordinate": dType.PTPMode.PTPMOVJXYZMode,
    "MoveLCoordinate": dType.PTPMode.PTPMOVLXYZMode,
}

value = 1
value_2 = 10


class TimerFunction(Timer):
    def __init__(
        self,
        interval: float,
        function: Callable[..., Any],
        args: Union[Iterable[Any], None] = [],
        kwargs: Union[Mapping[str, Any], None] = {},
    ) -> None:
        Timer.__init__(self, interval, self.run, args, kwargs)
        self.thread = None
        self.function = function
        # self.data_que = Queue()  # ワーカープロセスへ送るデータ
        self.ui_que = Queue()  # ワーカーから送られてくるデータ

    def run(self):
        self.thread = Timer(self.interval, self.run)
        self.thread.start()
        self.function(self.ui_que, *self.args, **self.kwargs)

    def cancel(self):
        if self.thread is not None:
            self.thread.cancel()
            self.thread.join()
            del self.thread

    def GetValue(self):
        return self.ui_que.get_nowait()


def Test(ui_que: Queue, err, err_2) -> Queue:
    global value, value_2
    value += err
    value_2 += err_2

    return_param = {"value": value, "value_2": value_2}
    ui_que.put(return_param)


def _VF(
    ui_que: Queue,
    api,
    cam: cv2.VideoCapture,
    values,
    control_law: Literal["P", "PI"] = "PI",
):
    if values["-color_R-"]:
        color = 0
    elif values["-color_G-"]:
        color = 1
    elif values["-color_B-"]:
        color = 2
    elif values["-color_W-"]:
        color = 3
    elif values["-color_Bk-"]:
        color = 4

    dst_org = dst_bin = None
    COG = None
    return_param = {"dst_org": dst_org, "dst_bin": dst_bin, "pose": None, "COG": COG}
    ui_que.put(return_param)
    # スナップショット撮影
    err, dst_org, dst_bin = SnapshotCvt(
        cam,
        Color_Space=values["-Color_Space-"],
        Color_Density=values["-Color_Density-"],
        Binarization=values["-Binarization-"],
        LowerThreshold=int(values["-LowerThreshold-"]),
        UpperThreshold=int(values["-UpperThreshold-"]),
        AdaptiveThreshold_type=values["-AdaptiveThreshold_type-"],
        AdaptiveThreshold_BlockSize=int(values["-AdaptiveThreshold_BlockSize-"]),
        AdaptiveThreshold_Constant=int(values["-AdaptiveThreshold_Constant-"]),
        color=color,
    )

    # 撮影エラーが発生した場合
    if err != 5:
        ui_que.put(return_param)
        # 処理を中断して切り上げる
        return
    else:
        return_param["dst_org"] = dst_org
        return_param["dst_bin"] = dst_bin
        ui_que.put(return_param)

    """
    # 重心位置計算
    COG, dst_org = Contours(
        rgb_img=dst_org,
        bin_img=dst_bin,
        CalcCOG=str(values["-CalcCOGMode-"]),
        Retrieval=str(values["-RetrievalMode-"]),
        Approximate=str(values["-ApproximateMode-"]),
        orientation=True,
        drawing_figure=False,
    )

    # 重心位置計算エラー時
    if COG is None:
        ui_que.put(return_param)
        # 処理を中断して切り上げる
        return
    else:
        return_param["dst_org"] = dst_org
        return_param["COG"] = COG

    # 画像座標系の中心座標を算出
    if len(dst_org.shape) == 2:
        y_r, x_r = dst_org.shape
    elif len(dst_org.shape) == 3:
        y_r, x_r, _ = dst_org.shape
    else:
        ValueError("Image size is incorrect.")
    y_r, x_r = y_r / 2, x_r / 2

    # 目標位置との偏差
    e_x = e_y = 0
    e_x = COG[0] - x_r
    e_y = COG[1] - y_r

    # 現在の Dobot の手先位置情報取得
    # Dobotの姿勢情報を保存する辞書
    current_pose = {
        "x": 0,
        "y": 0,
        "z": 0,
        "r": 0,
        "joint1Angle": 0,
        "joint2Angle": 0,
        "joint3Angle": 0,
        "joint4Angle": 0,
    }
    pose = dType.GetPose(api)
    for num, key in enumerate(current_pose.keys()):
        current_pose[key] = round(pose[num], 2)  # 繰り返し誤差 0.2 mm なので入力も合わせる

    # 計算した距離の誤差が±10以内の場合
    if (-10 <= e_x <= 10) and (-10 <= e_y <= 10):
        return_param["COG"] = COG
        return_param["pose"] = current_pose
        ui_que.put(return_param)
        return

    # ゲイン
    K_p = float(values["-Kp-"])
    K_i = float(values["-Ki-"])

    # 目標座標を算出する
    if control_law == "P":
        # P 制御
        x = -K_p * e_y
        y = -K_p * e_x
    elif control_law == "PI":
        # PI 制御
        # 偏差
        sum_err = {
            "x": 0,  # サンプリング時間毎にx方向の誤差を積分していく ⇒ この誤差はx方向の積分制御で必要
            "y": 0,  # サンプリング時間毎にy方向の誤差を積分していく ⇒ この誤差はy方向の積分制御で必要
        }
        sum_err["x"] += e_x
        sum_err["y"] += e_y
        x = -K_p * e_y - K_i * sum_err["y"]
        y = -K_p * e_x - K_i * sum_err["x"]

    # 現在の手先座標に画像座標系での目標位置を加える
    current_pose["x"] += x
    current_pose["y"] += y

    # Dobotの手先を目標位置まで移動させる。
    lastIndex = dType.SetPTPCmd(
        api,
        ptpMoveModeDict[values["-MoveMode-"]],
        current_pose["x"],
        current_pose["y"],
        current_pose["z"],
        current_pose["r"],
        1,
    )[0]
    # Wait for Executing Last Command
    while lastIndex > dType.GetQueuedCmdCurrentIndex(api)[0]:
        pass

    pose = dType.GetPose(api)
    for num, key in enumerate(current_pose.keys()):
        current_pose[key] = round(pose[num], 2)  # 繰り返し誤差 0.2 mm なので入力も合わせる

    return_param["pose"] = current_pose
    ui_que.put(return_param)
    return
    """


if __name__ == "__main__":
    import time

    from lib.DobotDLL import DobotDllType as dType
    from lib.DobotFunction.Communication import (
        Connect_Disconnect,
    )

    # api = dType.load()  # Dobot 制御ライブラリの読み出し
    # connection = False  # Dobotの接続状態
    # connection = Connect_Disconnect(connection, api)
    # pose = dType.GetPose(api)

    values = {
        "-MoveMode-": "MoveJCoordinate",
        "-Color_Space-": "RGB",
        "-Color_Density-": "なし",
        "-Binarization-": "Two",
        "-LowerThreshold-": "103",
        "-UpperThreshold-": "128",
        "-AdaptiveThreshold_type-": "Mean",
        "-AdaptiveThreshold_BlockSize-": "11",
        "-AdaptiveThreshold_Constant-": "2",
        "-CalcCOGMode-": "輪郭から重心を計算",
        "-RetrievalMode-": "2つの階層に分類する",
        "-ApproximateMode-": "中間点を保持する",
        "-color_R-": False,
        "-color_G-": False,
        "-color_B-": True,
        "-color_W-": False,
        "-color_Bk-": False,
        "-Kp-": 0.05,
        "-Ki-": 0.01,
    }
    api = 0  # dummy
    device_num = 0
    cam = cv2.VideoCapture(device_num, cv2.CAP_DSHOW)

    # err = 1
    # err_2 = 2
    # vf = TimerFunction(1, Test, {err ,err_2})

    vf = TimerFunction(1, _VF, [api, cam, values])

    vf.start()
    time.sleep(3)

    value = vf.GetValue()
    print(value)

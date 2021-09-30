import time
import sys
from queue import Queue
from threading import Thread

sys.path.append("../../")

import cv2

from lib.DobotFunction.Camera import SnapshotCvt, Contours


class VisualFeedback(object):
    """
    1つのスレッドを立ち上げて，その中で Visual feedback 制御を行うクラス．
    """
    def __init__(self, api, vf_cam: cv2.VideoCapture, values) -> None:
        super().__init__()

        if values["-color_R-"]:
            self.color = 0
        elif values["-color_G-"]:
            self.color = 1
        elif values["-color_B-"]:
            self.color = 2
        elif values["-color_W-"]:
            self.color = 3
        elif values["-color_Bk-"]:
            self.color = 4

        self.api = api
        self.cam = vf_cam
        self.values = values


        self.data_que = Queue() # ワーカープロセスへ送るデータ
        self.ui_que = Queue() # ワーカーから送られてくるデータ


    def Test(self, data_que: Queue, ui_que:Queue):
        """テストとして，Dobotの手先位置を指定された座標に1回移動させる関数

        Args:
            data_que (Queue): ワーカープロセスへ送るデータ
            ui_que (Queue): ワーカーから送られてくるデータ
        """
        from lib.DobotDLL import DobotDllType as dType

        ptpMoveModeDict = {
            "JumpCoordinate": dType.PTPMode.PTPJUMPXYZMode,
            "MoveJCoordinate": dType.PTPMode.PTPMOVJXYZMode,
            "MoveLCoordinate": dType.PTPMode.PTPMOVLXYZMode
        }

        # メインプロセスからデータを取得
        data = data_que.get()
        api = data["api"]
        values = data["values"]

        current_pose = {
            "x": 250,
            "y": 0,
            "z": 0,
            "r": 0,
            "joint1Angle": 0,
            "joint2Angle": 0,
            "joint3Angle": 0,
            "joint4Angle": 0
        }

        # Dobotの手先を目標位置まで移動させる。
        lastIndex = dType.SetPTPCmd(
            api,
            ptpMoveModeDict[values["-MoveMode-"]],
            current_pose["x"],
            current_pose["y"],
            current_pose["z"],
            current_pose["r"],
            1
        )[0]
        #Wait for Executing Last Command
        while lastIndex > dType.GetQueuedCmdCurrentIndex(api)[0]:
            pass

        ui_que.put("動作終了")


    def Test2(self, data_que: Queue, ui_que:Queue):
        """テストとして，Dobotの手先位置を指定された座標に連続で移動させる関数

        Args:
            data_que (Queue): ワーカープロセスへ送るデータ
            ui_que (Queue): ワーカーから送られてくるデータ
        """
        from lib.DobotDLL import DobotDllType as dType

        ptpMoveModeDict = {
            "JumpCoordinate": dType.PTPMode.PTPJUMPXYZMode,
            "MoveJCoordinate": dType.PTPMode.PTPMOVJXYZMode,
            "MoveLCoordinate": dType.PTPMode.PTPMOVLXYZMode
        }

        # メインプロセスからデータを取得
        data = data_que.get()
        api = data["api"]
        values = data["values"]

        current_pose = {
            "x": 200,
            "y": 0,
            "z": 0,
            "r": 0,
            "joint1Angle": 0,
            "joint2Angle": 0,
            "joint3Angle": 0,
            "joint4Angle": 0
        }

        for i in range(10):
            # Dobotの手先を目標位置まで移動させる。
            lastIndex = dType.SetPTPCmd(
                api,
                ptpMoveModeDict[values["-MoveMode-"]],
                current_pose["x"],
                current_pose["y"],
                current_pose["z"],
                current_pose["r"],
                1
            )[0]
            #Wait for Executing Last Command
            while lastIndex > dType.GetQueuedCmdCurrentIndex(api)[0]:
                pass

            current_pose["x"] += 5

        ui_que.put("動作終了")


    def VF_Control(self, data_que: Queue, ui_que:Queue):
        """子プロセスの処理

        Args:
            data_que (Queue): ワーカープロセスへ送るデータ
            ui_que (Queue): ワーカーから送られてくるデータ
        """
        from lib.DobotDLL import DobotDllType as dType
        #from lib.DobotFunction.Communication import SetPoseAct

        ptpMoveModeDict = {
            "JumpCoordinate": dType.PTPMode.PTPJUMPXYZMode,
            "MoveJCoordinate": dType.PTPMode.PTPMOVJXYZMode,
            "MoveLCoordinate": dType.PTPMode.PTPMOVLXYZMode
        }

        # メインプロセスからデータを取得
        data = data_que.get()
        api = data["api"]
        cam = data["cam"]
        values = data["values"]
        color = data["color"]

        # Dobotの姿勢情報を保存する辞書
        current_pose = {
            "x": 0,
            "y": 0,
            "z": 0,
            "r": 0,
            "joint1Angle": 0,
            "joint2Angle": 0,
            "joint3Angle": 0,
            "joint4Angle": 0
        }

        # 偏差
        sum_err = {
            "x": 0, # サンプリング時間毎にx方向の誤差を積分していく ⇒ この誤差はx方向の積分制御で必要
            "y": 0  # サンプリング時間毎にy方向の誤差を積分していく ⇒ この誤差はy方向の積分制御で必要
        }

        # ゲイン
        K_p = float(values["-Kp-"])
        K_i = float(values["-Ki-"])

        return_param = {
            "pose": None,
            "COG": []
        }
        try:
            while True:
                # スナップショット撮影
                _, img = SnapshotCvt(
                    cam,
                    Color_Space = values["-Color_Space-"],
                    Color_Density = values["-Color_Density-"],
                    Binarization=values["-Binarization-"],
                    LowerThreshold = int(values["-LowerThreshold-"]),
                    UpperThreshold = int(values["-UpperThreshold-"]),
                    AdaptiveThreshold_type = values["-AdaptiveThreshold_type-"],
                    AdaptiveThreshold_BlockSize = int(values["-AdaptiveThreshold_BlockSize-"]),
                    AdaptiveThreshold_Constant = int(values["-AdaptiveThreshold_Constant-"]),
                    color=color
                )

                # 重心位置計算
                COG, img = Contours(
                    img = img,
                    CalcCOG=str(values["-CalcCOGMode-"]),
                    Retrieval=str(values["-RetrievalMode-"]),
                    Approximate = str(values["-ApproximateMode-"]),
                    drawing_figure=False
                )

                # 重心位置が取得できた場合
                if COG:
                    # 画像座標系の中心座標を算出
                    if len(img.shape) == 2:
                        y_r, x_r = img.shape
                    elif len(img.shape) == 3:
                        y_r, x_r, _ = img.shape
                    else:
                        ValueError("Image size is incorrect.")
                    y_r, x_r = y_r/2, x_r/2

                    # 目標位置との偏差
                    e_x = COG[0] - x_r
                    e_y = COG[1] - y_r

                    if (-10<= e_x <= 10) and (-10 <= e_y <= 10):
                        return_param["COG"] = COG
                        return_param["pose"] = current_pose
                        ui_que.put(return_param)
                        return

                    sum_err["x"] += e_x
                    sum_err["y"] += e_y
                    # 目標座標を算出する
                    # P 制御
                    # x = -K_p * e_y
                    # y = -K_p * e_x

                    # PI 制御
                    x = -K_p*e_y - K_i * sum_err["y"]
                    y = -K_p*e_x - K_i * sum_err["x"]

                    # 現在の Dobot の手先位置情報取得
                    pose = dType.GetPose(api)
                    for num, key in enumerate(current_pose.keys()):
                        current_pose[key] = round(pose[num], 2) # 繰り返し誤差 0.2 mm なので入力も合わせる

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
                        1
                    )[0]
                    #Wait for Executing Last Command
                    while lastIndex > dType.GetQueuedCmdCurrentIndex(api)[0]:
                        #time.sleep(1)
                        pass

        except Exception as e:
            print(e)
        finally:
            return ui_que.put(return_param)




    def run(self):

        thread_run = Thread(target=self.VF_Control, args=(self.data_que, self.ui_que), daemon=True).start()
        # thread_run = Thread(target=self.Test, args=(self.data_que, self.ui_que), daemon=True).start()
        # thread_run = Thread(target=self.Test2, args=(self.data_que, self.ui_que), daemon=True).start()
        que = {
            "api": self.api,
            "cam": self.cam,
            "values": self.values,
            "color": self.color
        }
        self.data_que.put(que)

        while True:
            try:
                ui_data = self.ui_que.get_nowait()
            except:
                ui_data = None

            if ui_data:
                #print(ui_data)
                #break
                return ui_data



if __name__ == '__main__':
    from lib.DobotDLL import DobotDllType as dType
    from lib.DobotFunction.Communication import (
        Connect_Disconnect,
    )

    api = dType.load()  # Dobot 制御ライブラリの読み出し
    connection = False  # Dobotの接続状態
    connection = Connect_Disconnect(connection, api)
    pose = dType.GetPose(api)

    if connection:
        values = {
            "-MoveMode-": 'MoveJCoordinate',
            "-Color_Space-": 'RGB',
            "-Color_Density-": 'なし',
            "-Binarization-": 'Two',
            "-LowerThreshold-": '103',
            "-UpperThreshold-": '128',
            "-AdaptiveThreshold_type-": 'Mean',
            "-AdaptiveThreshold_BlockSize-": '11',
            "-AdaptiveThreshold_Constant-": '2',
            "-CalcCOGMode-": "輪郭から重心を計算",
            "-RetrievalMode-": '2つの階層に分類する',
            "-ApproximateMode-": '中間点を保持する'
        }
        device_num = 1
        cam = cv2.VideoCapture(device_num, cv2.CAP_DSHOW)


        vf = VisualFeedback(api, cam, values)
        vf.run()
# cording: utf-8

import sys, os

sys.path.append(os.getcwd())

import PySimpleGUI as sg
import numpy as np

from gui.DobotDLL import DobotDllType as dType
from gui.DobotFunction.DobotFunction import initDobot, Operation, OneAction
from ctypes import cdll

import time

# from timeout_decorator import timeout, TimeoutError

import cv2
from PIL import Image


def __filePath__(file_name):
    path = "./data/" + str(file_name)
    return path


# ----- The callback function ----- #
class Dobot_APP:
    def __init__(self):
        self.api = cdll.LoadLibrary("DobotDll.dll")
        self.CON_STR = {
            dType.DobotConnect.DobotConnect_NoError: "DobotConnect_NoError",
            dType.DobotConnect.DobotConnect_NotFound: "DobotConnect_NotFound",
            dType.DobotConnect.DobotConnect_Occupied: "DobotConnect_Occupied",
        }
        self.CurrentPose = None
        self.queue_index = 0
        self.suctioncupON = False  # True:ON, False:OFF
        self.gripper = True  # True:Close, False:Open
        self.capture = None
        self.IMAGE_org = None  # 撮影した生画像
        self.IMAGE_bin = None  # 二値化後の画像
        self.COG_Coordinate = None  # 画像の重心位置(COG: CenterOfGravity) : tuple
        self.Alignment_1 = None  # 位置合わせ（始点）: tuple
        self.Alignment_2 = None  # 位置合わせ（終点）: tuple
        self.Record = None  # 退避位置 : tuple
        # --- エラーフラグ ---#
        self.connection = 1  # Connect:1, DisConnect:0
        self.cammera_connection = 0  # Connect:1, DisConnect:0
        self.DOBOT_err = 1  # Error occurred:1, No error:0
        self.Input_err = 0  # Error occurred:1, No error:0
        self.WebCam_err = 1  # No error:0
        # --- GUIの初期化 ---#
        self.layout = self.Layout()
        self.Window = self.main()

    def Connect_click(self):
        """
        Dobotを接続する関数

        Returns
        -------
        result : int
            0 : 接続できた場合
            1 : 接続できなかった場合
            2 : すでに接続されていた場合

        """
        # Dobotがすでに接続されていた場合
        if self.connection == 0:
            return 0
        # Dobot Connect
        # ConectDobot(const char* pointName, int baudrate)
        state = dType.ConnectDobot(self.api, "", 115200)[0]
        if state != dType.DobotConnect.DobotConnect_NoError:
            return 1  # Dobotに接続できなかった場合
        else:
            dType.SetCmdTimeout(self.api, 3000)
            return 0  # Dobotに接続できた場合

    def Disconnect_click(self):
        """
        Dobotとの接続を切断する関数

        Returns
        -------
        result : int
            result : int
            0 : 接続されていない場合
            1 : 切断できた場合
        """
        # Dobotが接続されていない場合
        if self.connection == 0:
            return 0
        # DobotDisconnect
        dType.DisconnectDobot(self.api)
        return 1

    def SetJointPose_click(self, pose):
        """
        指定された作業座標系にアームの先端を移動させる関数
        関節座標系で移動

        Parameters
        ----------
        pose : list
            デカルト座標系もしくは関節座標系での移動先を示したリスト
            パラメータ数4個

        Returns
        -------
        response : int
            0 : 応答なし
            1 : 応答あり
        """
        response = 0  # Dobotからの応答 1:あり, 0:なし
        if self.connection == 0 or self.DOBOT_err == 1:
            self.DOBOT_err = 1
            return response

        timeout(5)
        try:
            dType.SetPTPCmd(
                self.api,
                dType.PTPMode.PTPMOVJANGLEMode,
                pose[0],
                pose[1],
                pose[2],
                pose[3],
                self.queue_index,
            )
        except TimeoutError:
            self.DOBOT_err = 1
            return response
        response = 1
        return response

    def SnapshotAndConv(self, values):
        """
        スナップショットの撮影と、以下の変換を同時に行う関数
        変換方法：
            1. 大局的閾値処理
            2. 適応的閾値処理
            3. 大津の二値化
            4. 上下の閾値による二値化

        Parameter
        ---------
        values: dictionary
            ボタン操作による返り値
        capture : OpenCV型
            接続しているカメラ情報
        Window
        PySimpleGUIのウインドウ画面

        Returns
        -------
        response : int
            0: 変換成功
            1. カメラ接続
            3. スナップショットの撮影エラー
            4. 画像の変換エラー
            5. プロパティの選択エラー
            6. 変換画像が存在しない
        """
        if self.capture is None:  # カメラが接続されていない場合
            return 1, None, None

        # -----------------
        # 静止画を撮影する。
        # -----------------
        result, img = Snapshot(self.capture)
        if (result != 0) or (result is None):
            return 3, None, None
        sg.popup("スナップショットを撮影しました。", title="スナップショット")
        IMAGE_org = img.copy()  # 撮影した画像を保存する
        cv2.imshow("Snapshot", img)  # スナップショットを表示する

        # ------------------
        # 画像の解像度を表示
        # ------------------
        [y, x, z] = img.shape
        # 画面上にスナップショットした画像の縦横の長さおよびチャンネル数を表示する。
        self.Window["-IMAGE_width-"].update(str(x))
        self.Window["-IMAGE_height-"].update(str(y))
        self.Window["-IMAGE_channel-"].update(str(z))

        # ---------------------------
        # 撮影した画像をRGBに変換する。
        # ---------------------------
        response, img_rgb = AnyImage2RGB(img)
        if response != 0:
            return 4, None, None

        # 二値化の変換タイプを選択する。
        Type = ThresholdTypeOption(values["-Threshold_type-"])
        if Type is None:
            return 5, None, None
        # -------------------- #
        #   GlobalThreshold    #
        # -------------------- #
        if values["-GlobalThreshold-"]:
            # ------------------ #
            #    大津の二値化     #
            # ------------------ #
            if values["-OTSU-"]:
                result, IMAGE_bin = OtsuThreshold(img_rgb, values["-Gaussian-"])
            # 選択されていない　⇒　大局的閾値処理を行う場合
            else:
                threshold = float(values["-threshold-"])
                result, IMAGE_bin = GlobalThreshold(
                    img_rgb, values["-Gaussian-"], threshold, Type=Type
                )
            if result != 0:
                return 4, None, None

        # -------------------- #
        #  AdaptiveThreshold   #
        # -------------------- #
        elif values["-AdaptiveThreshold-"]:
            # 適応的処理のタイプを選択する。
            method = AdaptiveThresholdTypeOption(values["-AdaptiveThreshold_type-"])
            if method is None:
                return 5, None, None

            # 処理方法が適切か判定
            if (
                (values["-Threshold_type-"] == "TRUNC")
                or (values["-Threshold_type-"] == "TOZERO")
                or (values["-Threshold_type-"] == "TOZERO_INV")
            ):
                return 5, None, None
            # BlockSizeが奇数か判定
            block_size = int(values["-AdaptiveThreshold_BlockSize-"])
            if block_size % 2 == 0:
                return 5, None, None

            const = int(values["-AdaptiveThreshold_constant-"])
            IMAGE_bin = AdaptiveThreshold(
                img, values["-Gaussian-"], method, Type, block_size=block_size, C=const
            )

        # -------------------- #
        #     TwoThreshold     #
        # -------------------- #
        elif values["-Twohreshold-"]:
            LowerThresh = int(values["-LowerThreshold-"])  # 下側の閾値
            UpperThresh = int(values["-UpperThreshold-"])  # 上側の閾値
            # 抽出したい色を選択
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

            result, IMAGE_bin = TwoThreshold(
                img, values["-Gaussian-"], LowerThresh, UpperThresh, color, Type
            )
            # エラー判定
            if result == 5:
                return 5, None, None
            elif result == 6:
                return 6, None, None
        return 0, IMAGE_org, IMAGE_bin

    def Contours(self, values, Window):
        # 静止画を撮影し二値化
        WebCam_err, IMAGE_org, IMAGE_bin = self.SnapshotAndConv(values)
        if WebCam_err != 0:
            return WebCam_err, None, None

        # 輪郭の階層情報の保持方法を選択
        kernalShape = KernelShapeOption(values["-KernelShape-"])
        # 輪郭の階層情報の保持方法を選択
        Mode = ContourRetrievalModeOption(values["-ContourRetrievalMode-"])
        # 輪郭の近似方法を選択
        Method = ApproximateModeOption(values["-ApproximateMode-"])

        if (
            (kernalShape is None) or (Mode is None) or (Method is None)
        ):  # いずれかのパラメータがNoneのとき
            WebCam_err = 5
            return WebCam_err, None, None

        img = IMAGE_bin.copy()
        contours, img = ExtractContours(
            img,
            kernelShape=kernalShape,
            RetrievalMode=Mode,
            ApproximateMode=Method,
            min_area=100,
        )

        if values["-CenterOfGravity-"] == "画像をもとに計算":
            cal_Method = 0
        else:
            cal_Method = 1

        G = CenterOfGravity(img, contours, cal_Method)
        if G is None:
            WebCam_err = 7
            return WebCam_err, None, None

        cv2.circle(img, G, 4, 100, 2, 4)  # 重心位置を円で表示
        cv2.imshow("Convert", img)  # 画像として出力

        self.COG_Coordinate = G  # 重心座標を保存
        Window["-CenterOfGravity_x-"].update(str(G[0]))
        Window["-CenterOfGravity_y-"].update(str(G[1]))
        return WebCam_err, G, [IMAGE_org, IMAGE_bin]

    """
    ----------------------
    GUI Layout
    ----------------------
    """

    def Layout(self):
        # ----- Menu Definition ----- #
        menu_def = [
            ["File", ["Open", "Save", "Exit", "Properties"]],
            ["Edit", []],
            ["Help"],
        ]

        # ----- Column Definition ----- #
        Connect = [
            [
                sg.Button("Conect to DOBOT", key="-Connect-"),
                sg.Button("Disconnect to DOBOT", key="-Disconnect-"),
            ],
        ]

        Gripper = [
            [sg.Button("SuctionCup ON/OFF", key="-SuctionCup-")],
            [sg.Button("Gripper Open/Close", key="-Gripper-")],
        ]

        CurrentPose = [
            [sg.Button("Get Pose", size=(7, 1), key="-GetPose-")],
            [
                sg.Text("J1", size=(2, 1)),
                sg.InputText("", size=(5, 1), disabled=True, key="-JointPose1-"),
                sg.Text("X", size=(1, 1)),
                sg.InputText("", size=(5, 1), disabled=True, key="-CoordinatePose_X-"),
            ],
            [
                sg.Text("J2", size=(2, 1)),
                sg.InputText("", size=(5, 1), disabled=True, key="-JointPose2-"),
                sg.Text("Y", size=(1, 1)),
                sg.InputText("", size=(5, 1), disabled=True, key="-CoordinatePose_Y-"),
            ],
            [
                sg.Text("J3", size=(2, 1)),
                sg.InputText("", size=(5, 1), disabled=True, key="-JointPose3-"),
                sg.Text("Z", size=(1, 1)),
                sg.InputText("", size=(5, 1), disabled=True, key="-CoordinatePose_Z-"),
            ],
            [
                sg.Text("J4", size=(2, 1)),
                sg.InputText("", size=(5, 1), disabled=True, key="-JointPose4-"),
                sg.Text("R", size=(1, 1)),
                sg.InputText("", size=(5, 1), disabled=True, key="-CoordinatePose_R-"),
            ],
        ]

        SetPose = [
            [
                sg.Button("Set pose", size=(7, 1), key="-SetJointPose-"),
                sg.Button("Set pose", size=(7, 1), key="-SetCoordinatePose-"),
            ],
            [
                sg.Text("J1", size=(2, 1)),
                sg.InputText("", size=(5, 1), key="-JointPoseInput_1-"),
                sg.Text("X", size=(1, 1)),
                sg.InputText("", size=(5, 1), key="-CoordinatePoseInput_X-"),
            ],
            [
                sg.Text("J2", size=(2, 1)),
                sg.InputText("", size=(5, 1), key="-JointPoseInput_2-"),
                sg.Text("Y", size=(1, 1)),
                sg.InputText("", size=(5, 1), key="-CoordinatePoseInput_Y-"),
            ],
            [
                sg.Text("J3", size=(2, 1)),
                sg.InputText("", size=(5, 1), key="-JointPoseInput_3-"),
                sg.Text("Z", size=(1, 1)),
                sg.InputText("", size=(5, 1), key="-CoordinatePoseInput_Z-"),
            ],
            [
                sg.Text("J4", size=(2, 1)),
                sg.InputText("", size=(5, 1), key="-JointPoseInput_4-"),
                sg.Text("R", size=(1, 1)),
                sg.InputText("", size=(5, 1), key="-CoordinatePoseInput_R-"),
            ],
        ]

        # 画像上の位置とDobotの座標系との位置合わせを行うGUI
        Alignment = [
            [sg.Button("MoveToThePoint", size=(16, 1), key="-MoveToThePoint-"),],
            [
                sg.Button("Set", size=(8, 1), key="-Record-"),
                sg.Button("Set", size=(8, 1), key="-Set_x1-"),
                sg.Button("Set", size=(8, 1), key="-Set_x2-"),
            ],
            [
                sg.Text("X0", size=(2, 1)),
                sg.InputText("", size=(5, 1), disabled=True, key="-x0-"),
                sg.Text("X1", size=(2, 1)),
                sg.InputText("", size=(5, 1), disabled=True, key="-x1-"),
                sg.Text("X2", size=(2, 1)),
                sg.InputText("", size=(5, 1), disabled=True, key="-x2-"),
            ],
            [
                sg.Text("Y0", size=(2, 1)),
                sg.InputText("", size=(5, 1), disabled=True, key="-y0-"),
                sg.Text("Y1", size=(2, 1)),
                sg.InputText("", size=(5, 1), disabled=True, key="-y1-"),
                sg.Text("Y2", size=(2, 1)),
                sg.InputText("", size=(5, 1), disabled=True, key="-y2-"),
            ],
            [
                sg.Text("Z0", size=(2, 1)),
                sg.InputText("", size=(5, 1), disabled=True, key="-z0-"),
            ],
            [
                sg.Text("R0", size=(2, 1)),
                sg.InputText("", size=(5, 1), disabled=True, key="-r0-"),
            ],
        ]

        WebCamConnect = [
            [
                sg.Button("WEB CAM on/off", size=(15, 1), key="-SetWebCam-"),
                sg.Button("Preview Opened", size=(11, 1), key="-Preview-"),
                sg.Button("Snapshot", size=(7, 1), key="-Snapshot-"),
            ],
            [
                sg.InputCombo(
                    ("TOSHIBA_Web_Camera-HD", "Logicool_HD_Webcam_C270",),
                    size=(15, 1),
                    key="-WebCam_Name-",
                    readonly=True,
                ),
                sg.InputCombo(
                    (
                        "640x480",
                        "352x288",
                        "320x240",
                        "176x144",
                        "160x120",
                        "1280x720",
                        "1280x800",
                    ),
                    size=(11, 1),
                    key="-WebCam_FrameSize-",
                    readonly=True,
                ),
            ],
            [
                sg.Text("width", size=(4, 1)),
                sg.InputText(
                    "0",
                    size=(5, 1),
                    disabled=True,
                    justification="right",
                    key="-IMAGE_width-",
                ),
                sg.Text("height", size=(4, 1)),
                sg.InputText(
                    "0",
                    size=(5, 1),
                    disabled=True,
                    justification="right",
                    key="-IMAGE_height-",
                ),
                sg.Text("channel", size=(6, 1)),
                sg.InputText(
                    "0",
                    size=(5, 1),
                    disabled=True,
                    justification="right",
                    key="-IMAGE_channel-",
                ),
            ],
        ]

        Contours = [
            [
                sg.Button("Contours", size=(7, 1), key="-Contours-"),
                sg.InputCombo(
                    ("画像をもとに計算", "輪郭をもとに計算",), size=(16, 1), key="-CenterOfGravity-"
                ),
            ],
            [
                sg.Text("Gx", size=(2, 1)),
                sg.InputText(
                    "0",
                    size=(5, 1),
                    disabled=True,
                    justification="right",
                    key="-CenterOfGravity_x-",
                ),
                sg.Text("Gy", size=(2, 1)),
                sg.InputText(
                    "0",
                    size=(5, 1),
                    disabled=True,
                    justification="right",
                    key="-CenterOfGravity_y-",
                ),
            ],
        ]

        ColorOfObject = [
            [
                sg.Radio(
                    "R",
                    group_id="color",
                    background_color="grey59",
                    text_color="red",
                    key="-color_R-",
                ),
                sg.Radio(
                    "G",
                    group_id="color",
                    background_color="grey59",
                    text_color="green",
                    key="-color_G-",
                ),
                sg.Radio(
                    "B",
                    group_id="color",
                    background_color="grey59",
                    text_color="blue",
                    key="-color_B-",
                ),
                sg.Radio(
                    "W",
                    group_id="color",
                    background_color="grey59",
                    text_color="snow",
                    key="-color_W-",
                ),
                sg.Radio(
                    "Bk",
                    group_id="color",
                    default=True,
                    background_color="grey59",
                    text_color="grey1",
                    key="-color_Bk-",
                ),
            ],
        ]

        Bin_CommonSettings = [
            [
                sg.Text("閾値の処理方法"),
                sg.InputCombo(
                    ("BINARY", "BINARY_INV", "TRUNC", "TOZERO", "TOZERO_INV",),
                    size=(12, 1),
                    key="-Threshold_type-",
                    readonly=True,
                ),
            ],
            [sg.Checkbox("ヒストグラム平坦化", key="-EqualizeHist-")],
            [sg.Checkbox("ガウシアンフィルタ", key="-Gaussian-"),],
        ]

        # 画像から物体の輪郭を切り出す関数の設定部分_GUI
        ContourExtractionSettings = [
            # 輪郭のモードを指定する
            [
                sg.Text("輪郭", size=(10, 1)),
                sg.InputCombo(
                    ("親子関係を無視する", "最外の輪郭を検出する", "2つの階層に分類する", "全階層情報を保持する",),
                    size=(20, 1),
                    key="-ContourRetrievalMode-",
                    readonly=True,
                ),
            ],
            # 近似方法を指定する
            [
                sg.Text("近似方法", size=(10, 1)),
                sg.InputCombo(
                    ("中間点を保持する", "中間点を保持しない"),
                    size=(18, 1),
                    key="-ApproximateMode-",
                    readonly=True,
                ),
            ],
            # カーネルの形を指定する
            [
                sg.Text("カーネルの形", size=(10, 1)),
                sg.InputCombo(
                    ("矩形", "楕円形", "十字型"),
                    size=(6, 1),
                    key="-KernelShape-",
                    readonly=True,
                ),
            ],
        ]

        Global_Threshold = [
            [
                sg.Radio(
                    "Global Threshold", group_id="threshold", key="-GlobalThreshold-"
                ),
            ],
            [
                sg.Text("Threshold", size=(7, 1)),
                sg.InputText(
                    "127", size=(4, 1), justification="right", key="-threshold-"
                ),
            ],
            [sg.Checkbox("大津の二値化", key="-OTSU-")],
        ]

        Adaptive_Threshold = [
            [
                sg.Radio(
                    "Adaptive Threshold",
                    group_id="threshold",
                    key="-AdaptiveThreshold-",
                ),
            ],
            [
                sg.InputCombo(
                    ("MEAN_C", "GAUSSIAN_C"),
                    size=(12, 1),
                    key="-AdaptiveThreshold_type-",
                    readonly=True,
                )
            ],
            [
                sg.Text("Block Size", size=(8, 1)),
                sg.InputText(
                    "11",
                    size=(4, 1),
                    justification="right",
                    key="-AdaptiveThreshold_BlockSize-",
                ),
            ],
            [
                sg.Text("Constant", size=(8, 1)),
                sg.InputText(
                    "2",
                    size=(4, 1),
                    justification="right",
                    key="-AdaptiveThreshold_constant-",
                ),
            ],
        ]

        TwoThreshold = [
            [
                sg.Radio(
                    "TwoThreshold",
                    group_id="threshold",
                    default=True,
                    key="-Twohreshold-",
                ),
            ],
            [
                sg.Text("Lower", size=(4, 1)),
                sg.Slider(
                    range=(0, 127),
                    default_value=10,
                    orientation="horizontal",
                    size=(12, 12),
                    key="-LowerThreshold-",
                ),
            ],
            [
                sg.Text("Upper", size=(4, 1)),
                sg.Slider(
                    range=(128, 256),
                    default_value=138,
                    orientation="horizontal",
                    size=(12, 12),
                    key="-UpperThreshold-",
                ),
            ],
        ]

        layout = [
            [sg.Col(Connect), sg.Col(Gripper)],
            [
                sg.Col(CurrentPose, size=(165, 136)),
                sg.Col(SetPose, size=(165, 136)),
                sg.Frame("位置合わせの設定", Alignment),
            ],
            [sg.Col(WebCamConnect), sg.Frame("画像の重心計算", Contours),],
            # [sg.Col(RetrievalMode), sg.Col(ApproximateMode), sg.Col(KernelShape),],
            [
                sg.Frame("輪郭抽出の設定", ContourExtractionSettings, size=(10, 10)),
                sg.Frame("二値化の共通設定", Bin_CommonSettings, size=(10, 10)),
            ],
            [sg.Frame("Color of object", ColorOfObject, background_color="grey59"),],
            [
                sg.Col(Global_Threshold, size=(200, 115)),
                sg.Col(Adaptive_Threshold, size=(200, 115)),
                sg.Col(TwoThreshold, size=(200, 115)),
            ],
            [sg.Quit()],
        ]

        return layout

    """
    ----------------------
    GUI EVENT
    ----------------------
    """

    def Event(self, event, values):
        # Dobotの接続を行う
        if event == "-Connect-":
            self.connection = self.Connect_click()
            if self.connection == 0:
                sg.popup("Dobotに接続しています。", title="接続")
                self.DOBOT_err = 0
            else:
                self.DOBOT_err = 1
            DobotError(self.DOBOT_err)
            return

        # Dobotの切断を行う
        elif event == "-Disconnect-":
            result = self.Disconnect_click()
            if result == 0:
                sg.popup("Dobotに接続できません。", title="Dobotの接続")
                return
            elif result == 1:
                sg.popup("Dobotの接続を切断しました。", title="Dobotの接続")
                self.connection = 1
                return

        elif event == "-SuctionCup-":
            if (self.DOBOT_err == 0) or (
                self.DOBOT_err == 3
            ):  # Dobotが接続されている、もしくはサクションカップが動作していないとき
                if self.suctioncupON:  # サクションカップが動作している場合
                    self.DOBOT_err = GripperON(self.api, self.queue_index, False, True)
                    if self.DOBOT_err != 0:
                        DobotError(self.DOBOT_err)
                        return
                    self.suctioncupON = False
                else:
                    self.DOBOT_err = GripperON(self.api, self.queue_index, True, True)
                    if self.DOBOT_err != 0:
                        DobotError(self.DOBOT_err)
                    self.suctioncupON = True
                    return
            else:  # それ以外のエラーの場合
                DobotError(self.DOBOT_err)
                return

        elif event == "-Gripper-":
            if self.DOBOT_err != 0:  # Dobotが接続されていない時
                DobotError(self.DOBOT_err)
                return
            else:
                if self.suctioncupON:  # サクションカップが動作している場合
                    if self.gripper:  # グリッパーが閉じている場合
                        self.DOBOT_err = GripperON(
                            self.api, self.queue_index, True, False
                        )  # グリッパーを開く
                        if self.DOBOT_err != 0:
                            DobotError(self.DOBOT_err)
                            return
                        self.gripper = False

                    else:  # グリッパーが開いている場合
                        self.DOBOT_err = GripperON(
                            self.api, self.queue_index, True, True
                        )  # グリッパーを閉じる
                        if self.DOBOT_err != 0:
                            DobotError(self.DOBOT_err)
                            return
                        self.gripper = True
                else:  # サクションカップが動作していない場合
                    self.DOBOT_err = 3
                    DobotError(self.DOBOT_err)
                return

        # Dobotの現在の姿勢を取得し表示する
        elif event == "-GetPose-":
            if self.DOBOT_err != 0:  # Dobotが接続されていない時
                DobotError(self.DOBOT_err)
            else:
                self.DOBOT_err, pose = GetPose_UpdateWindow(self.api, self.Window)
                if self.DOBOT_err != 0:
                    DobotError(self.DOBOT_err)
            return

        elif event == "-SetJointPose-":
            if self.connection == 0:
                sg.popup("Dobotに接続していません。", title="Dobotの接続")
                return
            else:
                if (
                    (values["-JointPoseInput_1-"] is "")
                    and (values["-JointPoseInput_2-"] is "")
                    and (values["-JointPoseInput_3-"] is "")
                    and (values["-JointPoseInput_4-"] is "")
                ):
                    sg.popup("移動先が入力されていません。", title="入力不良")
                    self.Input_err = 1
                    return

                # 入力姿勢の中に''があるか判定
                if (
                    (values["-JointPoseInput_1-"] is "")
                    or (values["-JointPoseInput_2-"] is "")
                    or (values["-JointPoseInput_3-"] is "")
                    or (values["-JointPoseInput_4-"] is "")
                ):
                    self.DOBOT_err, CurrentPose = GetPose_click(self.api)
                    if self.DOBOT_err == 1:  # GetPoseできなかった時
                        sg.popup("姿勢情報を取得できませんでした。", title="姿勢取得")
                        return
                    else:
                        if values["-JointPoseInput_1-"] is "":
                            values["-JointPoseInput_1-"] = CurrentPose[4]
                        if values["-JointPoseInput_2-"] is "":
                            values["-JointPoseInput_2-"] = CurrentPose[5]
                        if values["-JointPoseInput_3-"] is "":
                            values["-JointPoseInput_3-"] = CurrentPose[6]
                        if values["-JointPoseInput_4-"] is "":
                            values["-JointPoseInput_4-"] = CurrentPose[7]

                # 移動後の関節角度を指定
                DestPose = [
                    float(values["-JointPoseInput_1-"]),
                    float(values["-JointPoseInput_2-"]),
                    float(values["-JointPoseInput_3-"]),
                    float(values["-JointPoseInput_4-"]),
                ]

                response_2 = self.SetJointPose_click(DestPose)
                if response_2 == 0:
                    sg.popup("Dobotからの応答がありません。", title="Dobotの接続")
                    return
                else:
                    return

        elif event == "-SetCoordinatePose-":
            if self.connection == 0:
                sg.popup("Dobotに接続していません。", title="Dobotの接続")
                return
            else:
                if (
                    (values["-CoordinatePoseInput_X-"] is "")
                    and (values["-CoordinatePoseInput_Y-"] is "")
                    and (values["-CoordinatePoseInput_Z-"] is "")
                    and (values["-CoordinatePoseInput_R-"] is "")
                ):  # 移動先が1つも入力場合
                    sg.popup("移動先が入力されていません。", title="入力不良")
                    self.Input_err = 1
                    return

                self.DOBOT_err, pose = GetPose_UpdateWindow(self.api, self.Window)
                if self.DOBOT_err != 0:  # Dobotのエラーが発生した場合
                    DobotError(self.DOBOT_err)
                    return
                else:  # 入力姿勢の中に''があるか判定
                    if values["-CoordinatePoseInput_X-"] is "":
                        values["-CoordinatePoseInput_X-"] = pose[0]
                    if values["-CoordinatePoseInput_Y-"] is "":
                        values["-CoordinatePoseInput_Y-"] = pose[1]
                    if values["-CoordinatePoseInput_Z-"] is "":
                        values["-CoordinatePoseInput_Z-"] = pose[2]
                    if values["-CoordinatePoseInput_R-"] is "":
                        values["-CoordinatePoseInput_R-"] = pose[3]

                # 移動後の関節角度を指定
                DestPose = [
                    float(values["-CoordinatePoseInput_X-"]),
                    float(values["-CoordinatePoseInput_Y-"]),
                    float(values["-CoordinatePoseInput_Z-"]),
                    float(values["-CoordinatePoseInput_R-"]),
                ]

                self.DOBOT_err = SetCoordinatePose_click(
                    self.api, DestPose, self.queue_index
                )
                if self.DOBOT_err == 1:  # Dobotのエラーが発生した場合
                    sg.popup("Dobotからの応答がありません。", title="Dobotの接続")
                    return
            return

        elif event == "-Record-":
            if self.DOBOT_err != 0:  # Dobotが接続されていない時
                DobotError(self.DOBOT_err)
            else:
                self.DOBOT_err, pose = GetPose_UpdateWindow(self.api, self.Window)
                if self.DOBOT_err != 0:  # Dobotのエラーが発生した場合
                    DobotError(self.DOBOT_err)
                else:
                    self.Window["-x0-"].update(str(pose[0]))
                    self.Window["-y0-"].update(str(pose[1]))
                    self.Window["-z0-"].update(str(pose[2]))
                    self.Window["-r0-"].update(str(pose[3]))
                    self.Record = (pose[0], pose[1], pose[2], pose[3])  # 退避位置をセット

                    self.DOBOT_err = SetCoordinatePose_click(
                        self.api, pose, self.queue_index
                    )
                    if self.DOBOT_err != 0:  # Dobotのエラーが発生した場合
                        DobotError(self.DOBOT_err)
            return

        elif event == "-Set_x1-":
            if self.DOBOT_err != 0:  # Dobotが接続されていない時
                DobotError(self.DOBOT_err)
            else:
                self.DOBOT_err, pose = GetPose_UpdateWindow(self.api, self.Window)
                if self.DOBOT_err != 0:  # Dobotのエラーが発生した場合
                    DobotError(self.DOBOT_err)
                else:
                    self.Window["-x1-"].update(str(pose[0]))
                    self.Window["-y1-"].update(str(pose[1]))
                    self.Alignment_1 = (pose[0], pose[1])  # 位置合わせ座標（始点）をセット

                    self.DOBOT_err = SetCoordinatePose_click(
                        self.api, pose, self.queue_index
                    )
                    if self.DOBOT_err != 0:  # Dobotのエラーが発生した場合
                        DobotError(self.DOBOT_err)
            return

        elif event == "-Set_x2-":
            if self.DOBOT_err != 0:  # Dobotが接続されていない時
                DobotError(self.DOBOT_err)
            else:
                self.DOBOT_err, pose = GetPose_UpdateWindow(self.api, self.Window)
                if self.DOBOT_err != 0:  # Dobotのエラーが発生した場合
                    DobotError(self.DOBOT_err)
                else:
                    self.Window["-x2-"].update(str(pose[0]))
                    self.Window["-y2-"].update(str(pose[1]))
                    self.Alignment_2 = (pose[0], pose[1])  # 位置合わせ座標（始点）をセット

                    self.DOBOT_err = SetCoordinatePose_click(
                        self.api, pose, self.queue_index
                    )
                    if self.DOBOT_err != 0:  # Dobotのエラーが発生した場合
                        DobotError(self.DOBOT_err)
            return

        elif event == "-MoveToThePoint-":
            # 位置合わせの始点と終点がセットされているか確認
            if (self.Alignment_1 is None) or (self.Alignment_2 is None):
                sg.popup("位置合わせ座標が入力されていません", title="エラー")
                return
            # 退避位置がセットされているか確認
            elif self.Record is None:
                sg.popup("退避位置が入力されていません。", title="エラー")
                return

            # 重心位置を計算する
            self.WebCam_err, self.COG_Coordinate, IMAGE = self.Contours(
                values, self.Window
            )
            if self.WebCam_err != 0:  # エラーが発生した場合
                WebCamError(self.WebCam_err)
                return

                self.IMAGE_org = IMAGE[0]
                self.IMAGE_bin = IMAGE[1]
                self.Window["-CenterOfGravity_x-"].update(str(self.COG_Coordinate[0]))
                self.Window["-CenterOfGravity_y-"].update(str(self.COG_Coordinate[1]))

            # ------------------
            # 画像の解像度を表示
            # ------------------
            [height, width, channel] = IMAGE[0].shape
            # 画面上にスナップショットした画像の縦横の長さおよびチャンネル数を表示する。
            self.Window["-IMAGE_width-"].update(str(width))
            self.Window["-IMAGE_height-"].update(str(height))
            self.Window["-IMAGE_channel-"].update(str(channel))

            # --------------------
            # Dobotの移動位置を計算
            # --------------------
            try:
                x = self.Alignment_1[0] + self.COG_Coordinate[1] * (
                    self.Alignment_2[0] - self.Alignment_1[0]
                ) / float(height)
                y = self.Alignment_1[1] + self.COG_Coordinate[0] * (
                    self.Alignment_2[1] - self.Alignment_1[1]
                ) / float(width)
            except ZeroDivisionError:  # ゼロ割りが発生した場合
                sg.popup("画像のサイズが計測されていません", title="エラー")
                return

            # 現在の座標を取得し、画面を更新する。
            self.DOBOT_err, pose = GetPose_UpdateWindow(self.api, self.Window)
            if self.DOBOT_err != 0:  # Dobotのエラーが発生した場合
                DobotError(self.DOBOT_err)
                return
            else:
                DestPose = [x, y, pose[2], pose[3]]

                # ---------------------------------
                # Dobotを重心位置の真上まで移動させる
                # ---------------------------------
                self.DOBOT_err = SetCoordinatePose_click(
                    self.api, DestPose, self.queue_index
                )
                if self.DOBOT_err != 0:  # Dobotのエラーが発生した場合
                    DobotError(self.DOBOT_err)
                    return
                time.sleep(1)
                # ----------------------
                # Dobotのグリッパーを開く
                # ----------------------
                self.DOBOT_err = GripperON(
                    self.api, self.queue_index, True, False
                )  # グリッパーを開く
                if self.DOBOT_err != 0:  # Dobotのエラーが発生した場合
                    DobotError(self.DOBOT_err)
                    return

                # -------------------------------
                # DobotをZ=-35の位置まで降下させる
                # -------------------------------
                self.DOBOT_err, pose = GetPose_UpdateWindow(
                    self.api, self.Window
                )  # 現在の位置を取得
                if self.DOBOT_err != 0:  # Dobotのエラーが発生した場合
                    DobotError(self.DOBOT_err)
                    return

                else:
                    DestPose = [x, y, float(-35), pose[3]]

                self.DOBOT_err = SetCoordinatePose_click(
                    self.api, DestPose, self.queue_index
                )  # Z=-35の位置まで移動
                if self.DOBOT_err != 0:  # Dobotのエラーが発生した場合
                    DobotError(self.DOBOT_err)
                    return
                time.sleep(2)

                # ------------------------
                # Dobotのグリッパーを閉じる
                # ------------------------
                self.DOBOT_err = GripperON(
                    self.api, self.queue_index, True, True
                )  # グリッパーを閉じる
                if self.DOBOT_err != 0:  # Dobotのエラーが発生した場合
                    DobotError(self.DOBOT_err)
                    return
                time.sleep(1)

                # -------------------------------
                # Dobotを降下前の位置まで上昇させる
                # -------------------------------
                DestPose = [x, y, pose[2], pose[3]]
                self.DOBOT_err = SetCoordinatePose_click(
                    self.api, DestPose, self.queue_index
                )  # Z=-35の位置まで移動
                if self.DOBOT_err != 0:  # Dobotのエラーが発生した場合
                    DobotError(self.DOBOT_err)
                    return

                # ----------------------------
                # Dobotを退避位置まで移動させる
                # ----------------------------
                DestPose = [
                    self.Record[0],
                    self.Record[1],
                    self.Record[2],
                    self.Record[3],
                ]
                self.DOBOT_err = SetCoordinatePose_click(
                    self.api, DestPose, self.queue_index
                )  # Z=-35の位置まで移動
                if self.DOBOT_err != 0:  # Dobotのエラーが発生した場合
                    DobotError(self.DOBOT_err)
                    return

            return

        # -------------------------- #
        #    WebCamに関するイベント   #
        # -------------------------- #
        elif event == "-SetWebCam-":
            device_num = WebCamOption(values["-WebCam_Name-"])
            if device_num is None:
                sg.popup("選択したデバイスは存在しません。", title="エラー")
                return

            # self.cammera_connection, self.capture = WebCam_OnOff_click(device_num, self.capture, self.cammera_connection)
            self.capture = WebCam_OnOff_click(device_num, self.capture)
            if self.capture is None:
                self.WebCam_err = 1
                WebCamError(self.WebCam_err)
                return
            else:
                sg.popup("WebCameraに接続しました。", title="Camの接続")

        elif event == "-Preview-":
            if self.capture is None:  # カメラが接続されていない場合
                self.WebCam_err = 1
                WebCamError(self.WebCam_err)
                return

            self.WebCam_err = PreviewOpened_click(self.capture)
            if self.WebCam_err != 0:
                WebCamError(self.WebCam_err)
                return
            else:
                sg.popup("WebCameraの画像を閉じました。", title="画像の表示")
                return

        elif event == "-Snapshot-":
            self.WebCam_err, self.IMAGE_org, self.IMAGE_bin = self.SnapshotAndConv(
                values
            )
            if self.WebCam_err != 0:
                WebCamError(self.WebCam_err)
                return

            cv2.imshow("Snapshot", self.IMAGE_org)  # 画像として出力

            return

        elif event == "-Contours-":
            self.WebCam_err, self.COG_Coordinate, IMAGE = self.Contours(
                values, self.Window
            )

            if self.WebCam_err != 0:  # エラーが発生した場合
                WebCamError(self.WebCam_err)
                return

            self.IMAGE_org = IMAGE[0]
            self.IMAGE_bin = IMAGE[1]
            self.Window["-CenterOfGravity_x-"].update(str(self.COG_Coordinate[0]))
            self.Window["-CenterOfGravity_y-"].update(str(self.COG_Coordinate[1]))

            return

    def main(self):
        return sg.Window(
            "Dobot",
            self.layout,
            default_element_size=(40, 1),
            background_color="grey90",
        )

    def loop(self):
        while True:
            event, values = self.Window.Read(timeout=10)
            if event is "Quit":
                break
            if event != "__TIMEOUT__":
                self.Event(event, values)


def DobotError(error_num):
    """
    WebCameraと画像変換に関係するエラーを表示する関数

    Parameter
    ---------
    error_num : int
        0: エラーなし
        1: Dobotに接続されていない
        2: Dobotからの応答がない
        3: サクションカップが動作していない
        4: グリッパーが動作していません
    """
    if error_num == 0:  # エラーがない場合
        return
    elif error_num == 1:
        sg.popup("Dobotに接続できません。", title="接続")
    elif error_num == 2:
        sg.popup("Dobotからの応答がありません", title="応答なし")
    elif error_num == 3:
        sg.popup("サクションカップが動作していません", title="エラー")
    elif error_num == 4:
        sg.popup("グリッパーが動作していません", title="エラー")
    else:
        sg.popup("未知のエラーです。", title="未知エラー")
    return


def WebCamError(error_num):
    """
    WebCameraと画像変換に関係するエラーを表示する関数

    Parameter
    ---------
    error_num : int
        0: エラーなし
        1: カメラ接続
        2: プレビューエラー
        3: スナップショット撮影エラー
        4: 画像の変換エラー
        5: プロパティの選択エラー
        6: 変換画像が存在しない
        7: ゼロ除算
    """
    if error_num == 0:  # エラーがない場合
        return
    elif error_num == 1:  # カメラが接続されていないとき
        sg.popup("WebCameraが接続されていません。", title="接続エラー")
    elif error_num == 2:  # プレビューが表示されない場合
        sg.popup("Previewを表示できません", title="表示エラー")
    elif error_num == 3:  # スナップショット撮影エラー
        sg.popup("スナップショットを撮影できませんでした。", title="撮影エラー")
    elif error_num == 4:  # 画像変換エラー
        sg.popup("画像の変換ができませんでした。", title=" 変換エラー")
    elif error_num == 5:  # プロパティの選択エラー
        sg.popup("存在しない変換方法です。", title="選択エラー")
    elif error_num == 6:  # 変換画像が存在しない
        sg.popup("変換画像が存在しません。", title="変換エラー")
    elif error_num == 7:  # ゼロ除算です
        sg.popup("ゼロ除算です", title="計算エラー")
    else:
        sg.popup("未知のエラーです。", title="未知エラー")
    return


def GripperON(api, queue_index, SucCupCtr=False, gripperCtr=True):
    """
    グリッパーを開閉する関数

    Parameters
    ----------
    api : Dobot型
        DobotAPIのコンストラクタ
    queue_index : True or False
        データをQueueとして送るか
    SucCupCtr : True or False
        サクションカップを起動するか
        True:  On
        False: Off
    gripperCtr : True or False
        グリッパーを開くか
        True:  閉じる
        False: 開く

    Return
    ------
    respomse : int
        0: 動作した
        4: 動作しなかった
    """
    response = 4
    timeout(5)
    try:
        dType.SetEndEffectorGripper(api, SucCupCtr, gripperCtr, queue_index)
    except TimeoutError:
        return response
    response = 0
    return response


def GetPose_click(api):
    """
    デカルト座標系と関節座標系でのDobotの姿勢を返す関数

    Parameter
    ---------
    api : Dobot型
        DobotAPIのコンストラクタ

    Returns
    -------
    response : int
        0 : 応答あり
        2 : 応答なし
    PoseParams : list
        Dobotの姿勢を格納したリスト
    """
    response = 2
    timeout(5)
    try:
        pose = dType.GetPose(api)
    except TimeoutError:
        return response, None

    response = 0
    return response, pose


def GetPose_UpdateWindow(api, Window):
    """
    姿勢を取得し、ウインドウを更新する関数

    Parameters
    ----------
    api : Dobot型
        DobotAPIのコンストラクタ
    Window
        PySimpleGUIのウインドウ画面

    """
    DOBOT_err, pose = GetPose_click(api)
    if DOBOT_err == 0:
        Window["-JointPose1-"].update(str(pose[4]))
        Window["-JointPose2-"].update(str(pose[5]))
        Window["-JointPose3-"].update(str(pose[6]))
        Window["-JointPose4-"].update(str(pose[7]))
        Window["-CoordinatePose_X-"].update(str(pose[0]))
        Window["-CoordinatePose_Y-"].update(str(pose[1]))
        Window["-CoordinatePose_Z-"].update(str(pose[2]))
        Window["-CoordinatePose_R-"].update(str(pose[3]))

    return DOBOT_err, pose


def SetCoordinatePose_click(api, pose, queue_index):
    """
    指定された作業座標系にアームの先端を移動させる関数
    デカルト座標系で移動

    Parameters
    ----------
    api : Dobot型
        DobotAPIのコンストラクタ
    pose : list
        デカルト座標系もしくは関節座標系での移動先を示したリスト
        パラメータ数4個
    queue_index : True or False
        データをQueueとして送るか

    Returns
    -------
    response : int
        0 : 応答あり
        1 : 応答なし
    """
    response = 1
    if (len(pose) != 4) and (len(pose) != 8):  # pose配列の長さが不正の場合
        return response

    timeout(5)
    try:
        dType.SetPTPCmd(
            api,
            dType.PTPMode.PTPMOVJXYZMode,
            pose[0],
            pose[1],
            pose[2],
            pose[3],
            queue_index,
        )
    except TimeoutError:
        return response

    response = 0
    return response


def WebCam_OnOff_click(device_num, capture=None):
    """
    WebCameraを読み込む関数

    Parameter
    ---------
    device_num : int
        カメラデバイスを番号で指定
        0:PC内臓カメラ
        1:外部カメラ
    capture : OpenCV型
        接続しているカメラ情報

    Return
    ------
    capture : OpenCV型
        接続したデバイス情報を返す
    """
    if capture is None:  # カメラが接続されていないとき
        capture = cv2.VideoCapture(device_num)
        # カメラに接続できなかった場合
        if not capture.isOpened():
            return None
        # 接続できた場合
        else:
            return capture

    else:  # カメラに接続されていたとき
        capture.release()
        return None


def PreviewOpened_click(capture, window_name="frame", delay=1):
    """
    webカメラの画像を表示する関数

    Parameters
    ----------
    captur : OpenCV型
        接続しているカメラ情報
    window_name : str
        画像を表示する時のウインドウ名

    Returns
    -------
    response : int
        画像表示の可否を返す
        0: 表示できた。
        2: 表示できない。
    """
    response = 2
    if capture == None:
        return response

    while True:
        ret, frame = capture.read()
        if ret:
            cv2.imshow(window_name, frame)
            if cv2.waitKey(delay) & 0xFF == ord("e"):
                cv2.destroyWindow(window_name)
                break
        else:
            break
    response = 0
    return response


def Snapshot(capture):
    """
    WebCameraでスナップショットを撮影する関数
    capture : OpenCV型
        接続しているカメラ情報

    Return
    ------
    response : int
        0: 撮影できました。
        3: 撮影できませんでした。

    frame : OpenCV型
        撮影した画像
    """
    response = 3
    if capture == None:
        return response, None

    ret, frame = capture.read()  # 静止画像をGET
    if not capture.isOpened():
        return response, None

    response = 0
    return response, frame


def AnyImage2RGB(img):
    """
    画像の変換を行う関数

    Parameters
    ----------
    img : OpenCV型
        変換前の画像

    Return
    ------
    response : int
        0: 変換できました。
        6: 画像の元データが存在しません。
    new_image
        変換した画像
    """
    response = 6
    if img is None:  # 画像の元データが存在しない場合
        return response, None

    new_image = img.copy()
    # -------------------------------#
    #             RGB               #
    # -------------------------------#
    if new_image.shape[2] == 3:  # RGB画像
        b, g, r = cv2.split(new_image)
        new_image = cv2.merge((r, g, b))
        new_image = new_image[:, :, ::-1]
    # -------------------------------#
    #             RGBA              #
    # -------------------------------#
    elif new_image.shape[2] == 4:
        b, g, r, a = cv2.split(new_image)
        new_image = cv2.merge((r, g, b, a))
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2RGB)
    # -------------------------------#
    #          Gray Scale           #
    # -------------------------------#
    else:  # モノクロ
        gamma22LUT = np.array(
            [pow(x / 255.0, 2.2) for x in range(256)], dtype="float32"
        )
        img_barL = cv2.LUT(new_image, gamma22LUT)
        img_grayL = cv2.cvtColor(img_barL, cv2.COLOR_BGR2GRAY)
        img_gray = pow(img_grayL, 1.0 / 2.2) * 255
        new_image = img_gray
    response = 0
    return response, new_image


def GlobalThreshold(img, gaussian=False, threshold=127, Type=cv2.THRESH_BINARY):
    """
    画素値が閾値より大きければある値(白色'255')を割り当て，そうでなければ別の値(黒色)を割り当てる。
    img is NoneならNoneを、変換に成功すれば閾値処理された2値画像を返す。

    Parameters
    ----------
    img : OpenCV型
        変換前の画像データ
    gaussian : True or False
        ガウシアンフィルタを適応するか選択できる。
    threshold : flaot
        2値化するときの閾値
    Type
        閾値の処理方法
        ・cv2.THRESH_BINARY
        ・cv2.THRESH_BINARY_INV
        ・cv2.THRESH_TRUNC
        ・cv2.THRESH_TOZERO
        ・cv2.THRESH_TOZERO_INV

    Returns
    -------
    img : OpenCV型
        変換後の画像データ
    response : int
        0: 変換できました。
        6: 画像の元データが存在しません。
    """
    response = 6
    if img is None:  # 画像の元データが存在しない場合
        return response, None
    if gaussian:
        img = cv2.GaussianBlur(img, (5, 5), 0)

    ret, img = cv2.threshold(img, threshold, 255, Type)
    response = 0
    return response, img


def OtsuThreshold(img, gaussian=False):
    """
    入力画像が bimodal image (ヒストグラムが双峰性を持つような画像)であることを仮定すると、
    そのような画像に対して、二つのピークの間の値を閾値として選べば良いと考えることであろう。これが大津の二値化の手法である。
    双峰性を持たないヒストグラムを持つ画像に対しては良い結果が得られないことになる。

    Parameters
    ----------
    img : OpenCV型
        変換前の画像データ
    gaussian : True or False
        ガウシアンフィルタを適応するか選択できる。

    Returns
    -------
    img : OpenCV型
        変換後の画像データ
    response : int
        0: 変換できました。
        6: 画像の元データが存在しません。
    """
    response = 6
    if img is None:  # 画像の元データが存在しない場合
        return response, None
    # 画像のチャンネル数が2より大きい場合
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ガウシアンフィルタで前処理を行う場合
    if gaussian:
        img = cv2.GaussianBlur(img, (5, 5), 0)
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    response = 0

    return response, img


def AdaptiveThreshold(
    img,
    gaussian=False,
    method=cv2.ADAPTIVE_THRESH_MEAN_C,
    Type=cv2.THRESH_BINARY,
    block_size=11,
    C=2,
):
    """
    適応的閾値処理では，画像の小領域ごとに閾値の値を計算する．
    そのため領域によって光源環境が変わるような画像に対しては，単純な閾値処理より良い結果が得られる．
    img is NoneならNoneを、変換に成功すれば閾値処理された2値画像を返す。

    Parameters
    ----------
    img : OpenCV型
        変換前の画像データ
    gaussian : True or False
        ガウシアンフィルタを適応するか選択できる。
    method
        小領域中での閾値の計算方法
        ・cv2.ADAPTIVE_THRESH_MEAN_C : 近傍領域の中央値を閾値とする。
        ・cv2.ADAPTIVE_THRESH_GAUSSIAN_C : 近傍領域の重み付け平均値を閾値とする。
                                           重みの値はGaussian分布になるように計算。
    Type
        閾値の処理方法
        ・cv2.THRESH_BINARY
        ・cv2.THRESH_BINARY_INV
    block_size : int
        閾値計算に使用する近傍領域のサイズ。
        'ただし1より大きい奇数でなければならない。'
    C : int
        計算された閾値から引く定数。
    """
    if img is None:  # 画像の元データが存在しない場合
        return None
    # 画像のチャンネル数が2より大きい場合
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gaussian:
        img = cv2.GaussianBlur(img, (5, 5), 0)

    img = cv2.adaptiveThreshold(img, 255, method, Type, block_size, C)
    return img


def TwoThreshold(
    img,
    gaussian=False,
    LowerThreshold=0,
    UpperThreshold=128,
    PickupColor=4,
    Type=cv2.THRESH_BINARY,
):
    """
    上側と下側の2つの閾値で2値化を行う。
    二値化には大局的閾値処理を用いる。

    Parameters
    ----------
    img : OpenCV型
        変換前の画像データ
    gaussian : True or False
        ガウシアンフィルタを適応するか選択できる。
    LowerThreshold : int
        下側の閾値
        範囲：0～127
    UpperThreshold : int
        上側の閾値
        範囲：128~256
    PickupColor : int
        抽出したい色を指定する。
        デフォルトは黒
        0: 赤
        1: 緑
        2: 青
        3: 白
        4: 黄色
    Type
        閾値の処理方法
        ・cv2.THRESH_BINARY
        ・cv2.THRESH_BINARY_INV
        ・cv2.THRESH_TRUNC
        ・cv2.THRESH_TOZERO
        ・cv2.THRESH_TOZERO_INV

    Returns
    -------
    result : int
        処理が成功したか確認するための返り値
        0: 変換成功
        5: 抽出したい色の選択が不正
        6: img画像が存在しなかった
    IMAGE_bw : OpenCV型
        変換後の画像データ
    """
    if img is None:  # 画像の元データが存在しない場合
        return 6, None
    r, g, b = cv2.split(img)

    # for Red
    _, IMAGE_R_bw = GlobalThreshold(r, gaussian, LowerThreshold, Type)
    _, IMAGE_R__ = GlobalThreshold(r, gaussian, UpperThreshold, Type)
    IMAGE_R__ = cv2.bitwise_not(IMAGE_R__)
    # for Green
    _, IMAGE_G_bw = GlobalThreshold(g, gaussian, LowerThreshold, Type)
    _, IMAGE_G__ = GlobalThreshold(g, gaussian, UpperThreshold, Type)
    IMAGE_G__ = cv2.bitwise_not(IMAGE_G__)
    # for Blue
    _, IMAGE_B_bw = GlobalThreshold(b, gaussian, LowerThreshold, Type)
    _, IMAGE_B__ = GlobalThreshold(b, gaussian, UpperThreshold, Type)
    IMAGE_B__ = cv2.bitwise_not(IMAGE_B__)

    if PickupColor == 0:
        IMAGE_bw = IMAGE_R_bw * IMAGE_G__ * IMAGE_B__  # 画素毎の積を計算　⇒　赤色部分の抽出
    elif PickupColor == 1:
        IMAGE_bw = IMAGE_G_bw * IMAGE_B__ * IMAGE_R__  # 画素毎の積を計算　⇒　緑色部分の抽出
    elif PickupColor == 2:
        IMAGE_bw = IMAGE_B_bw * IMAGE_R__ * IMAGE_G__  # 画素毎の積を計算　⇒　青色部分の抽出
    elif PickupColor == 3:
        IMAGE_bw = IMAGE_R_bw * IMAGE_G_bw * IMAGE_B_bw  # 画素毎の積を計算　⇒　白色部分の抽出
    elif PickupColor == 4:
        IMAGE_bw = IMAGE_R__ * IMAGE_G__ * IMAGE_B__  # 画素毎の積を計算　⇒　青色部分の抽出
    else:
        return 5, None

    return 0, IMAGE_bw


def ExtractContours(
    org_img,
    kernelShape=cv2.MORPH_RECT,
    RetrievalMode=cv2.RETR_LIST,
    ApproximateMode=cv2.CHAIN_APPROX_SIMPLE,
    min_area=100,
):
    """
        画像に含まれるオブジェクトの輪郭を抽出する関数。
        黒い背景（暗い色）から白い物体（明るい色）の輪郭を検出すると仮定。

        Parameters
        ----------
        org_img : OpenCV型
            変換前の画像データ(二値)
        kernelShape
            モルフォロジー変換で使用する入力画像と処理の性質を決める構造的要素
            カーネルの種類
            ・cv2.MORPH_RECT: 矩形カーネル
            ・cv2.MORPH_ELLIPSE: 楕円形カーネル
            ・cv2.MORPH_CROSS: 十字型カーネル
        RetrievalMode
            輪郭の階層情報
            cv2.RETR_LIST: 輪郭の親子関係を無視する。
                           親子関係が同等に扱われるので、単なる輪郭として解釈される。
            cv2.RETR_EXTERNAL: 最も外側の輪郭だけを検出するモード
            cv2.RETR_CCOMP: 2レベルの階層に分類する。
                            物体の外側の輪郭を階層1、物体内側の穴などの輪郭を階層2として分類。
            cv2.RETR_TREE:  全階層情報を保持する。
        ApproximateMode
            輪郭の近似方法
            cv2.CHAIN_APPROX_NONE: 中間点も保持する。
            cv2.CHAIN_APPROX_SIMPLE: 中間点は保持しない。
        min_area : int
            領域が占める面積の閾値を指定
        """
    if org_img is None:  # 画像の元データが存在しない場合
        return None

    img_bin = org_img.copy()
    # 二値化処理
    # 画像のチャンネル数が2より大きい場合 ⇒ グレー画像に変換
    if len(img_bin.shape) > 2:
        img_bin = cv2.cvtColor(img_bin, cv2.COLOR_BGR2GRAY)

    """
        モルフォロジー変換
        主に二値画像を対象とし、画像上に写っている図形に対して作用するシンプルな処理を指します。
        クロージング処理
        クロージング処理はオープニング処理の逆の処理を指し、膨張の後に収縮 をする処理。
        前景領域中の小さな(黒い)穴を埋めるのに役立ちます。
        """
    # フィルタの設定
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    # クロージング処理
    img_morphing = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel)

    # 輪郭検出（Detection contours）
    tmp_img, contours, _ = cv2.findContours(
        img_morphing, RetrievalMode, ApproximateMode
    )

    # 輪郭近似（Contour approximation）
    approx = approx_contour(contours)

    # 等高線の描画（Contour line drawing）
    cp_org_img_for_draw = np.copy(org_img)
    drawing_edge(cp_org_img_for_draw, approx, min_area)

    return contours, cp_org_img_for_draw


def CenterOfGravity(org_img, contours, cal_Method=0):
    """
    オブジェクトの重心を計算する関数

    Parameter
    ---------
    org_img : OpenCV型
        重心計算方法用の画像
    contours : OpenCV型
        画像から抽出した輪郭情報
    cal_Method : int
        重心計算を行う方法を選択する
        0: 画像から重心を計算
        1: オブジェクトの輪郭から重心を計算

    Returns
    -------
    cx, cy : int
        オブジェクトの重心座標
    """
    # 計算方法が不正の場合
    if cal_Method > 1:
        cal_Method = 1

    img_bin = org_img.copy()
    # 二値化処理
    # 画像のチャンネル数が2より大きい場合 ⇒ グレー画像に変換
    if len(img_bin.shape) > 2:
        img_bin = cv2.cvtColor(img_bin, cv2.COLOR_BGR2GRAY)

    # 画像をもとに重心を求める場合
    if cal_Method == 0:
        M = cv2.moments(img_bin, False)

    # 輪郭から重心を求める場合
    else:
        maxCont = contours[0]
        for c in contours:
            if len(maxCont) < len(c):
                maxCont = c

        M = cv2.moments(maxCont)
    if int(M["m00"]) == 0:
        return None

    try:
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    except ZeroDivisionError:
        return None

    return (cx, cy)


def approx_contour(contours):
    """
    輪郭線の直線近似を行う関数

    Parameter
    ---------
    contours : OpenCV型
        画像から抽出した輪郭情報

    Return
    ------
    approx : list
        近似した輪郭情報
    """
    approx = []
    for i in range(len(contours)):
        cnt = contours[i]
        epsilon = 0.001 * cv2.arcLength(cnt, True)  # 実際の輪郭と近似輪郭の最大距離を表し、近似の精度を表すパラメータ
        approx.append(cv2.approxPolyDP(cnt, epsilon, True))
    return approx


def drawing_edge(img, contours, min_area):
    """
    入力されたimgに抽出した輪郭線を描く関数

    Parameters
    ----------
    img : OpenCV型
        輪郭線を描く元データ
    contours : OpenCV型
        画像から抽出した輪郭情報
    min_area : int
        領域が占める面積の閾値を指定
    """
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    cv2.drawContours(img, large_contours, -1, color=(0, 255, 0), thickness=1)


def WebCamOption(device_name):
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

    device_num = None
    if device_name == "TOSHIBA_Web_Camera-HD":
        device_num = 0
    if device_name == "Logicool_HD_Webcam_C270":
        device_num = 1

    return device_num


def ThresholdTypeOption(Type_name):
    """
    閾値の処理方法を選択する。
    選択可能な処理方法名：
        1. BINARY
        2. BINARY_INV
        3. TRUNC
        4. TOZERO
        5. TOZERO_INV
    Parameter
    ---------
    Type_name : string
        閾値の処理方法名
    Return
    ------
    Type : int
        名前が一致する処理方法を返す。
    """
    Type = None
    if Type_name == "BINARY":
        Type = cv2.THRESH_BINARY
    elif Type_name == "BINARY_INV":
        Type = cv2.THRESH_BINARY_INV
    elif Type_name == "TRUNC":
        Type = cv2.THRESH_TRUNC
    elif Type_name == "TOZERO":
        Type = cv2.THRESH_TOZERO
    elif Type_name == "TOZERO_INV":
        Type = cv2.THRESH_TOZERO_INV

    return Type


def AdaptiveThresholdTypeOption(Type_name):
    """
    適応的処理の処理方法を選択する。
    選択可能な処理方法名：
        1. MEAN_C
            近傍領域の中央値を閾値とする．
        2. GAUSSIAN_C
            近傍領域の重み付け平均値を閾値とする。
            重みの値はGaussian分布になるように計算。
    Parameter
    ---------
    Type_name : string
        閾値の処理方法名
    Return
    ------
    method : int
        名前が一致する処理方法を返す。
    """
    method = None
    if Type_name == "MEAN_C":
        method = cv2.ADAPTIVE_THRESH_MEAN_C
    elif Type_name == "GAUSSIAN_C":
        method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C

    return method


def KernelShapeOption(Type_name):
    """
    モルフォロジー変換に用いるカーネルの形を選択する。
    選択可能な形：
        1. 矩形
        2. 楕円形
        3. 十字型

    Parameter
    ---------
    Type_name : string
        カーネルの形
    Return
    ------
    method : int
        名前が一致する形を返す。
    """
    shape = None
    if Type_name == "矩形":
        shape = cv2.MORPH_RECT
    elif Type_name == "楕円形":
        shape = cv2.MORPH_ELLIPSE
    elif Type_name == "十字型":
        shape = cv2.MORPH_CROSS

    return shape


def ContourRetrievalModeOption(Type_name):
    """
    輪郭の階層情報の保持方法を選択する。
    選択可能な形：
        1. 親子関係を無視する
        2. 最も外側の輪郭を検出する
        3. 2つの階層に分類する
        4. 全階層情報を保存する

    Parameter
    ---------
    Type_name : string
        保存方法名
    Return
    ------
    mode : int
        名前が一致する形を返す。
    """
    mode = None
    if Type_name == "親子関係を無視する":
        mode = cv2.RETR_LIST
    elif Type_name == "最外の輪郭を検出する":
        mode = cv2.RETR_EXTERNAL
    elif Type_name == "2つの階層に分類する":
        mode = cv2.RETR_CCOMP
    elif Type_name == "全階層情報を保持する":
        mode = cv2.RETR_TREE

    return mode


def ApproximateModeOption(Type_name):
    """
    輪郭の近似方法を選択する。
    選択可能な形：
        1. 中間点を保持する
        2. 中間点を保持しない

    Parameter
    ---------
    Type_name : string
        中間点の保持方法
    Return
    ------
    method : int
        名前が一致する形を返す。
    """
    method = None
    if Type_name == "中間点を保持する":
        method = cv2.CHAIN_APPROX_NONE
    elif Type_name == "中間点を保持しない":
        method = cv2.CHAIN_APPROX_SIMPLE

    return method


# ボタンを押したときのイベントとボタンが返す値を代入
# event, values = window.Read()

# CON_STR = Dobot()

if __name__ == "__main__":
    window = Dobot_APP()
    window.loop()


import sys, os
from typing import Dict, List, Tuple, Union

sys.path.append(".")
sys.path.append("..")
sys.path.append("../../")

import time

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import PySimpleGUI as sg
import skimage.io as io

from lib.DobotDLL import DobotDllType as dType
from lib.DobotFunction.Camera import (
    DeviceNameToNum,
    ImageCvt,
    WebCam_OnOff,
    scale_box,
    SnapshotCvt,
    Contours,
)
from lib.DobotFunction.Communication import (
    Connect_Disconnect,
    ClearAlAlarms,
    _OneAction,
    SetPoseAct,
    GripperAutoCtrl,
)
from lib.DobotFunction.VisualFeedback import VisualFeedback
from lib.config.config import cfg
from lib.utils.base_utils import WriteDataToNdjson

# from ..DobotDLL

# from PIL import Image

_Dobot_err = {
    0: "DobotAct_NoError",
    1: "DobotConnect_NotFound",
    2: "DobotConnect_Occupied",
    3: "DobotAct_Timeout",
    4: "Set LeftUp Setting Error",
    5: "Set RightDown Setting Error",
    6: "Set Retreat position Setting Error",
}

_WebCam_err = {
    0: "WebCam_Connect",
    1: "WebCam_Release",
    2: "WebCam_NotFound",
    3: "WebCam_GetImage",
    4: "WebCam_NotGetImage",
    5: "Image_ConvertCompleted",
    6: "ImageFile_NotFound",
    7: "Image_ChannelError",
    8: "COG_CalculationCompleted",
}

_CamList = {
    "0": {"cam_num": None, "cam_object": None},
    "1": {"cam_num": None, "cam_object": None},
}


class Dobot_APP:
    def __init__(self):
        # dll_path = cfg.DOBOT_DLL_DIR + os.sep + "DobotDll.dll"
        # self.api = cdll.LoadLibrary(dll_path)
        self.api = dType.load()  # Dobot 制御ライブラリの読み出し
        # self.api = dType.load()
        self.connection = False  # Dobotの接続状態
        self.InitPose = {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "r": 0.0,
        }
        self.CurrentPose = {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "r": 0.0,
            "joint1Angle": 0.0,
            "joint2Angle": 0.0,
            "joint3Angle": 0.0,
            "joint4Angle": 0.0,
        }
        self.RecordPose = {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "r": 0.0,
        }  # Dobotがオブジェクトを退避させる位置
        # カメラ座標系とロボット座標系とのキャリブレーション時の左上の位置座標
        self.Alignment_1 = {
            "x": None,
            "y": None,
        }
        # カメラ座標系とロボット座標系とのキャリブレーション時の右下の位置座標
        self.Alignment_2 = {"x": None, "y": None}
        # --- エンドエフェクタ --- #
        self.MotorON = False  # 吸引用モータを制御
        # --- エラーフラグ --- #
        self.act_err = 0  # State: 0, Err: -1
        self.preview = False  # プレビューを画面上に表示するフラグ．True: 表示する．
        self.preview_cam = None
        self.IMAGE_Org = None  # スナップショットのオリジナル画像(RGB)
        self.IMAGE_bin = None  # 二値画像
        # --- 画像プレビュー画面の初期値 --- #
        self.fig_agg = None  # 画像のヒストグラムを表示する用の変数
        self.Image_height = 240  # 画面上に表示する画像の高さ
        self.Image_width = 320  # 画面上に表示する画像の幅
        # --- GUIの初期化 ---#
        self.layout = self.Layout()
        self.Window = self.main()

    """
    ----------------------
    GUI Layout
    ----------------------
    """

    def Layout(self):
        Connect = [
            [
                sg.Button("Connect or Disconnect", key="-Connect-"),
                sg.Button("Clear Alarm", key="-Alarm-"),
            ],
            [
                sg.Text("ptpMoveMode", size=(10, 1)),
                sg.Combo(
                    ("JumpCoordinate", "MoveJCoordinate", "MoveLCoordinate"),
                    default_value="MoveJCoordinate",
                    size=(15, 1),
                    key="-MoveMode-",
                    readonly=True,
                ),
            ],
            [],
        ]

        EndEffector = [
            [sg.Button("SuctionCup ON/OFF", key="-SuctionCup-")],
            [sg.Button("Gripper Open/Close", key="-Gripper-")],
        ]

        # タスク
        Task = [
            [
                sg.Button("タスク実行", size=(9, 1), disabled=True, key="-Task-"),
                sg.InputCombo(
                    ("Task_1", "Task_2", "Task_3", "Task_4", "Task_5"),
                    default_value="Task_1",
                    size=(9, 1),
                    disabled=True,
                    key="-TaskNum-",
                    readonly=True,
                ),
            ],
            [
                sg.Text("Kp", size=(2, 1)),
                sg.InputText(
                    default_text="0.05",
                    size=(5, 1),
                    disabled=False,
                    key="-Kp-",
                    readonly=False,
                ),
                sg.Text("Ki", size=(2, 1)),
                sg.InputText(
                    default_text="0.01",
                    size=(5, 1),
                    disabled=False,
                    key="-Ki-",
                    readonly=False,
                ),
            ],
        ]

        save_config = [
            [
                # sg.Input(size=(30, 1), key="-save_cfg_path-", disabled=True),
                sg.FileSaveAs(
                    file_types=(("JSON", "*.json"),),
                    enable_events=True,
                    key="-save_cfg-",
                    initial_folder="/tmp",
                ),
            ],
            [
                # sg.Input(size=(30, 1), disabled=True),
                sg.FileBrowse(
                    button_text="Loadfile path of Config",
                    change_submits=True,
                    enable_events=True,
                    disabled=False,
                    key="-load_cfg_path-",
                ),
            ],
        ]

        GetPose = [
            [sg.Button("Get Pose", size=(7, 1), key="-GetPose-")],
            [
                sg.Text("J1", size=(2, 1)),
                sg.InputText(
                    default_text="",
                    size=(5, 1),
                    disabled=True,
                    key="-Get_JointPose1-",
                    readonly=True,
                ),
                sg.Text("X", size=(1, 1)),
                sg.InputText(
                    default_text="",
                    size=(5, 1),
                    disabled=True,
                    key="-Get_CoordinatePose_X-",
                    readonly=True,
                ),
            ],
            [
                sg.Text("J2", size=(2, 1)),
                sg.InputText(
                    default_text="",
                    size=(5, 1),
                    disabled=True,
                    key="-Get_JointPose2-",
                    readonly=True,
                ),
                sg.Text("Y", size=(1, 1)),
                sg.InputText(
                    default_text="",
                    size=(5, 1),
                    disabled=True,
                    key="-Get_CoordinatePose_Y-",
                    readonly=True,
                ),
            ],
            [
                sg.Text("J3", size=(2, 1)),
                sg.InputText(
                    default_text="",
                    size=(5, 1),
                    disabled=True,
                    key="-Get_JointPose3-",
                    readonly=True,
                ),
                sg.Text("Z", size=(1, 1)),
                sg.InputText(
                    default_text="",
                    size=(5, 1),
                    disabled=True,
                    key="-Get_CoordinatePose_Z-",
                    readonly=True,
                ),
            ],
            [
                sg.Text("J4", size=(2, 1)),
                sg.InputText(
                    default_text="",
                    size=(5, 1),
                    disabled=True,
                    key="-Get_JointPose4-",
                    readonly=True,
                ),
                sg.Text("R", size=(1, 1)),
                sg.InputText(
                    default_text="",
                    size=(5, 1),
                    disabled=True,
                    key="-Get_CoordinatePose_R-",
                    readonly=True,
                ),
            ],
        ]

        SetPose = [
            [
                sg.Button("Set pose", size=(7, 1), key="-SetJointPose-"),
                sg.Button("Set pose", size=(7, 1), key="-SetCoordinatePose-"),
            ],
            [
                sg.Text("J1", size=(2, 1)),
                sg.InputText("", size=(5, 1), key="-JointPose1-"),
                sg.Text("X", size=(1, 1)),
                sg.InputText("", size=(5, 1), key="-CoordinatePose_X-"),
            ],
            [
                sg.Text("J2", size=(2, 1)),
                sg.InputText("", size=(5, 1), key="-JointPose2-"),
                sg.Text("Y", size=(1, 1)),
                sg.InputText("", size=(5, 1), key="-CoordinatePose_Y-"),
            ],
            [
                sg.Text("J3", size=(2, 1)),
                sg.InputText("", size=(5, 1), key="-JointPose3-"),
                sg.Text("Z", size=(1, 1)),
                sg.InputText("", size=(5, 1), key="-CoordinatePose_Z-"),
            ],
            [
                sg.Text("J4", size=(2, 1)),
                sg.InputText("", size=(5, 1), key="-JointPose4-"),
                sg.Text("R", size=(1, 1)),
                sg.InputText("", size=(5, 1), key="-CoordinatePose_R-"),
            ],
        ]
        # ----------------------------------------
        # キャリブレーションセッティング部分
        # ----------------------------------------
        Alignment = [
            [
                sg.Button(
                    button_text="MoveToThePoint", size=(14, 1), key="-MoveToThePoint-"
                ),
                sg.Text(
                    "OffSet",
                    size=(5, 1),
                    pad=(0, 0),
                ),
                sg.InputText(
                    default_text="43",
                    size=(2, 1),
                    disabled=False,
                    key="-offset-",
                    pad=(0, 0),
                ),
                sg.Text(
                    "[mm]",
                    size=(5, 1),
                    pad=(0, 0),
                ),
            ],
            [
                sg.Button("Set InitPose", size=(8, 1), key="-InitPose-"),
                sg.Button("Set Retreat", size=(8, 1), key="-Record-"),
                sg.Button(button_text="Set LeftUp", size=(8, 1), key="-Set_x1-"),
                sg.Button(button_text="Set RightDown", size=(10, 1), key="-Set_x2-"),
            ],
            [
                sg.Text("XInit", size=(3, 1)),
                sg.InputText(
                    default_text="", size=(5, 1), disabled=False, key="-x_init-"
                ),
                sg.Text("X0", size=(2, 1)),
                sg.InputText(
                    default_text="", size=(5, 1), disabled=False, key="-Retreat_x-"
                ),
                sg.Text("X1", size=(2, 1)),
                sg.InputText(
                    default_text="", size=(5, 1), disabled=False, key="-Alignment_x1-"
                ),
                sg.Text("X2", size=(2, 1)),
                sg.InputText(
                    default_text="", size=(5, 1), disabled=False, key="-Alignment_x2-"
                ),
            ],
            [
                sg.Text("YInit", size=(3, 1)),
                sg.InputText(
                    default_text="", size=(5, 1), disabled=False, key="-y_init-"
                ),
                sg.Text("Y0", size=(2, 1)),
                sg.InputText(
                    default_text="", size=(5, 1), disabled=False, key="-Retreat_y-"
                ),
                sg.Text("Y1", size=(2, 1)),
                sg.InputText(
                    default_text="", size=(5, 1), disabled=False, key="-Alignment_y1-"
                ),
                sg.Text("Y2", size=(2, 1)),
                sg.InputText(
                    default_text="", size=(5, 1), disabled=False, key="-Alignment_y2-"
                ),
            ],
            [
                sg.Text("ZInit", size=(3, 1)),
                sg.InputText(
                    default_text="", size=(5, 1), disabled=False, key="-z_init-"
                ),
                sg.Text("Z0", size=(2, 1)),
                sg.InputText(
                    default_text="", size=(5, 1), disabled=False, key="-Retreat_z-"
                ),
            ],
            [
                sg.Text("RInit", size=(3, 1)),
                sg.InputText(
                    default_text="", size=(5, 1), disabled=False, key="-r_init-"
                ),
                sg.Text("R0", size=(2, 1)),
                sg.InputText(
                    default_text="", size=(5, 1), disabled=False, key="-Retreat_r-"
                ),
            ],
        ]

        WebCamConnect = [
            [
                sg.Radio(
                    "Camera",
                    "picture",
                    default=True,
                    size=(8, 1),
                    background_color="grey63",
                    key="-WebCamChoice-",
                    pad=(0, 0),
                    enable_events=True,
                ),
                sg.Radio(
                    "Image",
                    "picture",
                    default=False,
                    size=(8, 1),
                    background_color="grey63",
                    key="-ImgChoice-",
                    pad=(0, 0),
                    enable_events=True,
                ),
            ],
            # ----------------------------------------
            # カメラの設定およびプレビュー部分_1
            # ----------------------------------------
            [
                # メインカメラの選択
                sg.Radio(
                    text="0",
                    group_id="main_cam",
                    default=True,
                    disabled=True,
                    background_color="grey59",
                    text_color="grey1",
                    key="-main_cam_0-",
                ),
                # PCに接続されているカメラの選択
                sg.InputCombo(
                    (
                        "Logicool_HD_Webcam_C270",
                        "TOSHIBA_Web_Camera-HD",
                    ),
                    default_value="TOSHIBA_Web_Camera-HD",
                    disabled=False,
                    size=(15, 1),
                    key="-WebCam_Name_0-",
                    readonly=True,
                ),
                # 解像度の選択
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
                    default_value="1280x800",
                    disabled=False,
                    size=(11, 1),
                    key="-WebCam_FrameSize_0-",
                    readonly=True,
                ),
                # Web_Cameraの接続/解放
                sg.Button(
                    "WEB CAM 0 on/off",
                    disabled=False,
                    size=(15, 1),
                    key="-SetWebCam_0-",
                ),
                # カメラのプレビュー
                sg.Button(
                    "Preview Opened", disabled=True, size=(13, 1), key="-Preview-"
                ),
                # 静止画撮影
                # sg.Button("Snapshot 0", size=(8, 1), key="-Snapshot-"),
            ],
            # ----------------------------------------
            # カメラの設定およびプレビュー部分_2
            # ----------------------------------------
            [
                sg.Radio(
                    text="1",
                    group_id="main_cam",
                    default=False,
                    disabled=True,
                    background_color="grey59",
                    text_color="grey1",
                    key="-main_cam_1-",
                ),
                # PCに接続されているカメラの選択
                sg.InputCombo(
                    (
                        "Logicool_HD_Webcam_C270",
                        "TOSHIBA_Web_Camera-HD",
                    ),
                    default_value="TOSHIBA_Web_Camera-HD",
                    disabled=False,
                    size=(15, 1),
                    key="-WebCam_Name_1-",
                    readonly=True,
                ),
                # 解像度の選択
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
                    default_value="1280x800",
                    disabled=False,
                    size=(11, 1),
                    key="-WebCam_FrameSize_1-",
                    readonly=True,
                ),
                # Web_Cameraの接続/解放
                sg.Button(
                    "WEB CAM 1 on/off",
                    disabled=False,
                    size=(15, 1),
                    key="-SetWebCam_1-",
                ),
                # カメラのプレビュー
                # sg.Button("Preview Opened 1", size=(13, 1), key="-Preview_1-"),
                # 静止画撮影
                sg.Button("Snapshot", disabled=False, size=(8, 1), key="-Snapshot-"),
            ],
            # -------------------------------------
            # ファイルから画像を読みだす部分
            # -------------------------------------
            [
                sg.Input(size=(30, 1), disabled=True),
                sg.FileBrowse(
                    button_text="Image file choice",
                    change_submits=True,
                    enable_events=True,
                    disabled=True,
                    key="-IMAGE_path-",
                ),
            ],
            [
                # 画像のサイズ・チャンネル数
                sg.Text("width", size=(4, 1)),
                sg.InputText(
                    default_text="0",
                    size=(5, 1),
                    disabled=True,
                    justification="right",
                    key="-IMAGE_width-",
                ),
                sg.Text("height", size=(4, 1)),
                sg.InputText(
                    default_text="0",
                    size=(5, 1),
                    disabled=True,
                    justification="right",
                    key="-IMAGE_height-",
                ),
                sg.Text("channel", size=(6, 1)),
                sg.InputText(
                    default_text="0",
                    size=(5, 1),
                    disabled=True,
                    justification="right",
                    key="-IMAGE_channel-",
                ),
            ],
            # ----------------------------------------
            # 画像の色・濃度・フィルタリング 部分
            # ----------------------------------------
            [
                sg.Text("色空間", size=(5, 1)),
                sg.InputCombo(
                    (
                        "RGB",
                        "Gray",
                    ),
                    default_value="RGB",
                    size=(4, 1),
                    key="-Color_Space-",
                    readonly=True,
                ),
                sg.Text("濃度変換", size=(7, 1)),
                sg.InputCombo(
                    (
                        "なし",
                        "線形濃度変換",
                        "非線形濃度変換",  # ガンマ処理
                        "ヒストグラム平坦化",
                    ),
                    default_value="なし",
                    size=(18, 1),
                    key="-Color_Density-",
                    readonly=True,
                ),
                sg.Text("空間フィルタリング", size=(16, 1)),
                sg.InputCombo(
                    (
                        "なし",
                        "平均化",
                        "ガウシアン",
                        "メディアン",
                    ),
                    default_value="なし",
                    size=(10, 1),
                    key="-Color_Filtering-",
                    readonly=True,
                ),
            ],
        ]

        # 二値化処理
        Binary = [
            [
                sg.InputCombo(
                    ("なし", "Global", "Otsu", "Adaptive", "Two"),
                    default_value="なし",
                    size=(10, 1),
                    enable_events=True,
                    key="-Binarization-",
                    readonly=True,
                ),
                sg.InputCombo(
                    ("Mean", "Gaussian", "Wellner"),
                    default_value="Mean",
                    size=(12, 1),
                    disabled=True,
                    key="-AdaptiveThreshold_type-",
                    readonly=True,
                ),
            ],
            [
                sg.Text("Lower", size=(4, 1)),
                sg.Slider(
                    range=(0, 127),
                    default_value=10,
                    orientation="horizontal",
                    disabled=True,
                    size=(12, 12),
                    key="-LowerThreshold-",
                ),
                sg.Text("Block Size", size=(8, 1)),
                sg.InputText(
                    default_text="11",
                    size=(4, 1),
                    disabled=True,
                    justification="right",
                    key="-AdaptiveThreshold_BlockSize-",
                ),
            ],
            [
                sg.Text("Upper", size=(4, 1)),
                sg.Slider(
                    range=(128, 256),
                    default_value=138,
                    orientation="horizontal",
                    disabled=True,
                    size=(12, 12),
                    key="-UpperThreshold-",
                ),
                sg.Text("Constant", size=(8, 1)),
                sg.InputText(
                    default_text="2",
                    size=(4, 1),
                    disabled=True,
                    justification="right",
                    key="-AdaptiveThreshold_Constant-",
                ),
            ],
            [
                sg.Radio(
                    text="R",
                    group_id="color",
                    disabled=True,
                    background_color="grey59",
                    text_color="red",
                    key="-color_R-",
                ),
                sg.Radio(
                    text="G",
                    group_id="color",
                    disabled=True,
                    background_color="grey59",
                    text_color="green",
                    key="-color_G-",
                ),
                sg.Radio(
                    text="B",
                    group_id="color",
                    disabled=True,
                    background_color="grey59",
                    text_color="blue",
                    key="-color_B-",
                ),
                sg.Radio(
                    text="W",
                    group_id="color",
                    disabled=True,
                    background_color="grey59",
                    text_color="snow",
                    key="-color_W-",
                ),
                sg.Radio(
                    text="Bk",
                    group_id="color",
                    default=True,
                    disabled=True,
                    background_color="grey59",
                    text_color="grey1",
                    key="-color_Bk-",
                ),
            ],
        ]

        # 画像から物体の輪郭を切り出す関数の設定部分_GUI
        ContourExtractionSettings = [
            [
                sg.InputCombo(
                    (
                        "画像から重心を計算",
                        "輪郭から重心を計算",
                    ),
                    default_value="画像から重心を計算",
                    size=(20, 1),
                    enable_events=True,
                    key="-CalcCOGMode-",
                    readonly=True,
                )
            ],
            [
                sg.Button(button_text="Contours", size=(7, 1), key="-Contours-"),
            ],
            # 輪郭のモードを指定する
            [
                sg.Text("輪郭", size=(3, 1)),
                sg.InputCombo(
                    (
                        "親子関係を無視する",
                        "最外の輪郭を検出する",
                        "2つの階層に分類する",
                        "全階層情報を保持する",
                    ),
                    default_value="全階層情報を保持する",
                    size=(20, 1),
                    disabled=True,
                    key="-RetrievalMode-",
                    readonly=True,
                ),
            ],
            # 近似方法を指定する
            [
                sg.Text("近似方法", size=(7, 1)),
                sg.InputCombo(
                    (
                        "中間点を保持する",
                        "中間点を保持しない",
                    ),
                    default_value="中間点を保持する",
                    size=(18, 1),
                    disabled=True,
                    key="-ApproximateMode-",
                    readonly=True,
                ),
            ],
            [
                sg.Text("角度計算", size=(7, 1)),
                sg.Radio(
                    "None",
                    "orientation",
                    default=True,
                    disabled=True,
                    size=(4, 1),
                    background_color="grey63",
                    key="-Not_calc_ori-",
                    pad=(0, 0),
                ),
                sg.Radio(
                    "楕円",
                    "orientation",
                    default=False,
                    disabled=True,
                    size=(3, 1),
                    background_color="grey63",
                    key="-Calc_Ellipse-",
                    pad=(0, 0),
                ),
                sg.Radio(
                    "CNN",
                    "orientation",
                    default=False,
                    disabled=True,
                    size=(3, 1),
                    background_color="grey63",
                    key="-Calc_CNN-",
                    pad=(0, 0),
                ),
            ],
            [
                sg.Text("Gx", size=(2, 1)),
                sg.InputText(
                    default_text="0",
                    size=(5, 1),
                    justification="right",
                    key="-CenterOfGravity_x-",
                    readonly=True,
                ),
                sg.Text("Gy", size=(2, 1)),
                sg.InputText(
                    default_text="0",
                    size=(5, 1),
                    justification="right",
                    key="-CenterOfGravity_y-",
                    readonly=True,
                ),
                sg.Text("Angle", size=(4, 1)),
                sg.InputText(
                    default_text="0",
                    size=(5, 1),
                    justification="right",
                    key="-Angle-",
                    readonly=True,
                ),
            ],
        ]

        layout = [
            [
                sg.Col(Connect),
                sg.Col(EndEffector),
                sg.Col(Task),
            ],
            [sg.Frame(title="データの保存 / 呼び出し", layout=save_config)],
            [
                sg.Col(GetPose, size=(165, 140)),
                sg.Col(SetPose, size=(165, 140)),
                sg.Frame(title="キャリブレーション", layout=Alignment),
            ],
            [
                sg.Col(WebCamConnect),
            ],
            [
                sg.Frame(title="二値化処理", layout=Binary),
                sg.Frame(title="画像の重心計算", layout=ContourExtractionSettings),
            ],
            [
                sg.Image(
                    filename="",
                    size=(self.Image_width, self.Image_height),
                    key="-IMAGE-",
                ),
                sg.Canvas(size=(self.Image_width, self.Image_height), key="-CANVAS-"),
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
        # ---------------------------------------------
        # 操作時にウインドウが変化するイベント群
        # ---------------------------------------------
        if event == "-WebCamChoice-":
            # カメラ_1関連を有効化
            self.Window["-main_cam_0-"].update(disabled=False)
            self.Window["-WebCam_Name_0-"].update(disabled=False)
            self.Window["-WebCam_FrameSize_0-"].update(disabled=False)
            self.Window["-SetWebCam_0-"].update(disabled=False)
            # カメラ_2関連を有効化
            self.Window["-main_cam_1-"].update(disabled=False)
            self.Window["-WebCam_Name_1-"].update(disabled=False)
            self.Window["-WebCam_FrameSize_1-"].update(disabled=False)
            self.Window["-SetWebCam_1-"].update(disabled=False)
            # 画像ファイル読み出しボタンを無効化
            self.Window["-IMAGE_path-"].update(disabled=True)
        if event == "-ImgChoice-":
            # 画像ファイル読み出しボタンを有効化
            self.Window["-IMAGE_path-"].update(disabled=False)
            # カメラ_1関連を無効化
            self.Window["-main_cam_0-"].update(disabled=True)
            self.Window["-WebCam_Name_0-"].update(disabled=True)
            self.Window["-WebCam_FrameSize_0-"].update(disabled=True)
            self.Window["-SetWebCam_0-"].update(disabled=True)
            # カメラ_2関連を無効化
            self.Window["-main_cam_1-"].update(disabled=True)
            self.Window["-WebCam_Name_1-"].update(disabled=True)
            self.Window["-WebCam_FrameSize_1-"].update(disabled=True)
            self.Window["-SetWebCam_1-"].update(disabled=True)

        # ---------------------------------------------
        # 二値化処理画面部分の変更
        # ---------------------------------------------
        if event == "-Binarization-":
            if values["-Binarization-"] == "Global":
                self.Window["-Color_Space-"].update("Gray")
                self.Window["-LowerThreshold-"].update(disabled=False)

                self.Window["-UpperThreshold-"].update(disabled=True)
                self.Window["-AdaptiveThreshold_type-"].update(
                    disabled=True, readonly=True
                )
                self.Window["-AdaptiveThreshold_BlockSize-"].update(disabled=True)
                self.Window["-AdaptiveThreshold_Constant-"].update(disabled=True)
                self.Window["-color_R-"].update(disabled=True)
                self.Window["-color_G-"].update(disabled=True)
                self.Window["-color_B-"].update(disabled=True)
                self.Window["-color_W-"].update(disabled=True)
                self.Window["-color_Bk-"].update(disabled=True)

            elif values["-Binarization-"] == "Otsu":
                self.Window["-Color_Space-"].update("Gray")

                self.Window["-LowerThreshold-"].update(disabled=True)
                self.Window["-UpperThreshold-"].update(disabled=True)
                self.Window["-AdaptiveThreshold_type-"].update(
                    disabled=True, readonly=True
                )
                self.Window["-AdaptiveThreshold_BlockSize-"].update(disabled=True)
                self.Window["-AdaptiveThreshold_Constant-"].update(disabled=True)
                self.Window["-color_R-"].update(disabled=True)
                self.Window["-color_G-"].update(disabled=True)
                self.Window["-color_B-"].update(disabled=True)
                self.Window["-color_W-"].update(disabled=True)
                self.Window["-color_Bk-"].update(disabled=True)

            elif values["-Binarization-"] == "Adaptive":
                self.Window["-Color_Space-"].update("Gray")
                self.Window["-AdaptiveThreshold_type-"].update(
                    disabled=False, readonly=True
                )
                self.Window["-AdaptiveThreshold_BlockSize-"].update(disabled=False)
                self.Window["-AdaptiveThreshold_Constant-"].update(disabled=False)

                self.Window["-LowerThreshold-"].update(disabled=True)
                self.Window["-UpperThreshold-"].update(disabled=True)
                self.Window["-color_R-"].update(disabled=True)
                self.Window["-color_G-"].update(disabled=True)
                self.Window["-color_B-"].update(disabled=True)
                self.Window["-color_W-"].update(disabled=True)
                self.Window["-color_Bk-"].update(disabled=True)

            elif values["-Binarization-"] == "Two":
                self.Window["-Color_Space-"].update("RGB")
                # 各色選択ボタンを有効化
                self.Window["-color_R-"].update(disabled=False)
                self.Window["-color_G-"].update(disabled=False)
                self.Window["-color_B-"].update(disabled=False)
                self.Window["-color_W-"].update(disabled=False)
                self.Window["-color_Bk-"].update(disabled=False)
                self.Window["-LowerThreshold-"].update(disabled=False)
                self.Window["-UpperThreshold-"].update(disabled=False)

                self.Window["-AdaptiveThreshold_type-"].update(
                    disabled=True, readonly=True
                )
                self.Window["-AdaptiveThreshold_BlockSize-"].update(disabled=True)
                self.Window["-AdaptiveThreshold_Constant-"].update(disabled=True)

        # ---------------------------------------------
        # COG計算画面部分の変更
        # ---------------------------------------------
        if event == "-CalcCOGMode-":
            if values["-CalcCOGMode-"] == "画像から重心を計算":
                # 保持する輪郭情報選択ダイヤログを無効化
                self.Window["-RetrievalMode-"].update(disabled=True)
                # 輪郭情報の近似計算ダイヤログを無効化
                self.Window["-ApproximateMode-"].update(disabled=True)
                # 角度情報計算ボタンを無効化
                self.Window["-Not_calc_ori-"].update(True, disabled=True)
                self.Window["-Calc_Ellipse-"].update(disabled=True)
                self.Window["-Calc_CNN-"].update(disabled=True)

            elif values["-CalcCOGMode-"] == "輪郭から重心を計算":
                # 保持する輪郭情報選択ダイヤログを有効化
                self.Window["-RetrievalMode-"].update(disabled=False)
                # 輪郭情報の近似計算ダイヤログを有効化
                self.Window["-ApproximateMode-"].update(disabled=False)
                # 角度情報計算ボタンを有効化
                self.Window["-Not_calc_ori-"].update(disabled=False)
                self.Window["-Calc_Ellipse-"].update(disabled=False)
                self.Window["-Calc_CNN-"].update(disabled=False)

        # ---------------------------------------------
        # values を保存する
        # ---------------------------------------------
        if event == "-save_cfg-":
            WriteDataToNdjson(values, values["-save_cfg-"])
            sg.Popup("画面の情報を JSON ファイルに保存しました。", title="File saved")

        # ---------------------------------------------
        # Dobotの接続を行う
        # ---------------------------------------------
        if event == "-Connect-":
            # self.connection = self.Connect_Disconnect_click(self.connection, self.api)
            self.connection, err = Connect_Disconnect(
                self.connection,
                self.api,
            )

            if self.connection:
                # Task 実行ボタンを起動
                self.Window["-Task-"].update(disabled=False)
                self.Window["-TaskNum-"].update(disabled=False)
                # Dobotの現在の姿勢を画面上に表示
                self.current_pose = self.GetPose_UpdateWindow()
            else:
                # Task 実行ボタンを無効化
                self.Window["-Task-"].update(disabled=True)
                self.Window["-TaskNum-"].update(disabled=True)

        # ---------------------------------------------
        # Dobotのアラームを解消する
        # ---------------------------------------------
        if event == "-Alarm-":
            ClearAlAlarms(self.api)
            sg.popup("アラームを解消しました．")

        # ---------------------------------------------
        # グリッパを動作させる
        # ---------------------------------------------
        elif event == "-Gripper-":
            if self.connection:
                # グリッパを開く
                GripperAutoCtrl(self.api)

        # ---------------------------------------------
        # 現在の姿勢を取得し、画面上のに表示する
        # ---------------------------------------------
        elif event == "-GetPose-":
            if self.connection:
                self.GetPose_UpdateWindow()

        elif event == "-SetJointPose-":
            # 移動後の関節角度を指定
            joint_pose = [
                float(values["-JointPose1-"]),
                float(values["-JointPose2-"]),
                float(values["-JointPose3-"]),
                float(values["-JointPose4-"]),
            ]
            response = self.SetJointPose_click(joint_pose)
            print(response)

        # ---------------------------------------------
        # デカルト座標系で指定位置に動作させる
        # ---------------------------------------------
        elif event == "-SetCoordinatePose-":
            if self.connection:
                if (
                    (values["-CoordinatePose_X-"] == "")
                    and (values["-CoordinatePose_Y-"] == "")
                    and (values["-CoordinatePose_Z-"] == "")
                    and (values["-CoordinatePose_R-"] == "")
                ):  # 移動先が1つも入力場合
                    sg.popup("移動先が入力されていません。", title="入力不良")
                    self.Input_err = 1
                    return

                pose = self.GetPose_UpdateWindow()
                if values["-CoordinatePose_X-"] == "":
                    values["-CoordinatePose_X-"] = pose["x"]
                if values["-CoordinatePose_Y-"] == "":
                    values["-CoordinatePose_Y-"] = pose["y"]
                if values["-CoordinatePose_Z-"] == "":
                    values["--CoordinatePose_Z-"] = pose["z"]
                if values["-CoordinatePose_R-"] == "":
                    values["-CoordinatePose_R-"] = pose["r"]

                # 移動後の関節角度を指定
                pose["x"] = float(values["-CoordinatePose_X-"])
                pose["y"] = float(values["-CoordinatePose_Y-"])
                pose["z"] = float(values["-CoordinatePose_Z-"])
                pose["r"] = float(values["-CoordinatePose_R-"])

                SetPoseAct(self.api, pose=pose, ptpMoveMode=values["-MoveMode-"])
                time.sleep(2)
            return

        # ---------------------------------- #
        # Dobotの動作初期位置を設定する #
        # ---------------------------------- #
        elif event == "-InitPose-":
            if self.connection:
                self.GetPose_UpdateWindow()

                self.Window["-x_init-"].update(str(self.CurrentPose["x"]))
                self.Window["-y_init-"].update(str(self.CurrentPose["y"]))
                self.Window["-z_init-"].update(str(self.CurrentPose["z"]))
                self.Window["-r_init-"].update(str(self.CurrentPose["r"]))

                self.InitPose = self.CurrentPose.copy()  # 現在の姿勢を記録

        # ---------------------------------- #
        # Dobotの動作終了位置を設定する #
        # ---------------------------------- #
        elif event == "-Record-":
            if self.connection:
                self.GetPose_UpdateWindow()

                self.Window["-Retreat_x-"].update(str(self.CurrentPose["x"]))
                self.Window["-Retreat_y-"].update(str(self.CurrentPose["y"]))
                self.Window["-Retreat_z-"].update(str(self.CurrentPose["z"]))
                self.Window["-Retreat_r-"].update(str(self.CurrentPose["r"]))

                self.RecordPose = self.CurrentPose.copy()  # 現在の姿勢を記録

        # ----------------------------------------------------------- #
        # 画像とDobotの座標系の位置合わせ用変数_1をセットする #
        # ----------------------------------------------------------- #
        elif event == "-Set_x1-":
            if self.connection:
                self.GetPose_UpdateWindow()

                self.Alignment_1["x"] = self.CurrentPose["x"]
                self.Alignment_1["y"] = self.CurrentPose["y"]
                self.Window["-Alignment_x1-"].update(str(self.Alignment_1["x"]))
                self.Window["-Alignment_y1-"].update(str(self.Alignment_1["y"]))

        # -------------------------------------------------------------- #
        #  画像とDobotの座標系の位置合わせ用変数_2をセットする   #
        # -------------------------------------------------------------- #
        elif event == "-Set_x2-":
            if self.connection:
                self.GetPose_UpdateWindow()

                self.Alignment_2["x"] = self.CurrentPose["x"]
                self.Alignment_2["y"] = self.CurrentPose["y"]
                self.Window["-Alignment_x2-"].update(str(self.Alignment_2["x"]))
                self.Window["-Alignment_y2-"].update(str(self.Alignment_2["y"]))

        # ------------------------------------ #
        # WebCamを選択&接続するイベント #
        # ------------------------------------ #
        elif event == "-SetWebCam_0-":
            device_name = values["-WebCam_Name_0-"]
            res = self.WebCAMOnOffBtn(device_name, dict_num=0)
            if res == 1:
                self.Window["-main_cam_0-"].update(disabled=True)
            else:
                self.Window["-main_cam_0-"].update(disabled=False)

            if values["-main_cam_0-"] or values["-main_cam_1-"]:
                if (_CamList["0"]["cam_object"] is not None) or (
                    _CamList["1"]["cam_object"] is not None
                ):
                    self.Window["-Preview-"].update(disabled=False)
                    # self.Window["-Snapshot-"].update(disabled=False)
                else:
                    self.Window["-Preview-"].update(disabled=True)
                    # self.Window["-Snapshot-"].update(disabled=True)

        elif event == "-SetWebCam_1-":
            device_name = values["-WebCam_Name_1-"]
            res = self.WebCAMOnOffBtn(device_name, dict_num=1)
            if res == 1:
                self.Window["-main_cam_1-"].update(disabled=True)
            else:
                self.Window["-main_cam_1-"].update(disabled=False)

            if values["-main_cam_0-"] or values["-main_cam_1-"]:
                if (_CamList["0"]["cam_object"] is not None) or (
                    _CamList["1"]["cam_object"] is not None
                ):
                    self.Window["-Preview-"].update(disabled=False)
                    # self.Window["-Snapshot-"].update(disabled=False)
                else:
                    self.Window["-Preview-"].update(disabled=True)
                    # self.Window["-Snapshot-"].update(disabled=True)

        # ---------------------------------#
        # プレビューを表示するイベント #
        # --------------------------------  #
        elif event == "-Preview-":
            self.preview = False if self.preview else True

        # --------------------------------------- #
        # スナップショットを撮影するイベント #
        # --------------------------------------- #
        elif event == "-Snapshot-":
            cam = None

            # WebCam が選択されている場合
            if values["-WebCamChoice-"]:
                if values["-main_cam_0-"]:
                    cam = _CamList["0"]["cam_object"]
                elif values["-main_cam_1-"]:
                    cam = _CamList["1"]["cam_object"]

                # どのラジオボタンもアクティブでない場合 -> カメラが1つも接続されていない場合．
                if type(cam) == cv2.VideoCapture:
                    self.IMAGE_Org, self.IMAGE_bin = self.SnapshotBtn(cam, values)
                else:
                    sg.popup(_WebCam_err[2], title="カメラ接続エラー")
            # 画像単体への処理が選択されている場合
            elif values["-ImgChoice-"]:
                # 画像ファイルが存在しない場合
                if not os.path.exists(values["-IMAGE_path-"]):
                    sg.popup(_WebCam_err[6], title="画像読み出しエラー")
                    return
                else:
                    img = io.imread(values["-IMAGE_path-"])
                    self.IMAGE_Org, self.IMAGE_bin = self.ImgcvtBtn(img, values)
            else:
                sg.popup(
                    "画像が取得できません。デバイスが1つも接続されていないか，読み込む画像のパスが不正です．", title="画像撮影エラー"
                )
                return

        # ---------------------------------------------
        # COGを計算するイベント
        # ---------------------------------------------
        elif event == "-Contours-":
            # --------------------------------------------- #
            # カメラに映るオブジェクトの重心位置を計算 #
            # --------------------------------------------- #
            if values["-WebCamChoice-"]:
                cam = None
                if values["-main_cam_0-"]:
                    cam = _CamList["0"]["cam_object"]
                elif values["-main_cam_1-"]:
                    cam = _CamList["1"]["cam_object"]

                # どのラジオボタンもアクティブでない場合 -> カメラが1つも接続されていない場合．
                if type(cam) == cv2.VideoCapture:
                    # スナップショットを撮影する
                    dst_org, dst_bin = self.SnapshotBtn(cam, values)
                else:
                    sg.popup(_WebCam_err[2], title="カメラ接続エラー")
                    return

            # --------------------------------------------- #
            # 画像に対してオブジェクトの重心位置を計算 #
            # --------------------------------------------- #
            elif values["-ImgChoice-"]:
                # 画像ファイルが存在しない場合
                if not os.path.exists(values["-IMAGE_path-"]):
                    sg.popup(_WebCam_err[6], title="画像読み出しエラー")
                    return
                else:
                    img = io.imread(values["-IMAGE_path-"])
                    dst_org, dst_bin = self.ImgcvtBtn(img, values)

            self.ContoursBtn(dst_org, dst_bin, values)

        # ---------------------------------------------
        # タスクを実行するイベント
        # ---------------------------------------------
        elif event == "-Task-":
            if values["-TaskNum-"] == "Task_1":
                # -------------------------------------------------------------- #
                # Task 1                                                                        #
                # オブジェクトの重心位置に移動→掴む→退避 動作を実行する #
                # -------------------------------------------------------------- #
                # <<< カメラの接続判定 <<< #
                cam = None
                if values["-WebCamChoice-"]:
                    if values["-main_cam_0-"]:
                        cam = _CamList["0"]["cam_object"]
                    elif values["-main_cam_1-"]:
                        cam = _CamList["1"]["cam_object"]
                else:
                    sg.popup("ラジオボタン: Camera が選択されていません．")
                    return

                # どのラジオボタンもアクティブでない場合 -> カメラが1つも接続されていない場合．
                if cam is None:
                    sg.popup(_WebCam_err[2], title="カメラ接続エラー")
                    return

                # <<< キャリブレーション座標の存在判定 <<< #
                if (
                    (not values["-Alignment_x1-"] and not values["-Alignment_y1-"])
                    or (not values["-Alignment_x2-"] and not values["-Alignment_y2-"])
                    or (
                        not values["-x_init-"]
                        and not values["-y_init-"]
                        and not values["-z_init-"]
                        and not values["-r_init-"]
                    )
                    or (
                        not values["-Retreat_x-"]
                        and not values["-Retreat_y-"]
                        and not values["-Retreat_z-"]
                        and not values["-Retreat_r-"]
                    )
                ):
                    sg.popup("キャリブレーション座標 x1 および x2, \nもしくは 退避位置 がセットされていません。")
                    return

                self.Alignment_1["x"] = float(values["-Alignment_x1-"])
                self.Alignment_1["y"] = float(values["-Alignment_y1-"])
                self.Alignment_2["x"] = float(values["-Alignment_x2-"])
                self.Alignment_2["y"] = float(values["-Alignment_y2-"])

                self.InitPose["x"] = float(values["-x_init-"])
                self.InitPose["y"] = float(values["-y_init-"])
                self.InitPose["z"] = float(values["-z_init-"])
                self.InitPose["r"] = float(values["-r_init-"])

                self.RecordPose["x"] = float(values["-Retreat_x-"])
                self.RecordPose["y"] = float(values["-Retreat_y-"])
                self.RecordPose["z"] = float(values["-Retreat_z-"])
                self.RecordPose["r"] = float(values["-Retreat_r-"])

                # タスク実行
                self.Task1(cam, values)

            elif values["-TaskNum-"] == "Task_2":
                # -------------------------------------------------------------- #
                # Task 2                                                                        #
                # オブジェクトの重心位置に移動→姿勢推定→掴む→退避 動作を実行する #
                # -------------------------------------------------------------- #
                # <<< カメラの接続判定 <<< #
                cam = None
                if values["-WebCamChoice-"]:
                    if values["-main_cam_0-"]:
                        cam = _CamList["0"]["cam_object"]
                    elif values["-main_cam_1-"]:
                        cam = _CamList["1"]["cam_object"]
                else:
                    sg.popup("ラジオボタン: Camera が選択されていません．")
                    return

                # どのラジオボタンもアクティブでない場合 -> カメラが1つも接続されていない場合．
                if cam is None:
                    sg.popup(_WebCam_err[2], title="カメラ接続エラー")
                    return

                # <<< キャリブレーション座標の存在判定 <<< #
                if (
                    (not values["-Alignment_x1-"] and not values["-Alignment_y1-"])
                    or (not values["-Alignment_x2-"] and not values["-Alignment_y2-"])
                    or (
                        not values["-x_init-"]
                        and not values["-y_init-"]
                        and not values["-z_init-"]
                        and not values["-r_init-"]
                    )
                    or (
                        not values["-Retreat_x-"]
                        and not values["-Retreat_y-"]
                        and not values["-Retreat_z-"]
                        and not values["-Retreat_r-"]
                    )
                ):
                    sg.popup("キャリブレーション座標 x1 および x2, \nもしくは 退避位置 がセットされていません。")
                    return

                self.Alignment_1["x"] = float(values["-Alignment_x1-"])
                self.Alignment_1["y"] = float(values["-Alignment_y1-"])
                self.Alignment_2["x"] = float(values["-Alignment_x2-"])
                self.Alignment_2["y"] = float(values["-Alignment_y2-"])

                self.InitPose["x"] = float(values["-x_init-"])
                self.InitPose["y"] = float(values["-y_init-"])
                self.InitPose["z"] = float(values["-z_init-"])
                self.InitPose["r"] = float(values["-r_init-"])

                self.RecordPose["x"] = float(values["-Retreat_x-"])
                self.RecordPose["y"] = float(values["-Retreat_y-"])
                self.RecordPose["z"] = float(values["-Retreat_z-"])
                self.RecordPose["r"] = float(values["-Retreat_r-"])

                # タスク実行
                self.Task2(cam, values)

            elif values["-TaskNum-"] == "Task_3":
                # ------------------------------------------------#
                # Task 3                                                     #
                # オブジェクトの重心位置の真上まで VF で移動 #
                # ------------------------------------------------#
                # <<< カメラの接続判定 <<< #
                cam = None
                if values["-WebCamChoice-"]:
                    if values["-main_cam_0-"]:
                        cam = _CamList["0"]["cam_object"]
                    elif values["-main_cam_1-"]:
                        cam = _CamList["1"]["cam_object"]
                else:
                    sg.popup("ラジオボタン: Camera が選択されていません．")
                    return

                # どのラジオボタンもアクティブでない場合 -> カメラが1つも接続されていない場合．
                if cam is None:
                    sg.popup(_WebCam_err[2], title="カメラ接続エラー")
                    return

                # タスク実行
                self.Task3(cam, values)

            elif values["-TaskNum-"] == "Task_4":
                # ------------------------------------------------#
                # Task 4                                                     #
                # オブジェクトの重心位置の真上まで VF で移動 #
                # ------------------------------------------------#
                # <<< カメラの接続判定 <<< #
                cam = None
                if values["-WebCamChoice-"]:
                    if values["-main_cam_0-"]:
                        cam = _CamList["0"]["cam_object"]
                    elif values["-main_cam_1-"]:
                        cam = _CamList["1"]["cam_object"]
                else:
                    sg.popup("ラジオボタン: Camera が選択されていません．")
                    return

                # どのラジオボタンもアクティブでない場合 -> カメラが1つも接続されていない場合．
                if cam is None:
                    sg.popup(_WebCam_err[2], title="カメラ接続エラー")
                    return

                # オブジェクトの退避先が設定されていない場合
                if (
                    not values["-x_init-"]
                    and not values["-y_init-"]
                    and not values["-z_init-"]
                    and not values["-r_init-"]
                ) or (
                    not values["-Retreat_x-"]
                    and not values["-Retreat_y-"]
                    and not values["-Retreat_z-"]
                    and not values["-Retreat_r-"]
                ):
                    sg.popup("オブジェクトの初期位置もしくは退避位置が設定されていません。", title=_Dobot_err[6])
                    return

                self.InitPose["x"] = float(values["-x_init-"])
                self.InitPose["y"] = float(values["-y_init-"])
                self.InitPose["z"] = float(values["-z_init-"])
                self.InitPose["r"] = float(values["-r_init-"])

                self.RecordPose["x"] = float(values["-Retreat_x-"])
                self.RecordPose["y"] = float(values["-Retreat_y-"])
                self.RecordPose["z"] = float(values["-Retreat_z-"])
                self.RecordPose["r"] = float(values["-Retreat_r-"])

                # タスク実行
                self.Task4(cam, values)

            elif values["-TaskNum-"] == "Task_5":
                # <<< カメラの接続判定 <<< #
                main_cam, sub_cam = None, None
                # 2つのカメラが使用可能か判定
                if (
                    _CamList["0"]["cam_object"] is not None
                    and _CamList["1"]["cam_object"] is not None
                ):
                    if values["-main_cam_0-"]:
                        main_cam = _CamList["0"]["cam_object"]
                        sub_cam = _CamList["1"]["cam_object"]
                    elif values["-main_cam_1-"]:
                        main_cam = _CamList["1"]["cam_object"]
                        sub_cam = _CamList["0"]["cam_object"]
                else:
                    sg.popup("このタスクを実行するには，Webカメラが2つ必要です．", title=_WebCam_err[2])
                    return

                # <<< キャリブレーション座標の存在判定 <<< #
                if (
                    (not values["-Alignment_x1-"] and not values["-Alignment_y1-"])
                    or (not values["-Alignment_x2-"] and not values["-Alignment_y2-"])
                    or (
                        not values["-x_init-"]
                        and not values["-y_init-"]
                        and not values["-z_init-"]
                        and not values["-r_init-"]
                    )
                    or (
                        not values["-Retreat_x-"]
                        and not values["-Retreat_y-"]
                        and not values["-Retreat_z-"]
                        and not values["-Retreat_r-"]
                    )
                ):
                    sg.popup("キャリブレーション座標 x1 および x2, \nもしくは 退避位置 がセットされていません。")
                    return

                self.Alignment_1["x"] = float(values["-Alignment_x1-"])
                self.Alignment_1["y"] = float(values["-Alignment_y1-"])
                self.Alignment_2["x"] = float(values["-Alignment_x2-"])
                self.Alignment_2["y"] = float(values["-Alignment_y2-"])

                self.InitPose["x"] = float(values["-x_init-"])
                self.InitPose["y"] = float(values["-y_init-"])
                self.InitPose["z"] = float(values["-z_init-"])
                self.InitPose["r"] = float(values["-r_init-"])

                self.RecordPose["x"] = float(values["-Retreat_x-"])
                self.RecordPose["y"] = float(values["-Retreat_y-"])
                self.RecordPose["z"] = float(values["-Retreat_z-"])
                self.RecordPose["r"] = float(values["-Retreat_r-"])

                # タスク実行
                self.Task5(main_cam=main_cam, sub_cam=sub_cam, values=values)

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
            if event == "Quit":
                break
            elif event != "__TIMEOUT__":
                self.Event(event, values)

            if self.preview:
                # WebCam が選択されている場合
                if values["-WebCamChoice-"]:
                    if values["-main_cam_0-"]:
                        cam = _CamList["0"]["cam_object"]
                    elif values["-main_cam_1-"]:
                        cam = _CamList["1"]["cam_object"]
                else:
                    sg.popup("カメラを使用した機能が停止しました．")
                    self.preview = False
                    return

                # どのラジオボタンもアクティブでない場合 -> カメラが1つも接続されていない場合．
                if cam is None:
                    return
                self.SnapshotBtn(cam, values, drawing=True)

    """
    ----------------------
    Button Function
    ----------------------
    """

    def SetJointPose_click(self, pose: dict):
        """
        指定された作業座標系にアームの先端を移動させる関数
        関節座標系で移動

        Args:
            pose (dict):
                デカルト座標系もしくは関節座標系での移動先を示したリスト
                パラメータ数4個

        Returns:
            response (int):
                0 : 応答なし
                1 : 応答あり
        """
        return _OneAction(self.api, pose)

    def SetCoordinatePose_click(self, pose: dict, queue_index: int = 1):
        """デカルト座標系で指定された位置にアームの先端を移動させる関数

        Arg:
            pose(dict): デカルト座標系および関節座標系で指定された姿勢データ

        Return:
            response(int):
                0 : 応答あり
                1 : 応答なし
        """
        lastIndex = dType.SetPTPCmd(
            self.api,
            dType.PTPMode.PTPMOVJXYZMode,
            pose["x"],
            pose["y"],
            pose["z"],
            pose["r"],
            1,
        )[0]
        while lastIndex > dType.GetQueuedCmdCurrentIndex(api)[0]:
            pass
        return 0

    def GetPose_UpdateWindow(self) -> dict:
        """
        姿勢を取得し、ウインドウを更新する関数

        Return:
            CurrentPose (dict): Dobotの現在の姿勢
        """

        pose = dType.GetPose(self.api)  # 現在のDobotの位置と関節角度を取得
        for num, key in enumerate(self.CurrentPose.keys()):
            self.CurrentPose[key] = pose[num]

        self.Window["-Get_JointPose1-"].update(str(self.CurrentPose["joint1Angle"]))
        self.Window["-Get_JointPose2-"].update(str(self.CurrentPose["joint2Angle"]))
        self.Window["-Get_JointPose3-"].update(str(self.CurrentPose["joint3Angle"]))
        self.Window["-Get_JointPose4-"].update(str(self.CurrentPose["joint4Angle"]))
        self.Window["-Get_CoordinatePose_X-"].update(str(self.CurrentPose["x"]))
        self.Window["-Get_CoordinatePose_Y-"].update(str(self.CurrentPose["y"]))
        self.Window["-Get_CoordinatePose_Z-"].update(str(self.CurrentPose["z"]))
        self.Window["-Get_CoordinatePose_R-"].update(str(self.CurrentPose["r"]))

        return self.CurrentPose.copy()

    def WebCAMOnOffBtn(self, device_name: str, dict_num: int = 0) -> int:
        """カメラを接続，解除するための関数．

        Args:
            device_name (str): 接続したいカメラ名
            dict_num (int): _CamList の key 番号．指定した番号にカメラ情報を保存する．key 番号が存在しない場合エラー．

        Return:
            response (int): Web CAM に接続できたか．
                0: Connect
                1: Release
                2: Not found
        """
        # Webカメラの番号を取得する
        cam_num = DeviceNameToNum(device_name=device_name)
        # webカメラの番号が取得できなかった場合
        if cam_num is None:
            sg.popup("選択したデバイスは存在しません。", title="カメラ接続エラー")
            return 2  # Not found！

        # ----------------------------#
        # カメラを接続するイベント #
        # --------------------------- #
        # カメラを初めて接続する場合 -> カメラを新規接続
        # 接続したいカメラが接続していカメラと同じ場合 -> カメラを解放
        if (_CamList[str(dict_num)]["cam_num"] is None) or (
            _CamList[str(dict_num)]["cam_num"] == cam_num
        ):
            response, cam_obj = WebCam_OnOff(
                cam_num, cam=_CamList[str(dict_num)]["cam_object"]
            )

        # 接続したいカメラと接続しているカメラが違う場合 -> 接続しているカメラを解放し、新規接続
        elif _CamList[str(dict_num)]["cam_num"] is not None:
            # まず接続しているカメラを開放する．
            response, cam_obj = WebCam_OnOff(
                device_num=_CamList[str(dict_num)]["cam_num"],
                cam=_CamList[str(dict_num)]["cam_object"],
            )
            # 開放できた場合
            if response == 1:
                sg.popup(_WebCam_err[response], title="Camの接続")
                # 次に新しいカメラを接続する．
                response, cam_obj = WebCam_OnOff(cam_num, cam=cam_obj)

        if response == 0:  # Connect！
            _CamList[str(dict_num)]["cam_num"] = cam_num
            _CamList[str(dict_num)]["cam_object"] = cam_obj
            # 1 回でもエラーが発生した場合，カメラパラメタをリセットする．
        else:  # Release or Not found
            _CamList[str(dict_num)]["cam_num"] = None
            _CamList[str(dict_num)]["cam_object"] = None

        # カメラの接続状況を表示
        sg.popup(_WebCam_err[response], title="Camの接続")

        return response

    def SnapshotBtn(
        self, cam: cv2.VideoCapture, values: list, drawing: bool = True
    ) -> np.ndarray:
        """スナップショットの撮影から一連の画像処理を行う関数。

        Args:
            cam(cv2.VideoCapture): 接続しているカメラ情報
            values (list): ウインドウ上のボタンの状態などを記録している変数
            drawing (bool, optional): 画面上に画面を描画するか．Default to True.
        Returns:
            dst_org (np.ndarray): オリジナルのスナップショット
            dst_bin (np.ndarray): 二値化処理後の画像
        """
        dst_org = img = None
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

        err, dst_org, img = SnapshotCvt(
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

        if err != 5:
            sg.popup("画像処理エラー")
            return None, None  # None, None

        # --------------------------- #
        # 画像サイズなどをダイアログ上に表示 #
        # --------------------------- #
        c = 1
        if len(img.shape) == 2:
            h, w = img.shape
        elif len(img.shape) == 3:
            h, w, c = img.shape
        else:
            ValueError("Image size is incorrect.")
        self.Window["-IMAGE_width-"].update(str(w))
        self.Window["-IMAGE_height-"].update(str(h))
        self.Window["-IMAGE_channel-"].update(str(c))

        if drawing:
            self.ImageDrawingWindow(img)

        return dst_org, img

    def ImgcvtBtn(
        self, img: np.ndarray, values: list, drawing: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """1枚の画像に対して一連の画像処理を行う関数。

        Args:
            img(np.ndarray): 処理対象の画像
            values (list): ウインドウ上のボタンの状態などを記録している変数
            drawing (bool, optional): 画面上に画面を描画するか．Default to True.
        Returns:
            dst_org (np.ndarray): 入力した画像．
            dst_bin (np.ndarray): 処理後の画像．
        """
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

        err, dst_org, dst_bin = ImageCvt(
            img,
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

        if err != 5:
            sg.popup("画像処理エラー")
            return None, None  # None, None

        # --------------------------- #
        # 画像サイズなどをダイアログ上に表示 #
        # --------------------------- #
        c = 1
        if len(dst_bin.shape) == 2:
            h, w = dst_bin.shape
        elif len(dst_bin.shape) == 3:
            h, w, c = dst_bin.shape
        else:
            ValueError("Image size is incorrect.")
        self.Window["-IMAGE_width-"].update(str(w))
        self.Window["-IMAGE_height-"].update(str(h))
        self.Window["-IMAGE_channel-"].update(str(c))

        if drawing:
            self.ImageDrawingWindow(dst_bin)

        return dst_org, dst_bin

    def ContoursBtn(
        self,
        rgb_img: np.ndarray,
        bin_img: np.ndarray,
        values: list,
        drawing: bool = True,
    ) -> Tuple[int, List[float]]:
        """スナップショットの撮影からオブジェクトの重心位置計算までの一連の画像処理を行う関数。

        Args:
            rgb_img(np.ndarray): RGB画像
            bin_img(np.ndarray): 二値画像
            values (list): ウインドウ上のボタンの状態などを記録している変数
        Returns:
            err (int): エラー番号．7: 入力画像のチャンネルエラー, 8: COGの計算が正常に終了
            COG (List[float]): COG=[x, y], オブジェクトの重心位置
        """

        if len(bin_img.shape) == 2 and values["-Binarization-"] != "なし":
            COG, dst_rgb = Contours(
                rgb_img=rgb_img.copy(),
                bin_img=bin_img.copy(),
                CalcCOG=str(values["-CalcCOGMode-"]),
                Retrieval=str(values["-RetrievalMode-"]),
                Approximate=str(values["-ApproximateMode-"]),
                orientation=values["-Calc_Ellipse-"],
                drawing_figure=False,
            )
        else:
            sg.popup(_WebCam_err[7], title="チャネルエラー")
            return 7, []

        if drawing:
            self.ImageDrawingWindow(dst_rgb)

        if COG:
            self.Window["-CenterOfGravity_x-"].update(str(COG[0]))
            self.Window["-CenterOfGravity_y-"].update(str(COG[1]))
            self.Window["-Angle-"].update(str(COG[2]))
        return 8, COG

    def ImageDrawingWindow(self, img: np.ndarray) -> None:
        """画面上に撮影した画像を表示する関数

        Args:
            img (np.ndarray): 画面に表示させたい画面
        """
        src = img.copy()
        src = scale_box(src, self.Image_width, self.Image_height)
        # エンコード用に画像のチェネルを B, G, R の順番に変更
        src = cv2.cvtColor(src, cv2.COLOR_RGB2BGR)
        imgbytes = cv2.imencode(".png", src)[1].tobytes()
        self.Window["-IMAGE-"].update(data=imgbytes)

        # --------------------------------- #
        # 画面上に撮影した画像のヒストグラムを表示する #
        # --------------------------------- #
        canvas_elem = self.Window["-CANVAS-"]
        canvas = canvas_elem.TKCanvas
        ticks = [0, 42, 84, 127, 169, 211, 255]
        fig, ax = plt.subplots(figsize=(3, 2))
        ax = Image_hist(src, ax, ticks)
        if self.fig_agg:
            # ** IMPORTANT ** Clean up previous drawing before drawing again
            delete_figure_agg(self.fig_agg)
        self.fig_agg = draw_figure(canvas, fig)
        self.fig_agg.draw()

    def OffSet(self, values: list) -> Union[Dict[str, float], None]:
        """
        内視鏡の中心位置にエンドエフェクタの中心位置を移動させる際の、(X, Y) 方向の距離を計算する関数．

        Args:
            values (list): ウインドウ上のボタンの状態などを記録している変数．

        Returns:
            Union[Dict[str, float], None]: ロボット座標系における (X, Y) 方向の移動距離
            * No Error: {"x": float, "y": float}
            * Error: {"x": None, "y": None}
        """
        offset = int(values["-offset-"])
        j1 = self.CurrentPose["joint1Angle"]
        if offset > 0:
            return_param = {"x": None, "y": None}
            rob_x = offset * np.cos(np.radians(j1))
            rob_y = offset * np.sin(np.radians(j1))

            return_param["x"] = rob_x
            return_param["y"] = rob_y
            return return_param
        else:
            return None

    def Task1(self, cam: cv2.VideoCapture, values: list):
        """
        Task1 実行関数。
        キャリブレーションありの状態
        オブジェクトの重心位置に移動→掴む→退避 動作を実行する

        Args:
            cam(cv2.VideoCapture): 接続しているカメラ情報
            values (list): ウインドウ上のボタンの状態などを記録している変数
        """
        if self.connection and (type(cam) == cv2.VideoCapture):
            dst_org, dst_bin = self.SnapshotBtn(cam, values, drawing=False)
            # 画像を撮影 & 重心位置を計算
            err, COG = self.ContoursBtn(dst_org, dst_bin, values, drawing=False)

            if err == 7:
                return
            # 最終的に戻ってくる初期位置を保持
            init_pose = self.InitPose
            # 現在のDobotの姿勢を取得
            pose = self.GetPose_UpdateWindow()  # pose -> self.CurrentPose
            # ------------------------------ #
            # Dobotの移動後の姿勢を計算 #
            # ------------------------------ #
            if len(dst_bin.shape) == 2:
                h, w = dst_bin.shape
            elif len(dst_bin.shape) == 3:
                h, w, c = dst_bin.shape
            try:
                pose["x"] = (
                    self.Alignment_1["x"]
                    + COG[1] * (self.Alignment_2["x"] - self.Alignment_1["x"]) / h
                )
                pose["y"] = (
                    self.Alignment_1["y"]
                    + COG[0] * (self.Alignment_2["y"] - self.Alignment_1["y"]) / w
                )
            except ZeroDivisionError:  # ゼロ割が発生した場合
                sg.popup("画像のサイズが計測されていません", title="エラー")
                return

            # Dobotをオブジェクト重心の真上まで移動させる。
            SetPoseAct(self.api, pose=pose, ptpMoveMode=values["-MoveMode-"])
            # グリッパーを開く。
            GripperAutoCtrl(self.api)
            # DobotをZ=-35の位置まで降下させる。
            pose["z"] = self.RecordPose["z"].copy()
            SetPoseAct(self.api, pose=pose, ptpMoveMode=values["-MoveMode-"])
            # グリッパを閉じる。
            GripperAutoCtrl(self.api)
            # DobotをZ=20の位置まで上昇させる。
            # pose["z"] = self.CurrentPose["z"]
            pose["z"] = 20
            SetPoseAct(self.api, pose=pose, ptpMoveMode=values["-MoveMode-"])
            # 退避位置まで移動させる。
            pose = self.RecordPose.copy()
            # DobotをZ=20の位置まで上昇させる。
            pose["z"] = 20
            SetPoseAct(
                self.api,
                pose=pose,
                ptpMoveMode=values["-MoveMode-"],
            )
            # DobotをZ=-35の位置まで降下させる。
            pose["z"] = self.RecordPose["z"].copy()
            SetPoseAct(self.api, pose=pose, ptpMoveMode=values["-MoveMode-"])
            # グリッパを開く．
            GripperAutoCtrl(self.api)
            # DobotをZ=20の位置まで上昇させる。
            pose["z"] = 20
            SetPoseAct(self.api, pose=pose, ptpMoveMode=values["-MoveMode-"])
            # グリッパを閉じる．
            GripperAutoCtrl(self.api)
            # グリッパを初期位置まで移動させる．
            SetPoseAct(self.api, pose=init_pose, ptpMoveMode=values["-MoveMode-"])

        else:
            sg.popup("Dobotかカメラが接続されていません。")
            return

    def Task2(self, cam: cv2.VideoCapture, values: list):
        """
        Task2 実行関数。
        キャリブレーションありの状態
        オブジェクトの重心位置に移動→姿勢推定→掴む→退避 動作を実行する

        Args:
            cam(cv2.VideoCapture): 接続しているカメラ情報
            values (list): ウインドウ上のボタンの状態などを記録している変数
        """
        if self.connection and (type(cam) == cv2.VideoCapture):
            dst_org, dst_bin = self.SnapshotBtn(cam, values, drawing=False)
            # 画像を撮影 & 重心位置を計算
            err, COG = self.ContoursBtn(dst_org, dst_bin, values, drawing=False)

            if err == 7:
                return
            # 最終的に戻ってくる初期位置を保持
            init_pose = self.InitPose
            # 現在のDobotの姿勢を取得
            pose = self.GetPose_UpdateWindow()  # pose -> self.CurrentPose
            # ------------------------------ #
            # Dobotの移動後の姿勢を計算 #
            # ------------------------------ #
            if len(dst_bin.shape) == 2:
                h, w = dst_bin.shape
            elif len(dst_bin.shape) == 3:
                h, w, _ = dst_bin.shape
            try:
                pose["x"] = (
                    self.Alignment_1["x"]
                    + COG[1] * (self.Alignment_2["x"] - self.Alignment_1["x"]) / h
                )
                pose["y"] = (
                    self.Alignment_1["y"]
                    + COG[0] * (self.Alignment_2["y"] - self.Alignment_1["y"]) / w
                )
            except ZeroDivisionError:  # ゼロ割が発生した場合
                sg.popup("画像のサイズが計測されていません", title="エラー")
                return

            # Dobotをオブジェクト重心の真上まで移動させる。
            SetPoseAct(self.api, pose=pose, ptpMoveMode=values["-MoveMode-"])
            # エンドエフェクタを推定した角度に回転する．
            if COG[2] is not None:
                pose["r"] = COG[2] - 90
                SetPoseAct(self.api, pose=pose, ptpMoveMode=values["-MoveMode-"])
            # グリッパーを開く。
            GripperAutoCtrl(self.api)
            # DobotをZ=-35の位置まで降下させる。
            pose["z"] = self.RecordPose["z"].copy()
            SetPoseAct(self.api, pose=pose, ptpMoveMode=values["-MoveMode-"])
            # グリッパを閉じる。
            GripperAutoCtrl(self.api)
            # DobotをZ=20の位置まで上昇させる。
            # pose["z"] = self.CurrentPose["z"]
            pose["z"] = 20
            SetPoseAct(self.api, pose=pose, ptpMoveMode=values["-MoveMode-"])
            # 退避位置まで移動させる。
            pose = self.RecordPose.copy()
            # DobotをZ=20の位置まで上昇させる。
            pose["z"] = 20
            SetPoseAct(
                self.api,
                pose=pose,
                ptpMoveMode=values["-MoveMode-"],
            )
            # DobotをZ=-35の位置まで降下させる。
            pose["z"] = self.RecordPose["z"].copy()
            SetPoseAct(self.api, pose=pose, ptpMoveMode=values["-MoveMode-"])
            # グリッパを開く．
            GripperAutoCtrl(self.api)
            # DobotをZ=20の位置まで上昇させる。
            pose["z"] = 20
            SetPoseAct(self.api, pose=pose, ptpMoveMode=values["-MoveMode-"])
            # グリッパを閉じる．
            GripperAutoCtrl(self.api)
            # グリッパを初期位置まで移動させる．
            SetPoseAct(self.api, pose=init_pose, ptpMoveMode=values["-MoveMode-"])

        else:
            sg.popup("Dobotかカメラが接続されていません。")
            return

    def Task3(self, cam: cv2.VideoCapture, values: list):
        """
        オブジェクトの重心位置の真上まで VF で移動

        Args:
            cam (cv2.VideoCapture): 接続しているカメラ情報
            values (list): ウインドウ上のボタンの状態などを記録している変数
        """
        if (
            self.connection
            and (type(cam) == cv2.VideoCapture)
            and (values["-Binarization-"] != "なし")
        ):
            try:
                vf = VisualFeedback(self.api, cam, values)
            except Exception as e:
                sg.popup(e, title="VF コントロールエラー")
                return
            else:
                data = vf.run(target="vf")

            if (
                data is not None
                and data["pose"] is not None
                and data["COG"] is not None
            ):
                self.current_pose = data["pose"]
                self.Window["-CenterOfGravity_x-"].update(str(data["COG"][0]))
                self.Window["-CenterOfGravity_y-"].update(str(data["COG"][1]))
            else:
                sg.popup("VF の戻り値が正常に取得できませんでした．")
                return
        else:
            sg.popup("Dobotかカメラが接続されていない。もしくは，画像が二値化されていません．")
            return

    def Task4(self, cam: cv2.VideoCapture, values: list):
        """
        オブジェクトの重心位置の真上まで VF で移動

        Args:
            cam (cv2.VideoCapture): 接続しているカメラ情報
            values (list): ウインドウ上のボタンの状態などを記録している変数
        """
        if (
            self.connection
            and (type(cam) == cv2.VideoCapture)
            and (values["-Binarization-"] != "なし")
            and (self.RecordPose)
        ):
            try:
                vf = VisualFeedback(self.api, cam, values)
            except Exception as e:
                sg.popup(e, title="VF コントロールエラー")
                return
            else:
                data = vf.run(target="vf")

            if (
                data is not None
                and data["pose"] is not None
                and data["COG"] is not None
            ):
                # 最終的に戻ってくる初期位置を保持
                init_pose = self.InitPose
                # 現在のDobotの姿勢を取得
                pose = self.GetPose_UpdateWindow()  # pose -> self.CurrentPose
                self.Window["-CenterOfGravity_x-"].update(str(data["COG"][0]))
                self.Window["-CenterOfGravity_y-"].update(str(data["COG"][1]))
                self.Window["-Angle-"].update(str(data["COG"][2]))
            else:
                sg.popup("VF の戻り値が正常に取得できませんでした．")
                return

            # Dobotをオブジェクト重心の真上まで移動させる。
            offset = self.OffSet(values)
            if offset is not None:
                pose["x"] += offset["x"]
                pose["y"] += offset["y"]
            SetPoseAct(self.api, pose=pose, ptpMoveMode=values["-MoveMode-"])
            # エンドエフェクタを推定した角度に回転する．
            if data["COG"][2] is not None:
                pose["r"] = data["COG"][2] - 90
                SetPoseAct(self.api, pose=pose, ptpMoveMode=values["-MoveMode-"])
            # グリッパーを開く。
            GripperAutoCtrl(self.api)
            # DobotをZ=-35の位置まで降下させる。
            pose["z"] = self.RecordPose["z"].copy()
            SetPoseAct(self.api, pose=pose, ptpMoveMode=values["-MoveMode-"])
            # グリッパを閉じる。
            GripperAutoCtrl(self.api)
            # DobotをZ=20の位置まで上昇させる。
            # pose["z"] = self.CurrentPose["z"]
            pose["z"] = 20
            SetPoseAct(self.api, pose=pose, ptpMoveMode=values["-MoveMode-"])
            # 退避位置まで移動させる。
            pose = self.RecordPose.copy()
            # DobotをZ=20の位置まで上昇させる。
            pose["z"] = 20
            SetPoseAct(
                self.api,
                pose=pose,
                ptpMoveMode=values["-MoveMode-"],
            )
            # DobotをZ=-35の位置まで降下させる。
            pose["z"] = self.RecordPose["z"].copy()
            SetPoseAct(self.api, pose=pose, ptpMoveMode=values["-MoveMode-"])
            # グリッパを開く．
            GripperAutoCtrl(self.api)
            # DobotをZ=20の位置まで上昇させる。
            pose["z"] = 20
            SetPoseAct(self.api, pose=pose, ptpMoveMode=values["-MoveMode-"])
            # グリッパを閉じる．
            GripperAutoCtrl(self.api)
            # グリッパを初期位置まで移動させる．
            SetPoseAct(self.api, pose=init_pose, ptpMoveMode=values["-MoveMode-"])

        else:
            sg.popup("Dobotかカメラが接続されていない。もしくは，画像が二値化されていません．")
            return

    def Task5(
        self, main_cam: cv2.VideoCapture, sub_cam: cv2.VideoCapture, values: list
    ):
        if (
            self.connection
            and (type(main_cam) == cv2.VideoCapture)
            and (type(sub_cam) == cv2.VideoCapture)
            and (values["-Binarization-"] != "なし")
            and (self.RecordPose)
        ):
            dst_org = dst_bin = None
            COG = []
            # 最終的に戻ってくる初期位置を保持
            # init_pose = self.GetPose_UpdateWindow()  # pose -> self.CurrentPose
            init_pose = self.InitPose
            # Dobotを初期位置まで移動させる。
            SetPoseAct(self.api, pose=init_pose, ptpMoveMode=values["-MoveMode-"])

            # VF Class のインスタンスを作成
            try:
                vf = VisualFeedback(self.api, sub_cam, values)
            except Exception as e:
                sg.popup(e, title="VF コントロールエラー")
                return

            dst_org, dst_bin = self.SnapshotBtn(main_cam, values, drawing=False)
            # 画像を撮影 & 重心位置を計算
            err, COG = self.ContoursBtn(dst_org, dst_bin, values, drawing=False)

            if err == 7 or not COG:
                return

            # 現在のDobotの姿勢を取得
            pose = self.GetPose_UpdateWindow()  # pose -> self.CurrentPose
            # ------------------------------ #
            # Dobotの移動後の姿勢を計算 #
            # ------------------------------ #
            if len(dst_bin.shape) == 2:
                h, w = dst_bin.shape
            elif len(dst_bin.shape) == 3:
                h, w, _ = dst_bin.shape
            try:
                pose["x"] = (
                    self.Alignment_1["x"]
                    + COG[1] * (self.Alignment_2["x"] - self.Alignment_1["x"]) / h
                )
                pose["y"] = (
                    self.Alignment_1["y"]
                    + COG[0] * (self.Alignment_2["y"] - self.Alignment_1["y"]) / w
                )
            except ZeroDivisionError:  # ゼロ割が発生した場合
                sg.popup("画像のサイズが計測されていません", title="エラー")
                return

            # Dobotをオブジェクト重心の真上まで移動させる。
            SetPoseAct(self.api, pose=pose, ptpMoveMode=values["-MoveMode-"])
            time.sleep(2)
            data = vf.run(target="vf")

            if (
                data is not None
                and data["pose"] is not None
                and data["COG"] is not None
            ):
                # 最終的に戻ってくる初期位置を保持
                # init_pose = self.GetPose_UpdateWindow()  # pose -> self.CurrentPose
                # 現在のDobotの姿勢を取得
                pose = self.GetPose_UpdateWindow()  # pose -> self.CurrentPose
                self.Window["-CenterOfGravity_x-"].update(str(data["COG"][0]))
                self.Window["-CenterOfGravity_y-"].update(str(data["COG"][1]))
                self.Window["-Angle-"].update(str(data["COG"][2]))
                del vf
            else:
                sg.popup("VF の戻り値が正常に取得できませんでした．")
                return

            # Dobotをオブジェクト重心の真上まで移動させる。
            offset = self.OffSet(values)
            if offset is not None:
                pose["x"] += offset["x"]
                pose["y"] += offset["y"]
            SetPoseAct(self.api, pose=pose, ptpMoveMode=values["-MoveMode-"])
            # エンドエフェクタを推定した角度に回転する．
            if data["COG"][2] is not None:
                pose["r"] = data["COG"][2] - 90
                SetPoseAct(self.api, pose=pose, ptpMoveMode=values["-MoveMode-"])
            # グリッパーを開く。
            GripperAutoCtrl(self.api)
            # DobotをZ=-35の位置まで降下させる。
            pose["z"] = self.RecordPose["z"].copy()
            SetPoseAct(self.api, pose=pose, ptpMoveMode=values["-MoveMode-"])
            # グリッパを閉じる。
            GripperAutoCtrl(self.api)
            # DobotをZ=20の位置まで上昇させる。
            # pose["z"] = self.CurrentPose["z"]
            pose["z"] = 20
            SetPoseAct(self.api, pose=pose, ptpMoveMode=values["-MoveMode-"])
            # 退避位置まで移動させる。
            pose = self.RecordPose.copy()
            # DobotをZ=20の位置まで上昇させる。
            pose["z"] = 20
            SetPoseAct(
                self.api,
                pose=pose,
                ptpMoveMode=values["-MoveMode-"],
            )
            # DobotをZ=-35の位置まで降下させる。
            pose["z"] = self.RecordPose["z"].copy()
            SetPoseAct(self.api, pose=pose, ptpMoveMode=values["-MoveMode-"])
            # グリッパを開く．
            GripperAutoCtrl(self.api)
            # DobotをZ=20の位置まで上昇させる。
            pose["z"] = 20
            SetPoseAct(self.api, pose=pose, ptpMoveMode=values["-MoveMode-"])
            # グリッパを閉じる．
            GripperAutoCtrl(self.api)
            # グリッパを初期位置まで移動させる．
            SetPoseAct(self.api, pose=init_pose, ptpMoveMode=values["-MoveMode-"])

        else:
            sg.popup("Dobotかカメラが接続されていない。もしくは，画像が二値化されていません．")
            return


def WebCamOption(device_name: str) -> int:
    """
    接続するWebCameraを選択する関数

    Args:
        device_name (int):
            使用したいデバイス名を指定

    Return:
        device_num (int):
            名前が一致したデバイスに割り当てられた番号を返す
    """
    if device_name == "TOSHIBA_Web_Camera-HD":
        device_num = 0
    elif device_name == "Logicool_HD_Webcam_C270":
        device_num = 1
    else:
        device_num = None

    return device_num


def Image_hist(img, ax, ticks=None):
    """
    rgb_img と matplotlib.axes を受け取り、
    axes にRGBヒストグラムをplotして返す
    """
    if len(img.shape) == 2:
        color = ["k"]
    elif len(img.shape) == 3:
        color = ["r", "g", "b"]
    for (i, col) in enumerate(color):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        hist = np.sqrt(hist)
        ax.plot(hist, color=col)

    if ticks:
        ax.set_xticks(ticks)
    ax.set_title("histogram")
    ax.set_xlim([0, 256])

    return ax


def draw_figure(canvas, figure, loc=(0, 0)):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg


def delete_figure_agg(figure_agg):
    figure_agg.get_tk_widget().forget()
    plt.close("all")


if __name__ == "__main__":
    debug = False

    if debug:
        dll_path = os.path.join(cfg.DLL_DIR, "DobotDll.dll")
        # api = dType.load(dll_path)
        api = dType.load()

        connection_flag = False
        connection_flag, result = Connect_Disconnect(connection_flag, api)

        if connection_flag:
            pose = {
                "x": 193,
                "y": -20,
                "z": 21,
                "r": 46,
                "joint1Angle": 0.0,
                "joint2Angle": 0.0,
                "joint3Angle": 0.0,
                "joint4Angle": 0.0,
            }

            ptpMoveMode = "JumpCoordinate"
            SetPoseAct(api, pose, ptpMoveMode)

    else:
        window = Dobot_APP()
        window.loop()

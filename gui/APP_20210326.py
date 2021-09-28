import sys,os
sys.path.append(".")
sys.path.append("..")
sys.path.append("../../")
import time
import traceback

import cv2
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, FigureCanvasAgg
import PySimpleGUI as sg

from DobotFunction.Camera import WebCam_OnOff, Snapshot, scale_box, Preview, Color_cvt
from DobotFunction.Communication import Connect_Disconnect, Operation, _OneAction, SetPoseAct, GripperAutoCtrl
from DobotDLL import DobotDllType as dType

from ImageProcessing.Binarization import GlobalThreshold, AdaptiveThreshold, TwoThreshold
from ImageProcessing.Contrast import Contrast_cvt
from ImageProcessing.CenterOfGravity import CenterOfGravity
from src.config.config import cfg
from timeout_decorator import timeout, TimeoutError
#from ..DobotDLL

# from PIL import Image


class Dobot_APP:
    def __init__(self):
        dll_path = cfg.DOBOT_DLL_DIR + os.sep + "DobotDll.dll"
        #self.api = cdll.LoadLibrary(dll_path)
        self.api = dType.load(dll_path)
        self.connection = False #Dobotの接続状態
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
        self.RecordPose = {} # Dobotがオブジェクトを退避させる位置
        self.Alignment_1 = {}
        self.Alignment_2 = {}
        self.cam = None
        self.cam_num = None
        self.IMAGE_Org = None # スナップショットのオリジナル画像(RGB)
        self.IMAGE_bin = None # 二値画像
        # --- エンドエフェクタ --- #
        self.MotorON = False # 吸引用モータを制御
        # --- エラーフラグ --- #
        self.Dobot_err = {
            0: "DobotAct_NoError",
            1: "DobotConnect_NotFound",
            2: "DobotConnect_Occupied",
            3: "DobotAct_Timeout",
        }
        self.act_err = 0  # State: 0, Err: -1
        self.WebCam_err = {
            0: "WebCam_Connect",
            1: "WebCam_Release",
            2: "WebCam_NotFound",
            3: "WebCam_GetImage",
            4: "WebCam_NotGetImage"
        }
        # --- 画像プレビュー画面の初期値 --- #
        self.fig_agg = None     # 画像のヒストグラムを表示する用の変数
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
            [sg.Button("Connect or Disconnect", key="-Connect-")],
        ]

        EndEffector = [
            [sg.Button('SuctionCup ON/OFF', key='-SuctionCup-')],
            [sg.Button('Gripper Open/Close', key='-Gripper-')],
        ]

        # タスク1
        Task = [
            [sg.Button("タスク実行", size=(9, 1), key='-Task-'),
             sg.InputCombo(
                (
                    "Task_1",
                ),
                default_value="Task_1",
                size=(9, 1),
                key='-TaskNum-'),
            ],
        ]

        GetPose = [
            [sg.Button('Get Pose', size=(7, 1), key='-GetPose-')],
            [sg.Text('J1', size=(2, 1)),
             sg.InputText(default_text='',
                          size=(5, 1),
                          disabled=True,
                          key='-Get_JointPose1-',
                          readonly=True,),
             sg.Text('X', size=(1, 1)),
             sg.InputText(default_text='',
                          size=(5, 1),
                          disabled=True,
                          key='-Get_CoordinatePose_X-',
                          readonly=True),
            ],
            [sg.Text('J2', size=(2, 1)),
             sg.InputText(default_text='',
                          size=(5, 1),
                          disabled=True,
                          key='-Get_JointPose2-',
                          readonly=True),
             sg.Text('Y', size=(1, 1)),
             sg.InputText(default_text='',
                          size=(5, 1),
                          disabled=True,
                          key='-Get_CoordinatePose_Y-',
                          readonly=True)
            ],
            [sg.Text('J3', size=(2, 1)),
             sg.InputText(default_text='',
                          size=(5, 1),
                          disabled=True,
                          key='-Get_JointPose3-',
                          readonly=True),
             sg.Text('Z', size=(1, 1)),
             sg.InputText(default_text='',
                          size=(5, 1),
                          disabled=True,
                          key='-Get_CoordinatePose_Z-',
                          readonly=True),
            ],
            [sg.Text('J4', size=(2, 1)),
             sg.InputText(default_text='',
                          size=(5, 1),
                          disabled=True,
                          key='-Get_JointPose4-',
                          readonly=True),
             sg.Text('R', size=(1, 1)),
             sg.InputText(default_text='',
                          size=(5, 1),
                          disabled=True,
                          key='-Get_CoordinatePose_R-',
                          readonly=True),
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

        # 画像上の位置とDobotの座標系との位置合わせを行うGUI
        Alignment = [
            [sg.Button(button_text='MoveToThePoint',
                            size=(16, 1),
                            key='-MoveToThePoint-'),
            ],
            [sg.Button('Set', size=(8, 1), key='-Record-'),
             sg.Button(button_text='Set LeftUp',
                            size=(8, 1),
                            key='-Set_x1-'),
             sg.Button(button_text='Set RightDown',
                            size=(10, 1),
                            key='-Set_x2-'),
            ],
            [sg.Text('X0', size=(2, 1)),
             sg.InputText(default_text='',
                                size=(5, 1),
                                disabled=True,
                                key='-x0-'),
             sg.Text('X1', size=(2, 1)),
             sg.InputText(default_text='',
                                size=(5, 1),
                                disabled=True,
                                key='-x1-'),
             sg.Text('X2', size=(2, 1)),
             sg.InputText(default_text='',
                                size=(5, 1),
                                disabled=True,
                                key='-x2-'),
            ],
            [sg.Text('Y0', size=(2, 1)),
             sg.InputText(default_text='',
                                size=(5, 1),
                                disabled=True,
                                key='-y0-'),
             sg.Text('Y1', size=(2, 1)),
             sg.InputText(default_text='',
                                size=(5, 1),
                                disabled=True,
                                key='-y1-'),
             sg.Text('Y2', size=(2, 1)),
             sg.InputText(default_text='',
                                size=(5, 1),
                                disabled=True,
                                key='-y2-'),
            ],
            [sg.Text('Z0', size=(2, 1)),
             sg.InputText(default_text='',
                                size=(5, 1),
                                disabled=True,
                                key='-z0-'),
            ],
            [sg.Text('R0', size=(2, 1)),
             sg.InputText(default_text='',
                                size=(5, 1),
                                disabled=True,
                                key='-r0-'),
            ]
        ]

        WebCamConnect = [
            [
                # Web_Cameraの接続/解放
                sg.Button("WEB CAM on/off", size=(15, 1), key="-SetWebCam-"),
                # カメラのプレビュー
                sg.Button("Preview Opened", size=(11, 1), key="-Preview-"),
                # 静止画撮影
                sg.Button("Snapshot", size=(7, 1), key="-Snapshot-"),
            ],
            [
                # PCに接続されているカメラの選択
                sg.InputCombo(
                    ("TOSHIBA_Web_Camera-HD", "Logicool_HD_Webcam_C270",),
                    default_value="TOSHIBA_Web_Camera-HD",
                    size=(15, 1),
                    key="-WebCam_Name-",
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
                    size=(11, 1),
                    key="-WebCam_FrameSize-",
                    readonly=True,
                ),
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
            # 画像の色・濃度・フィルタリング
            [
                sg.Text('色空間', size=(5, 1)),
                sg.InputCombo(
                    (
                        'RGB',
                        'Gray',
                    ),
                    default_value='RGB',
                    size=(4, 1),
                    key='-Color_Space-',
                    readonly=True
                ),
                sg.Text('濃度変換', size=(7, 1)),
                sg.InputCombo(
                    (
                        'なし',
                        '線形濃度変換',
                        '非線形濃度変換',  # ガンマ処理
                        'ヒストグラム平坦化',
                    ),
                    default_value="なし",
                    size=(18, 1),
                    key='-Color_Density-',
                    readonly=True
                ),
                sg.Text('空間フィルタリング', size=(16, 1)),
                sg.InputCombo(
                    (
                        'なし',
                        '平均化',
                        'ガウシアン',
                        'メディアン',
                    ),
                    default_value="なし",
                    size=(10, 1),
                    key='-Color_Filtering-',
                    readonly=True
                ),
            ],
        ]

        # 二値化処理
        Binary = [
            [
                sg.InputCombo(
                    (
                        'なし',
                        'Global',
                        'Otsu',
                        'Adaptive',
                        'Two'
                    ),
                    default_value="なし",
                    size=(10, 1),
                    enable_events=True,
                    key='-Binarization-',
                    readonly=True
                ),
                sg.InputCombo(
                    (
                        'Mean',
                        'Gaussian',
                        'Wellner'
                    ),
                    default_value='Mean',
                    size=(12, 1),
                    disabled=True,
                    key='-AdaptiveThreshold_type-',
                    readonly=True
                )
            ],
            [
                sg.Text('Lower', size=(4, 1)),
                sg.Slider(
                    range=(0, 127),
                    default_value=10,
                    orientation='horizontal',
                    disabled=True,
                    size=(12, 12),
                    key='-LowerThreshold-'
                ),
                sg.Text('Block Size', size=(8, 1)),
                sg.InputText(
                    default_text='11',
                    size=(4, 1),
                    disabled=True,
                    justification='right',
                    key='-AdaptiveThreshold_BlockSize-',
                ),
            ],
            [
                sg.Text('Upper', size=(4, 1)),
                sg.Slider(
                    range=(128, 256),
                    default_value=138,
                    orientation='horizontal',
                    disabled=True,
                    size=(12, 12),
                    key='-UpperThreshold-'
                ),
                sg.Text('Constant', size=(8, 1)),
                sg.InputText(
                    default_text='2',
                    size=(4, 1),
                    disabled=True,
                    justification='right',
                    key='-AdaptiveThreshold_Constant-'
                )
            ],
            [sg.Radio(text='R',
                      group_id='color',
                      disabled=True,
                      background_color='grey59',
                      text_color='red',
                      key='-color_R-'),
             sg.Radio(text='G',
                      group_id='color',
                      disabled=True,
                      background_color='grey59',
                      text_color='green',
                      key='-color_G-'),
             sg.Radio(text='B',
                      group_id='color',
                      disabled=True,
                      background_color='grey59',
                      text_color='blue',
                      key='-color_B-'),
             sg.Radio(text='W',
                      group_id='color',
                      disabled=True,
                      background_color='grey59',
                      text_color='snow',
                      key='-color_W-'),
             sg.Radio(text='Bk',
                      group_id='color',
                      default=True,
                      disabled=True,
                      background_color='grey59',
                      text_color='grey1',
                      key='-color_Bk-')
            ],
        ]

        # 画像から物体の輪郭を切り出す関数の設定部分_GUI
        ContourExtractionSettings = [
            # 輪郭のモードを指定する
            [sg.Button(button_text='Contours',
                       size=(7, 1),
                       key='-Contours-'),
            ],
            [sg.Text('輪郭', size=(3, 1)),
             sg.InputCombo(
                ('親子関係を無視する',
                 '最外の輪郭を検出する',
                 '2つの階層に分類する',
                 '全階層情報を保持する',
                ),
                default_value='全階層情報を保持する',
                size=(20, 1),
                key='-RetrievalMode-',
                readonly=True),
            ],
            # 近似方法を指定する
            [sg.Text('近似方法', size=(7, 1)),
             sg.InputCombo(
                (
                    '中間点を保持する',
                    '中間点を保持しない',
                ),
                default_value='中間点を保持する',
                size=(18, 1),
                key='-ApproximateMode-',
                readonly=True),
            ],
            [sg.Text('Gx', size=(2, 1)),
             sg.InputText(default_text='0',
                          size=(5, 1),
                          justification='right',
                          key='-CenterOfGravity_x-',
                          readonly=True),
             sg.Text('Gy', size=(2, 1)),
             sg.InputText(default_text='0',
                          size=(5, 1),
                          justification='right',
                          key='-CenterOfGravity_y-',
                          readonly=True),
            ],
        ]

        layout = [
            [sg.Col(Connect), sg.Col(EndEffector), sg.Col(Task),],
            [sg.Col(GetPose, size=(165, 140)),
             sg.Col(SetPose, size=(165, 140)),
             sg.Frame(title="キャリブレーション", layout=Alignment)],
            [sg.Col(WebCamConnect),],
            [sg.Frame(title="二値化処理", layout=Binary),
             sg.Frame(title='画像の重心計算', layout=ContourExtractionSettings),
            ],
            [sg.Image(filename='', size=(self.Image_width, self.Image_height), key='-IMAGE-'),
             sg.Canvas(size=(self.Image_width, self.Image_height), key='-CANVAS-')
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
        #---------------------------------------------
        # 操作時にウインドウが変化するイベント群
        #---------------------------------------------
        if event == "-Binarization-":
            if values["-Binarization-"] != "なし":
                self.Window['-Color_Space-'].update("Gray")
                self.Window['-LowerThreshold-'].update(disabled=False)

            if values["-Binarization-"] == "Adaptive":
                self.Window['-AdaptiveThreshold_type-'].update(disabled=False, readonly=True)
                self.Window['-AdaptiveThreshold_BlockSize-'].update(disabled=False)
                self.Window['-AdaptiveThreshold_Constant-'].update(disabled=False)
            elif values["-Binarization-"] == "Two":
                # 各色選択ボタンを有効化
                self.Window['-color_R-'].update(disabled=False)
                self.Window['-color_G-'].update(disabled=False)
                self.Window['-color_B-'].update(disabled=False)
                self.Window['-color_W-'].update(disabled=False)
                self.Window['-color_Bk-'].update(disabled=False)
                self.Window['-UpperThreshold-'].update(disabled=False)

        # Dobotの接続を行う
        if event == "-Connect-":
            # self.connection = self.Connect_Disconnect_click(self.connection, self.api)
            self.connection, err = Connect_Disconnect(
                self.connection,
                self.api,
            )

            if self.connection:
                # Dobotの現在の姿勢を画面上に表示
                self.current_pose = self.GetPose_UpdateWindow()

        # --------------------- #
        # グリッパを動作させる #
        # --------------------- #
        elif event == '-Gripper-':
            if self.connection:
                # グリッパを開く
                GripperAutoCtrl(self.api)

        # ------------------------------------------- #
        # 現在の姿勢を取得し、画面上のに表示する #
        # ------------------------------------------- #
        elif event == "-GetPose-":
            if self.connection:
                self.GetPose_UpdateWindow()

        elif event == "-SetJointPose-":
            # 移動後の関節角度を指定
            DestPose = [
                float(values["-JointPoseInput_1-"]),
                float(values["-JointPoseInput_2-"]),
                float(values["-JointPoseInput_3-"]),
                float(values["-JointPoseInput_4-"]),
            ]
            response = self.SetJointPose_click()
            print(response)

        # ---------------------------------- #
        # Dobotの動作終了位置を設定する #
        # ---------------------------------- #
        elif event == '-Record-':
            if self.connection:
                self.GetPose_UpdateWindow()

                self.Window['-x0-'].update(str(self.CurrentPose["x"]))
                self.Window['-y0-'].update(str(self.CurrentPose["y"]))
                self.Window['-z0-'].update(str(self.CurrentPose["z"]))
                self.Window['-r0-'].update(str(self.CurrentPose["r"]))

                self.RecordPose = self.CurrentPose # 現在の姿勢を記録

        # ----------------------------------------------------------- #
        # 画像とDobotの座標系の位置合わせ用変数_1をセットする #
        # ----------------------------------------------------------- #
        elif event == '-Set_x1-':
            if self.connection:
                self.GetPose_UpdateWindow()

                self.Alignment_1["x"] = self.CurrentPose["x"]
                self.Alignment_1["y"] = self.CurrentPose["y"]
                self.Window['-x1-'].update(str(self.Alignment_1["x"]))
                self.Window['-y1-'].update(str(self.Alignment_1["y"]))

        # -------------------------------------------------- #
        #  画像とDobotの座標系の位置合わせ用変数_2をセットする   #
        # -------------------------------------------------- #
        elif event == '-Set_x2-':
            if self.connection:
                self.GetPose_UpdateWindow()

                self.Alignment_2["x"] = self.CurrentPose["x"]
                self.Alignment_2["y"] = self.CurrentPose["y"]
                self.Window['-x2-'].update(str(self.Alignment_2["x"]))
                self.Window['-y2-'].update(str(self.Alignment_2["y"]))


        # ------------------------------------ #
        # WebCamを選択&接続するイベント #
        # ------------------------------------ #
        elif event == "-SetWebCam-":
            # Webカメラの番号を取得する
            cam_num = WebCamOption(values["-WebCam_Name-"])
            # webカメラの番号が取得できなかった場合
            if cam_num is None:
                sg.popup("選択したデバイスは存在しません。", title="カメラ接続エラー")
                return

            # ----------------------------#
            # カメラを接続するイベント #
            # --------------------------- #
            # カメラを初めて接続する場合 -> カメラを新規接続
            # 接続したいカメラが接続していカメラと同じ場合 -> カメラを解放
            if (self.cam_num == None) or (self.cam_num == cam_num):
                response, self.cam = WebCam_OnOff(cam_num, cam=self.cam)
                sg.popup(self.WebCam_err[response], title="Camの接続")

            # 接続したいカメラと接続しているカメラが違う場合 -> 接続しているカメラを解放し、新規接続
            elif (self.cam_num != None) and (self.cam_num != cam_num):
                # まず接続しているカメラを開放する．
                response, self.cam = WebCam_OnOff(self.cam_num, cam=self.cam)
                # 開放できた場合
                if response == 1:
                    sg.popup(self.WebCam_err[response], title="Camの接続")
                    # 次に新しいカメラを接続する．
                    response, self.cam = WebCam_OnOff(cam_num, cam=self.cam)
                    sg.popup(self.WebCam_err[response], title="Camの接続")

            # カメラが問題なく見つかった場合
            if response != 2: self.cam_num = cam_num
            else: self.cam_num = None


        #---------------------------------#
        # プレビューを表示するイベント #
        #--------------------------------  #
        elif event == "-Preview-":
            window_name = "frame"

            while True:
                if type(self.cam) == cv2.VideoCapture:  # カメラが接続されている場合
                    response, img = Snapshot(self.cam)
                    if response:
                        #----------------------------------------#
                        # 画像サイズなどをダイアログ上に表示 #
                        #----------------------------------------#
                        self.Window["-IMAGE_width-"].update(str(img.shape[0]))
                        self.Window["-IMAGE_height-"].update(str(img.shape[1]))
                        self.Window["-IMAGE_channel-"].update(
                            str(img.shape[2]))

                        print("プレビューを表示します。")
                        response = Preview(img, window_name=window_name)
                        if cv2.waitKey(0) & 0xFF == ord("e"):
                            cv2.destroyWindow(window_name)
                            break
                    else:
                        sg.popup("SnapShotを撮影できませんでした．", title="撮影エラー")
                        break
                else:
                    sg.popup("カメラが接続されていません．", title="カメラ接続エラー")
                    break

        # --------------------------------------- #
        # スナップショットを撮影するイベント #
        # --------------------------------------- #
        elif event == '-Snapshot-':
            if type(self.cam) == cv2.VideoCapture:
                self.IMAGE_Org, self.IMAGE_bin = self.SnapshotBtn(values)
            else:
                sg.popup(self.WebCam_err[2], title="カメラ接続エラー")

        # -------------------------- #
        # COGを計算するイベント #
        # ------------------------- #
        elif event == '-Contours-':
            if type(self.cam) == cv2.VideoCapture:
                self.ContoursBtn(values)
            else:
                sg.popup(self.WebCam_err[2], title="カメラ接続エラー")

        # -------------------------------------------------------------- #
        # オブジェクトの重心位置に移動→掴む→退避 動作を実行する #
        # -------------------------------------------------------------- #
        elif event == '-Task-':
            if values['-TaskNum-'] == "Task_1":
                if (self.connection == True) and (type(self.cam) == cv2.VideoCapture):
                    if (not self.Alignment_1) or \
                       (not self.Alignment_2) or \
                       (not self.RecordPose):
                        sg.popup("キャリブレーション座標 x1 および x2, \nもしくは 退避位置 がセットされていません。")
                    else:
                        # 画像を撮影 & 重心位置を計算
                        COG = self.ContoursBtn(values)
                        # 現在のDobotの姿勢を取得
                        pose = self.GetPose_UpdateWindow() # pose -> self.CurrentPose
                        # ------------------------------ #
                        # Dobotの移動後の姿勢を計算 #
                        # ------------------------------ #
                        try:
                            pose["x"] = self.Alignment_1["x"] + COG[0] * \
                                (self.Alignment_2["x"]-self.Alignment_1["x"]) / float(self.Image_height)
                            pose["y"] = self.Alignment_1["y"] + COG[1] * \
                                (self.Alignment_2["y"]-self.Alignment_1["y"]) / float(self.Image_width)
                        except ZeroDivisionError: # ゼロ割が発生した場合
                            sg.popup('画像のサイズが計測されていません', title='エラー')
                            return

                        # Dobotをオブジェクト重心の真上まで移動させる。
                        self.SetCoordinatePose_click(pose)
                        # グリッパーを開く。
                        GripperAutoCtrl(self.api)
                        # DobotをZ=-35の位置まで降下させる。
                        pose["z"]=-35
                        self.SetCoordinatePose_click(pose)
                        # グリッパを閉じる。
                        GripperAutoCtrl(self.api)
                        # Dobotを上昇させる。
                        pose["z"] = self.CurrentPose["z"]
                        self.SetCoordinatePose_click(pose)
                        # 退避位置まで移動させる。
                        self.SetCoordinatePose_click(self.RecordPose)




                else: sg.popup("Dobotかカメラが接続されていません。")


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
            if event != "__TIMEOUT__":
                self.Event(event, values)

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

    def SetCoordinatePose_click(self, pose: dict, queue_index: int=1):
        """デカルト座標系で指定された位置にアームの先端を移動させる関数

        Arg:
            pose(dict): デカルト座標系および関節座標系で指定された姿勢データ

        Return:
            response(int):
                0 : 応答あり
                1 : 応答なし
        """
        dType.SetPTPCmd(self.api,
                                   dType.PTPMode.PTPMOVJXYZMode,
                                   pose["x"],
                                   pose["y"],
                                   pose["z"],
                                   pose["r"],
                                   queue_index
        )
        time.sleep(5)
        return 0


    def GetPose_UpdateWindow(self) -> dict:
        """
        姿勢を取得し、ウインドウを更新する関数

        Return:
            CurrentPose (dict): Dobotの現在の姿勢
        """

        pose = dType.GetPose(self.api)  # 現在のDobotの位置と関節角度を取得
        for num, key in  enumerate(self.CurrentPose.keys()):
            self.CurrentPose[key] = pose[num]

        self.Window["-Get_JointPose1-"].update(str(self.CurrentPose["joint1Angle"]))
        self.Window["-Get_JointPose2-"].update(str(self.CurrentPose["joint2Angle"]))
        self.Window["-Get_JointPose3-"].update(str(self.CurrentPose["joint3Angle"]))
        self.Window["-Get_JointPose4-"].update(str(self.CurrentPose["joint4Angle"]))
        self.Window["-Get_CoordinatePose_X-"].update(str(self.CurrentPose["x"]))
        self.Window["-Get_CoordinatePose_Y-"].update(str(self.CurrentPose["y"]))
        self.Window["-Get_CoordinatePose_Z-"].update(str(self.CurrentPose["z"]))
        self.Window["-Get_CoordinatePose_R-"].update(str(self.CurrentPose["r"]))

        return self.CurrentPose


    def SnapshotBtn(self, values: list) -> np.ndarray:
        """スナップショットの撮影から一連の画像処理を行う関数。

        Args:
            values (list): ウインドウ上のボタンの状態などを記録している変数
        Returns:
            dst_org (np.ndarray): オリジナルのスナップショット
            dst_bin (np.ndarray): 二値化処理後の画像
        """
        dst_org = dst_bin = None
        response, img = Snapshot(self.cam)
        if response != 3:
            sg.popup(self.WebCam_err[response], title="撮影エラー")
            return

        dst_org = img.copy()
        #--------------------------- #
        # 画像サイズなどをダイアログ上に表示 #
        #--------------------------- #
        self.Window["-IMAGE_width-"].update(str(img.shape[0]))
        self.Window["-IMAGE_height-"].update(str(img.shape[1]))
        self.Window["-IMAGE_channel-"].update(str(img.shape[2]))

        # ---------------------------
        # 撮影した画像を変換する。
        # ---------------------------
        # 色空間変換
        if values['-Color_Space-'] != 'RGB':
            img = Color_cvt(img, values['-Color_Space-'])
        # ノイズ除去

        # 濃度変換
        if values['-Color_Density-'] != 'なし':
            img = Contrast_cvt(img, values['-Color_Density-'])

        # ------------
        # 画像処理
        # ------------
        if values['-Binarization-'] != 'なし':  # 二値化処理
            if values['-Binarization-'] == 'Global':  # 大域的二値化処理
                img= GlobalThreshold(img, threshold=int(values["-LowerThreshold-"]))
            elif values['-Binarization-'] == 'Otsu':  # 大津の二値化処理
                img = GlobalThreshold(img, Type='Otsu')
            elif values['-Binarization-'] == 'Adaptive':
                img = AdaptiveThreshold(
                    img=img,
                    method=str(values['-AdaptiveThreshold_type-']),
                    block_size=int(values['-AdaptiveThreshold_BlockSize-']),
                    C=int(values['-AdaptiveThreshold_Constant-'])
                )
            elif values['-Binarization-'] == 'Two':  # 2つの閾値を用いた二値化処理
                # ピックアップする色を番号に変換
                if values['-color_R-']: color = 0
                elif values['-color_G-']: color = 1
                elif values['-color_B-']: color = 2
                elif values['-color_W-']: color = 3
                elif values['-color_Bk-']: color = 4
                img = TwoThreshold(
                    img=img,
                    LowerThreshold=int(values["-LowerThreshold-"]),
                    UpperThreshold=int(values["-UpperThreshold-"]),
                    PickupColor=color
                )
            dst_bin = img.copy()

        # ----------------------- #
        # 画面上に撮影した画像を表示する #
        # ----------------------- #
        img = scale_box(img, self.Image_width, self.Image_height)
        imgbytes = cv2.imencode('.png', img)[1].tobytes()
        self.Window['-IMAGE-'].update(data=imgbytes)

        # --------------------------------- #
        # 画面上に撮影した画像のヒストグラムを表示する #
        # --------------------------------- #
        canvas_elem = self.Window['-CANVAS-']
        canvas = canvas_elem.TKCanvas
        ticks = [0, 42, 84, 127, 169, 211, 255]
        fig, ax = plt.subplots(figsize=(3, 2))
        ax = Image_hist(img, ax, ticks)
        if self.fig_agg:
            # ** IMPORTANT ** Clean up previous drawing before drawing again
            delete_figure_agg(self.fig_agg)
        self.fig_agg = draw_figure(canvas, fig)
        self.fig_agg.draw()

        return dst_org, dst_bin

    def ContoursBtn(self, values: list):
        """スナップショットの撮影からオブジェクトの重心位置計算までの一連の画像処理を行う関数。

        Args:
            values (list): ウインドウ上のボタンの状態などを記録している変数
        Returns:
            COG(list): COG=[x, y], オブジェクトの重心位置
        """
        COG = []
        # 輪郭情報
        RetrievalMode = {
            '親子関係を無視する': cv2.RETR_LIST,
            '最外の輪郭を検出する': cv2.RETR_EXTERNAL,
            '2つの階層に分類する': cv2.RETR_CCOMP,
            '全階層情報を保持する': cv2.RETR_TREE
        }
        # 輪郭の中間点情報
        ApproximateMode = {
            '中間点を保持する': cv2.CHAIN_APPROX_NONE,
            '中間点を保持しない': cv2.CHAIN_APPROX_SIMPLE
        }

        # スナップショットを撮影する
        self.IMAGE_Org, self.IMAGE_bin = self.SnapshotBtn(values)
        try:
            if type(self.IMAGE_bin) != np.ndarray:
                raise TypeError('入力はnumpy配列を使用してください。')

            COG = CenterOfGravity(
                bin_img=self.IMAGE_bin,
                RetrievalMode=RetrievalMode[str(values['-RetrievalMode-'])],
                ApproximateMode=ApproximateMode[str(values['-ApproximateMode-'])],
                min_area=100,
                cal_Method=1
            )
        except Exception as e:
            traceback.print_exc()
        else:
            self.Window['-CenterOfGravity_x-'].update(str(COG[0]))
            self.Window['-CenterOfGravity_y-'].update(str(COG[1]))
        finally:
            return COG


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
        color = ['k']
    elif len(img.shape) == 3:
        color = ['r', 'g', 'b']
    for (i, col) in enumerate(color):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        hist = np.sqrt(hist)
        ax.plot(hist, color=col)

    if ticks:
        ax.set_xticks(ticks)
    ax.set_title('histogram')
    ax.set_xlim([0, 256])

    return ax


def draw_figure(canvas, figure, loc=(0, 0)):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


def delete_figure_agg(figure_agg):
    figure_agg.get_tk_widget().forget()
    plt.close('all')


if __name__ == "__main__":
    """
    response, cam = WebCam_OnOff(device_num=0)
    window_name = "frame"
    while True:
        # ret, frame = cam.read()
        ret, frame = Snapshot(cam)
        if ret:
            # cv2.imshow("frame", frame)
            Preview(frame, window_name=window_name)
            if cv2.waitKey(0) & 0xFF == ord("e"):
                cv2.destroyWindow(window_name)
                break
    """
    window = Dobot_APP()
    window.loop()

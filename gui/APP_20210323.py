import sys
import os

sys.path.append(".")
sys.path.append("..")
sys.path.append("../../")

import cv2
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, FigureCanvasAgg
import PySimpleGUI as sg

from DobotFunction.Camera import WebCam_OnOff, Snapshot, scale_box, Preview, Color_cvt
from DobotFunction.Communication import Connect_Disconnect, Operation, _OneAction, GripperAutoCtrl
from DobotDLL import DobotDllType as dType

from ImageProcessing.Binarization import GlobalThreshold, AdaptiveThreshold, TwoThreshold
from ImageProcessing.Contrast import Contrast_cvt
from ImageProcessing.CenterOfGravity import CenterOfGravity
from src.config.config import cfg
#from ..DobotDLL

# from PIL import Image


class Dobot_APP:
    def __init__(self):
        dll_path = cfg.DOBOT_DLL_DIR + os.sep + "DobotDll.dll"
        #self.api = cdll.LoadLibrary(dll_path)
        self.api = dType.load(dll_path)
        self.pose_key = [
            "x",
            "y",
            "z",
            "r",
            "joint1Angle",
            "joint2Angle",
            "joint3Angle",
            "joint4Angle",
        ]
        self.connection = False #Dobotの接続状態
        self.current_pose = {}  # Dobotの現在の姿勢
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
        # --- 画像プレビュー画面の初期値 --- #
        self.fig_agg = None     # 画像のヒストグラムを表示する用の変数
        self.Image_height = 240  # 画面上に表示する画像の高さ
        self.Image_width = 320  # 画面上に表示する画像の幅
        # --- GUIの初期化 ---#
        self.layout = self.Layout()
        self.Window = self.main()

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

    def GetPose_UpdateWindow(self) -> dict:
        """
        姿勢を取得し、ウインドウを更新する関数

        Return:
            pose (dict):
                Dobotの現在の姿勢
        """

        pose = dType.GetPose(self.api)  # 現在のDobotの位置と関節角度を取得
        pose = dict(zip(self.pose_key, pose))  # 辞書形式に変換

        self.Window["-JointPose1-"].update(str(pose["joint1Angle"]))
        self.Window["-JointPose2-"].update(str(pose["joint2Angle"]))
        self.Window["-JointPose3-"].update(str(pose["joint3Angle"]))
        self.Window["-JointPose4-"].update(str(pose["joint4Angle"]))
        self.Window["-CoordinatePose_X-"].update(str(pose["x"]))
        self.Window["-CoordinatePose_Y-"].update(str(pose["y"]))
        self.Window["-CoordinatePose_Z-"].update(str(pose["z"]))
        self.Window["-CoordinatePose_R-"].update(str(pose["r"]))

        return pose

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

        GetPose = [
            [sg.Button('Get Pose', size=(7, 1), key='-GetPose-')],
            [sg.Text('J1', size=(2, 1)),
             sg.InputText(default_text='',
                          size=(5, 1),
                          disabled=True,
                          key='-JointPose1-',
                          readonly=True,),
             sg.Text('X', size=(1, 1)),
             sg.InputText(default_text='',
                          size=(5, 1),
                          disabled=True,
                          key='-CoordinatePose_X-',
                          readonly=True),
            ],
            [sg.Text('J2', size=(2, 1)),
             sg.InputText(default_text='',
                          size=(5, 1),
                          disabled=True,
                          key='-JointPose2-',
                          readonly=True),
             sg.Text('Y', size=(1, 1)),
             sg.InputText(default_text='',
                          size=(5, 1),
                          disabled=True,
                          key='-CoordinatePose_Y-',
                          readonly=True)
            ],
            [sg.Text('J3', size=(2, 1)),
             sg.InputText(default_text='',
                          size=(5, 1),
                          disabled=True,
                          key='-JointPose3-',
                          readonly=True),
             sg.Text('Z', size=(1, 1)),
             sg.InputText(default_text='',
                          size=(5, 1),
                          disabled=True,
                          key='-CoordinatePose_Z-',
                          readonly=True),
            ],
            [sg.Text('J4', size=(2, 1)),
             sg.InputText(default_text='',
                          size=(5, 1),
                          disabled=True,
                          key='-JointPose4-',
                          readonly=True),
             sg.Text('R', size=(1, 1)),
             sg.InputText(default_text='',
                          size=(5, 1),
                          disabled=True,
                          key='-CoordinatePose_R-',
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
            [sg.Button('MoveToThePoint', size=(
                16, 1), key='-MoveToThePoint-'), ],
            [sg.Button('Set', size=(8, 1), key='-Record-'),
             sg.Button('Set', size=(8, 1), key='-Set_x1-'),
             sg.Button('Set', size=(8, 1), key='-Set_x2-'), ],
            [sg.Text('X0', size=(2, 1)), sg.InputText('', size=(5, 1), disabled=True, key='-x0-'),
             sg.Text('X1', size=(2, 1)), sg.InputText(
                 '', size=(5, 1), disabled=True, key='-x1-'),
             sg.Text('X2', size=(2, 1)), sg.InputText('', size=(5, 1), disabled=True, key='-x2-'), ],
            [sg.Text('Y0', size=(2, 1)), sg.InputText('', size=(5, 1), disabled=True, key='-y0-'),
             sg.Text('Y1', size=(2, 1)), sg.InputText(
                 '', size=(5, 1), disabled=True, key='-y1-'),
             sg.Text('Y2', size=(2, 1)), sg.InputText('', size=(5, 1), disabled=True, key='-y2-'), ],
            [sg.Text('Z0', size=(2, 1)), sg.InputText(
                '', size=(5, 1), disabled=True, key='-z0-'), ],
            [sg.Text('R0', size=(2, 1)), sg.InputText(
                '', size=(5, 1), disabled=True, key='-r0-'), ]
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
                        'Glay',
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
             sg.InputCombo(('親子関係を無視する',
                            '最外の輪郭を検出する',
                            '2つの階層に分類する',
                            '全階層情報を保持する',),
                            default_value='全階層情報を保持する',
                            size=(20, 1),
                            key='-RetrievalMode-',
                            readonly=True),
            ],
            # 近似方法を指定する
            [sg.Text('近似方法', size=(7, 1)),
             sg.InputCombo(('中間点を保持する',
                            '中間点を保持しない'),
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

        # タスク1
        task_1 = [
            [sg.Text("タスク 1", size=(5, 1)),
             sg.Button(button_text="タスク 1",
                            size=(5, 1),
                            key='-task_1-'),
            ]
        ]

        layout = [
            [sg.Col(Connect), sg.Col(EndEffector)],
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
                self.Window['-LowerThreshold-'].update(disabled=False)

            if values["-Binarization-"] == "Adaptive":
                self.Window['-AdaptiveThreshold_type-'].update(disabled=False)
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
            # カメラを初めて接続する場合
            if (cam_num != None) and (self.cam_num == None):
                response, self.cam = WebCam_OnOff(cam_num, cam=self.cam)
                if response == -1:  # カメラが接続されていない場合
                    sg.popup("WebCameraに接続できません．", title="カメラ接続エラー")
                elif response == 0:  # カメラを開放した場合
                    sg.popup("WebCameraを開放しました。", title="Camの接続")
                else:
                    sg.popup("WebCameraに接続しました。", title="Camの接続")

            # 接続したいカメラが接続していカメラと同じ場合
            elif (cam_num != None) and (self.cam_num == cam_num):
                response, self.cam = WebCam_OnOff(cam_num, cam=self.cam)
                if response == -1:  # カメラが接続されていない場合
                    sg.popup("WebCameraに接続できません．", title="カメラ接続エラー")
                elif response == 0:  # カメラを開放した場合
                    sg.popup("WebCameraを開放しました。", title="Camの接続")
                else:
                    sg.popup("WebCameraに接続しました。", title="Camの接続")

            # 接続したいカメラと接続しているカメラが違う場合
            elif (cam_num != None) and (self.cam_num != cam_num):
                # まず接続しているカメラを開放する．
                ch_1, self.cam = WebCam_OnOff(cam_num, cam=self.cam)
                # 開放できた場合
                if ch_1 == 0:
                    sg.popup("WebCameraを開放しました。", title="Camの接続")
                    # 次に新しいカメラを接続する．
                    ch_2, self.cam = WebCam_OnOff(cam_num, cam=self.cam)
                    # カメラが接続されていない場合
                    if ch_2 == -1:
                        sg.popup("WebCameraに接続できません．", title="カメラ接続エラー")
                    else:
                        sg.popup("WebCameraに接続しました．", title="Camの接続")
                # 新しいカメラを接続した場合
                elif ch_1 == 1:
                    sg.popup("新しくWebCameraに接続しました．", title="Camの接続")
                # カメラの接続に失敗した場合
                else:
                    self.cam = None

            self.cam_num = cam_num

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

        # ---------------------------- #
        #  スナップショットを撮影するイベント  #
        # ---------------------------- #
        elif event == '-Snapshot-':
            self.IMAGE_Org, self.IMAGE_bin = self.SnapshotBtn(values)

        # ------------------- #
        #  COGを計算するイベント  #
        # ------------------- #
        elif event == '-Contours-':
            # スナップショットを撮影する
            self.IMAGE_Org, self.IMAGE_bin = self.SnapshotBtn(values)
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

            if type(self.IMAGE_bin) != np.ndarray:
                raise TypeError('入力はnumpy配列を使用してください。')

            COG = CenterOfGravity(bin_img=self.IMAGE_bin,
                                  RetrievalMode=RetrievalMode[str(values['-RetrievalMode-'])],
                                  ApproximateMode=ApproximateMode[str(values['-ApproximateMode-'])],
                                  min_area=100,
                                  cal_Method=1)

            self.Window['-CenterOfGravity_x-'].update(str(COG[0]))
            self.Window['-CenterOfGravity_y-'].update(str(COG[1]))


        # --------------------------------------------------- #
        #  オブジェクトの重心位置に移動→掴む→退避 動作を実行する  #
        # --------------------------------------------------- #
        #elif event == '-task_1-':


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
        if response != 1:
            sg.popup("スナップショットを撮影できませんでした。", title="撮影エラー")
            raise Exception("スナップショットを撮影できませんでした。")

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

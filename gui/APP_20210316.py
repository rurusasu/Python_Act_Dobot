import sys
import os

sys.path.append(".")
sys.path.append("..")
sys.path.append("../../")

from ctypes import*

import cv2
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, FigureCanvasAgg
import PySimpleGUI as sg


from ImageProcessing.Contrast import Contrast_cvt
from DobotFunction.Camera import WebCam_OnOff, Snapshot, scale_box, Color_cvt
from DobotFunction.Communication import Connect_Disconnect, Operation, _OneAction
from DobotDLL import DobotDllType as dType
from src.config.config import cfg
#from ..DobotDLL

# from PIL import Image


class Dobot_APP:
    def __init__(self):
        #dll_path = cfg.DOBOT_DLL_DIR + os.sep + "DobotDll.dll"
        #self.api = cdll.LoadLibrary(dll_path)
        self.api = dType.load()
        self.CON_STR = {
            dType.DobotConnect.DobotConnect_NoError: "DobotConnect_NoError",
            dType.DobotConnect.DobotConnect_NotFound: "DobotConnect_NotFound",
            dType.DobotConnect.DobotConnect_Occupied: "DobotConnect_Occupied",
        }
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
        self.current_pose = {}  # Dobotの現在の姿勢
        self.cam = None
        self.cam_num = None
        # --- エラーフラグ --- #
        self.connection = 1  # Connect: 0, DisConnect: 1, Err: -1
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
        Connect = [sg.Button("Connect or Disconnect", key="-Connect-")]

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
            [
                sg.Text('重心位置計算', size=(5, 1)),
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

        Contours = [
            [
                sg.Button('Contours', size=(7, 1), key='-Contours-'),
                sg.InputCombo(('画像をもとに計算',
                               '輪郭をもとに計算', ), size=(16, 1), key='-CenterOfGravity-')
            ],
            [
                sg.Text('Gx', size=(2, 1)),
                sg.InputText('0', size=(5, 1), disabled=True,
                             justification='right', key='-CenterOfGravity_x-'),
                sg.Text('Gy', size=(2, 1)),
                sg.InputText('0', size=(5, 1), disabled=True,
                             justification='right', key='-CenterOfGravity_y-'),
            ],
        ]

        ColorOfObject = [
            [
                sg.Radio('R', group_id='color', background_color='grey59',
                         text_color='red', key='-color_R-'),
                sg.Radio('G', group_id='color', background_color='grey59',
                         text_color='green', key='-color_G-'),
                sg.Radio('B', group_id='color', background_color='grey59',
                         text_color='blue', key='-color_B-'),
                sg.Radio('W', group_id='color', background_color='grey59',
                         text_color='snow', key='-color_W-'),
                sg.Radio('Bk', group_id='color', default=True,
                         background_color='grey59', text_color='grey1', key='-color_Bk-')
            ],
        ]

        Bin_CommonSettings = [
            [
                sg.Text('閾値の処理方法'),
                sg.InputCombo(('BINARY',
                               'BINARY_INV',
                               'TRUNC',
                               'TOZERO',
                               'TOZERO_INV', ), size=(12, 1), key='-Threshold_type-', readonly=True)
            ],
            [sg.Checkbox('ヒストグラム平坦化', key='-EqualizeHist-')],
            [sg.Checkbox('ガウシアンフィルタ', key='-Gaussian-'), ],
        ]

        # 画像から物体の輪郭を切り出す関数の設定部分_GUI
        ContourExtractionSettings = [
            # 輪郭のモードを指定する
            [sg.Text('輪郭', size=(10, 1)),
             sg.InputCombo(('親子関係を無視する',
                            '最外の輪郭を検出する',
                            '2つの階層に分類する',
                            '全階層情報を保持する',), size=(20, 1), key='-ContourRetrievalMode-', readonly=True), ],
            # 近似方法を指定する
            [sg.Text('近似方法', size=(10, 1)),
             sg.InputCombo(('中間点を保持する',
                            '中間点を保持しない'), size=(18, 1), key='-ApproximateMode-', readonly=True), ],
            # カーネルの形を指定する
            [sg.Text('カーネルの形', size=(10, 1)),
             sg.InputCombo(('矩形',
                            '楕円形',
                            '十字型'), size=(6, 1), key='-KernelShape-', readonly=True), ],
        ]

        # 大域的二値化
        Global_Threshold = [
            [sg.Radio('Global Threshold', group_id='threshold',
                      key='-GlobalThreshold-'), ],
            [sg.Text('Threshold', size=(7, 1)), sg.InputText(
                '127', size=(4, 1), justification='right', key='-threshold-')],
            [sg.Checkbox('大津の二値化', key='-OTSU-')],
        ]

        # 適応的二値化
        Adaptive_Threshold = [
            [sg.Radio('Adaptive Threshold', group_id='threshold',
                      key='-AdaptiveThreshold-'), ],
            [sg.InputCombo(('MEAN_C', 'GAUSSIAN_C'), size=(
                12, 1), key='-AdaptiveThreshold_type-', readonly=True)],
            [sg.Text('Block Size', size=(8, 1)), sg.InputText('11', size=(
                4, 1), justification='right', key='-AdaptiveThreshold_BlockSize-'), ],
            [sg.Text('Constant', size=(8, 1)), sg.InputText('2', size=(
                4, 1), justification='right', key='-AdaptiveThreshold_constant-')],
        ]

        # 2つの閾値を用いた二値化
        TwoThreshold = [
            [sg.Radio('TwoThreshold', group_id='threshold',
                      default=True, key='-Twohreshold-'), ],
            [sg.Text('Lower', size=(4, 1)), sg.Slider(range=(0, 127), default_value=10,
                                                      orientation='horizontal', size=(12, 12), key='-LowerThreshold-'), ],
            [sg.Text('Upper', size=(4, 1)),  sg.Slider(range=(128, 256), default_value=138,
                                                       orientation='horizontal', size=(12, 12), key='-UpperThreshold-')]
        ]

        layout = [
            Connect,
            [sg.Col(SetPose, size=(165, 136)), ],
            [sg.Col(WebCamConnect),  sg.Frame('画像の重心計算', Contours), ],
            [sg.Image(filename='', size=(self.Image_width, self.Image_height), key='-IMAGE-'),
             sg.Canvas(size=(self.Image_width, self.Image_height), key='-CANVAS-')],
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
            # self.connection = self.Connect_Disconnect_click(self.connection, self.api)
            self.connection = Connect_Disconnect(
                self.connection, self.api, self.CON_STR
            )

            if self.connection == 0:
                # Dobotの現在の姿勢を画面上に表示
                self.current_pose = self.GetPose_UpdateWindow()

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

        # ---------------------------------------- #
        #  スナップショットを撮影するイベント  #
        # ---------------------------------------- #
        elif event == '-Snapshot-':
            response, img = Snapshot(self.cam)
            if response != 1:
                sg.popup("スナップショットを撮影できませんでした。", title="撮影エラー")

            #----------------------------------------#
            # 画像サイズなどをダイアログ上に表示 #
            #----------------------------------------#
            self.Window["-IMAGE_width-"].update(str(img.shape[0]))
            self.Window["-IMAGE_height-"].update(str(img.shape[1]))
            self.Window["-IMAGE_channel-"].update(str(img.shape[2]))

            # ---------------------------
            #    撮影した画像を変換する。
            # ---------------------------
            # 色空間変換
            if values['-Color_Space-'] != 'RGB':
                img = Color_cvt(img, values['-Color_Space-'])
            # ノイズ除去

            # 濃度変換
            if values['-Color_Density-'] != 'なし':
                img = Contrast_cvt(img, values['-Color_Density-'])

            # -----------------------------------
            # 画面上に撮影した画像を表示する
            # -----------------------------------
            img = scale_box(img, self.Image_width, self.Image_height)
            imgbytes = cv2.imencode('.png', img)[1].tobytes()
            self.Window['-IMAGE-'].update(data=imgbytes)

            # ------------------------------------------------
            # 画面上に撮影した画像のヒストグラムを表示する
            # ------------------------------------------------
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

        # elif event == '-Contours-':

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


def WebCam_OnOff(device_num: int, cam: cv2.VideoCapture = None) -> cv2.VideoCapture:
    """
    WebCameraを読み込む関数

    Args:
        device_num (int):
            カメラデバイスを番号で指定
            0: PC内臓カメラ
            1: 外部カメラ
        cam (cv2.VideoCapture):
            接続しているカメラ情報

    Return:
        response (int):
            動作終了を表すフラグ
            0: カメラを開放した
            1: カメラに接続した
            -1: エラー
        cam (cv2.VideoCapture):
            接続したデバイス情報を返す
    """
    if cam is None:  # カメラが接続されていないとき
        cam = cv2.VideoCapture(device_num)
        # カメラに接続できなかった場合
        if not cam.isOpened():
            return -1, None
        # 接続できた場合
        else:
            return 1, cam

    else:  # カメラに接続されていたとき
        cam.release()
        return 0, None


def Preview(
    img: np.ndarray = None, window_name: str = "frame", preview: str = "cv2"
) -> int:
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

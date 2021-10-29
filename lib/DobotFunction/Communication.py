# cording: ustf-8
import sys, os

from matplotlib.pyplot import pink
from numpy import dtype

sys.path.append(".")
sys.path.append("..")
sys.path.append("../../")

import csv
import time
import traceback

from lib.DobotDLL import DobotDllType as dType
from timeout_decorator import timeout, TimeoutError


CON_STR = {
    dType.DobotConnect.DobotConnect_NoError: "DobotConnect_NoError",
    dType.DobotConnect.DobotConnect_NotFound: "DobotConnect_NotFound",
    dType.DobotConnect.DobotConnect_Occupied: "DobotConnect_Occupied",
}

HomePoint = {"x": 200, "y": 0, "z": 0, "r": 0}

# 関節座標系における各モータの速度および加速度の初期値
ptpJointParams = {
    "j1Velocity": 200,
    "j1Acceleration": 200,
    "j2Velocity": 200,
    "j2Acceleration": 200,
    "j3Velocity": 200,
    "j3Acceleration": 200,
    "j4Velocity": 200,
    "j4Acceleration": 200,
}

# デカルト座標系における各モータの速度および加速度の初期値
ptpCoordinateParams = {
    "xyzVelocity": 200,
    "xyzAcceleration": 200,
    "rVelocity": 200,
    "rAcceleration": 200,
}

ptpMoveModeDict = {
    "JumpCoordinate": dType.PTPMode.PTPJUMPXYZMode,
    "MoveJCoordinate": dType.PTPMode.PTPMOVJXYZMode,
    "MoveLCoordinate": dType.PTPMode.PTPMOVLXYZMode,
}


# -----------------
# Dobotの初期化
# -----------------
def Connect_Disconnect(connection_flag: bool, api, CON_STR: tuple = CON_STR):
    """Dobotを接続状態を制御する関数

    Args:
        connection_flag(bool): 現在Dobotが接続されているか。
        api(dType): Dobot API
        CON_STR(tuple): 接続時のエラーテーブル

    Returns:
        rec(bool): 接続結果
            True: 接続した
            False: 接続していない
        Dobot_Err(int): 処理結果
    """
    rec = False
    err = 0

    # Dobotがすでに接続されていた場合
    if connection_flag:
        dType.DisconnectDobot(api)  # DobotDisconnect
        print("Dobotとの接続を解除しました．")
        rec, err = False, 1

    # Dobotが接続されていない場合
    else:
        portName = dType.SearchDobot(api, maxLen=128)
        try:
            # if ("COM3" in portName) or ("COM4" in portName):
            if "COM3" in portName:
                char_size = 115200
                state = dType.ConnectDobot(api, "COM3", char_size)
                # 接続時にエラーが発生しなかった場合
                if CON_STR[state[0]] == "DobotConnect_NoError":
                    print("Dobotに通信速度 {} で接続されました．".format(char_size))
                    initDobot(api)  # Dobotの初期設定を行う
                    rec, err = True, state[0]

                elif CON_STR[state[0]] == "DobotConnect_NotFound":
                    print("Dobot を見つけることができません！")
                    rec, err = False, state[0]

                elif CON_STR[state[0]] == "DobotConnect_Occupied":
                    print(
                        "Dobot が占有されています。接続するには、アプリケーションを再起動する、Dobot 背面の Reset ボタンを押す、接続USBケーブルを再接続する、のどれかを行う必要があります。"
                    )
                    rec, err = False, state[0]
                else:
                    dType.DisconnectDobot(api)
                    raise Exception("接続時に予期せぬエラーが発生しました！！")
            else:
                raise Exception(
                    "Dobotがハードウェア上で接続されていない可能性があります。接続されている場合は、portNameに格納されている変数を確認してください。"
                )
        except Exception as e:
            print(e)
        finally:
            return rec, err


def initDobot(api):
    dType.SetCmdTimeout(api, 3000)  # TimeOut Setup
    dType.SetQueuedCmdClear(api)  # Clean Command Queued
    dSN = dType.GetDeviceSN(api)  # デバイスのシリアルナンバーを取得する
    print(dSN)
    dName = dType.GetDeviceName(api)  # デバイス名を取得する
    print(dName)
    # majorV, minorV, revision = dType.GetDeviceVersion(api)  # デバイスのバージョンを取得する
    # print(majorV, minorV, revision)

    # Home Params の設定
    dType.SetHOMEParams(
        api, HomePoint["x"], HomePoint["y"], HomePoint["z"], HomePoint["r"], isQueued=1
    )  # Async Motion Params Setting
    dType.SetHOMECmd(api, temp=0, isQueued=1)  # Async Home

    # JOGパラメータの設定
    dType.SetJOGJointParams(
        api, 200, 200, 200, 200, 200, 200, 200, 200, isQueued=1
    )  # 関節座標系での各モータの速度および加速度の設定
    dType.SetJOGCoordinateParams(
        api, 200, 200, 200, 200, 200, 200, 200, 200, isQueued=1
    )  # デカルト座標系での各方向への速度および加速度の設定
    # JOG動作の速度、加速度の比率を設定
    dType.SetJOGCommonParams(api, 100, 100, isQueued=1)

    # ----------------------- #
    # PTPパラメータの設定 #
    # ----------------------- #
    # 関節座標系の各モータの速度および加速度を設定
    dType.SetPTPJointParams(
        api=api,
        j1Velocity=ptpJointParams["j1Velocity"],
        j1Acceleration=ptpJointParams["j1Acceleration"],
        j2Velocity=ptpJointParams["j2Velocity"],
        j2Acceleration=ptpJointParams["j2Acceleration"],
        j3Velocity=ptpJointParams["j3Velocity"],
        j3Acceleration=ptpJointParams["j3Acceleration"],
        j4Velocity=ptpJointParams["j4Velocity"],
        j4Acceleration=ptpJointParams["j4Acceleration"],
        isQueued=1,
    )

    params = dType.GetPTPJointParams(api)
    s = """\
    起動直後のMagicianの関節座標系における各モータの速度および加速度
    j1Velocity = {}
    j1Acceleration = {}
    j2Velocity = {}
    j2Acceleration = {}
    j3Velocity = {}
    j3Acceleration = {}
    j4Velocity = {}
    j4Acceleration = {}
    """
    print(
        s.format(
            params[0],
            params[1],
            params[2],
            params[3],
            params[4],
            params[5],
            params[6],
            params[7],
        )
    )

    # デカルト座標系での各方向への速度および加速度の設定
    dType.SetPTPCoordinateParams(
        api=api,
        xyzVelocity=ptpCoordinateParams["xyzVelocity"],
        xyzAcceleration=ptpCoordinateParams["xyzAcceleration"],
        rVelocity=ptpCoordinateParams["rVelocity"],
        rAcceleration=ptpCoordinateParams["rAcceleration"],
        isQueued=1,
    )

    params = dType.GetPTPCoordinateParams(api)
    s = """\
    起動直後のMagicianのデカルト座標系における各モータの速度および加速度
    xyzVelocity = {}
    xyzAcceleration = {}
    rVelocity = {}
    rAcceleration = {}
    """
    print(s.format(params[0], params[1], params[2], params[3]))

    # PTP動作の速度、加速度の比率を設定
    lastIndex = dType.SetPTPCommonParams(api, 100, 100, isQueued=1)[0]

    # Wait for Executing Last Command
    while lastIndex > dType.GetQueuedCmdCurrentIndex(api)[0]:
        pass
    return None


# -----------------------------------
# Dobotの動作用_汎用関数
# -----------------------------------
# 直交座標系での動作
def Operation(api, file_name, axis, volume=1, initPOS=None):
    """
    A function that sends a motion command in any direction

    Parameters
    ----------
    api : CDLL
    axis : str
        移動方向
    volume : int
        移動量
    """
    axis_list = ["x", "y", "z", "r"]
    if initPOS != None:
        pose = initPOS
    else:
        pose = dType.GetPose(api)

    if axis in axis_list:
        if axis == "x":
            _OneAction(api, pose[0] + volume, pose[1], pose[2], pose[3])
        elif axis == "y":
            _OneAction(api, pose[0], pose[1] + volume, pose[2], pose[3])
        elif axis == "z":
            _OneAction(api, pose[0], pose[1], pose[2] + volume, pose[3])
        else:
            print("rは実装されていません。")
    else:
        print("移動軸に問題があります！")

    # 座標をファイルへ書き込む
    csv_write(file_name, dType.GetPose(api))

    # 1回動作指令を出す関数
    # def _OneAction(api, x=None, y=None, z=None, r=None, mode=dType.PTPMode.PTPMOVLXYZMode):
    """One step operation"""


#    if x is None or y is None or z is None or r is None:
#        pose = dType.GetPose(api)
#        if x is None: x = pose[0]
#        if y is None: y = pose[1]
#        if z is None: z = pose[2]
#        if r is None: r = pose[3]
#    try:
#        lastIndex = dType.SetPTPCmd(api, mode, x, y, z, r, isQueued=1)[0]
#        _Act(api, lastIndex)
#    except TimeoutError: return -1
#    else: return 0


def _OneAction(api, pose, mode=dType.PTPMode.PTPMOVLXYZMode):
    """One step operation"""
    # if pose["x"] == None or y is None or z is None or r is None:
    current_pose = dType.GetPose(api)
    if pose["x"] is None:
        pose["x"] = current_pose[0]
    if pose["y"] is None:
        pose["y"] = current_pose[1]
    if pose["z"] is None:
        pose["z"] = current_pose[2]
    if pose["r"] is None:
        pose["r"] = current_pose[3]
    try:
        lastIndex = dType.SetPTPCmd(
            api, mode, pose["x"], pose["y"], pose["z"], pose["r"], isQueued=1
        )[0]
        __Act(api, lastIndex)
    except TimeoutError:
        return -1
    else:
        return 0


def __Act(api, lastIndex):
    """Function to execute command"""
    # キューに入っているコマンドを実行
    dType.SetQueuedCmdStartExec(api)

    # Wait for Executing Last Command
    while lastIndex > dType.GetQueuedCmdCurrentIndex(api)[0]:
        pass

    # キューに入っているコマンドを停止
    dType.SetQueuedCmdStopExec(api)


def SetPoseAct(api, pose: dict, ptpMoveMode: str, queue_index: int = 1):
    """デカルト座標系で指定された位置にアームの先端を移動させる関数

    Arg:
        api(dtype): DobotAPIのコンストラクタ
        pose(dict): デカルト座標系および関節座標系で指定された姿勢データ
        ptpMoveMode(str): 各座標系におけるDobotの制御方法
        queue_index(int): データをQueueとして送るか。default to 0
        * 0: 送らない
        * 1: 送る

    Return:
        response(int):
            0 : 応答あり
            1 : 応答なし
    """
    response = ""
    try:
        for ptpmode in ptpMoveModeDict:
            if ptpMoveMode in ptpmode:
                response = ptpMoveModeDict[ptpMoveMode]
                break
        if (not response) and (response != 0):
            raise ValueError("指定された制御方法は存在しません。")
    except (ValueError, TypeError) as e:
        traceback.print_exc()
    else:
        lastIndex = dType.SetPTPCmd(
            api,
            ptpMoveModeDict[ptpMoveMode],
            pose["x"],
            pose["y"],
            pose["z"],
            pose["r"],
            queue_index,
        )[0]
        while lastIndex > dType.GetQueuedCmdCurrentIndex(api)[0]:
            pass
    return 0


def GripperAutoCtrl(api) -> None:
    """DobotAPIの出力に基づいてグリッパを自動的に開閉制御する関数

    Args:
        api (dType): DobotAPIのコンストラクタ

    Return:
        None
    """
    sleep_time = 0.7  # [second]
    [value] = dType.GetEndEffectorGripper(api)
    # --------------------------- #
    # グリッパが閉じている場合 #
    # --------------------------- #
    if value:
        # モータを起動する
        _GripperOpenClose(api, motorCtrl=True, gripperCtrl=True)
        # グリッパを開く
        _GripperOpenClose(api, motorCtrl=True, gripperCtrl=False)
        time.sleep(sleep_time)
        # モータを停止する
        _GripperOpenClose(api, motorCtrl=False, gripperCtrl=False)
    # --------------------------- #
    # グリッパが開いている場合 #
    # --------------------------- #
    else:
        # モータを起動する
        _GripperOpenClose(api, motorCtrl=True, gripperCtrl=False)
        # グリッパを閉じる
        _GripperOpenClose(api, motorCtrl=True, gripperCtrl=True)
        time.sleep(sleep_time)
        # モータを停止する
        _GripperOpenClose(api, motorCtrl=False, gripperCtrl=True)


def _GripperOpenClose(
    api, motorCtrl: bool = False, gripperCtrl: bool = True, queue_index: int = 0
) -> None:
    """グリッパーを開閉する関数

    Args:
        api(dType): DobotAPIのコンストラクタ
        motorCtrl(bool optional): 吸引モータを起動する。default to True
            * True:  On
            * False: Off
        gripperCtrl(bool optional): グリッパを開閉する。default to True
            True:  Close
            False: Open
        queue_index(int optional): データをQueueとして送るか。default to 0
            * 0: 送らない
            * 1: 送る

    Return: None
    """
    lastIndex = dType.SetEndEffectorGripper(api, motorCtrl, gripperCtrl, queue_index)[0]
    while lastIndex > dType.GetQueuedCmdCurrentIndex(api)[0]:
        pass


# ----------------------------------
# CsvFileへの書き込み関数
# ----------------------------------
def csv_write(filename, data):
    """Write Data to csv file"""
    if data is not None:  # 書き込むデータが無いとき
        return
    array = [str(row) for row in data]
    with open(filename, "a", encoding="utf_8", errors="", newline="") as f:
        # ファイルへの書き込みを行う
        if _wirte(f, array) is not None:
            print("x=%f,  y=%f,  z=%f,  r=%f" % (data[0], data[1], data[2], data[3]))
            # print('書き込みが完了しました。')
        else:
            print("ファイルの書き込みに失敗しました。")


def _wirte(f, data):
    """write content"""
    error = 1  # エラーチェック用変数
    witer = csv.writer(f, lineterminator="\n")
    error = witer.writerows([data])

    return error  # エラーが無ければNoneを返す


if __name__ == "__main__":
    from DobotDLL import DobotDllType as dType
    from src.config.config import cfg

    dll_path = cfg.DOBOT_DLL_DIR + os.sep + "DobotDll.dll"
    api = dType.load(dll_path)
    CON_STR = {
        dType.DobotConnect.DobotConnect_NoError: "DobotConnect_NoError",
        dType.DobotConnect.DobotConnect_NotFound: "DobotConnect_NotFound",
        dType.DobotConnect.DobotConnect_Occupied: "DobotConnect_Occupied",
    }

    # ---------------------------- #
    # Dobot の接続ポートの確認 #
    # ---------------------------- #
    # portName = dType.SearchDobot(api, maxLen=128)
    # print(portName)

    # ------------------- #
    # Dobot のコネクト #
    # ------------------- #
    connection_flag = False

    connection_flag, result = Connect_Disconnect(connection_flag, api, CON_STR)
    if connection_flag:
        [value] = dType.GetEndEffectorSuctionCup(api)
        print(value)

    """
    # グリッパ: 閉、モータ: OFF -> 1
    [value] = dType.GetEndEffectorGripper(api)
    print("Gripper CLOSE, Motor OFF: {}".format(value))
    # グリッパ: 開、モータ: ON -> 0
    result = _GripperOpenClose(api, motorCtrl=True, gripperCtrl=False)
    time.sleep(5)
    [value] = dType.GetEndEffectorGripper(api)
    print("Gripper OPEN Motor ON: {}".format(value))
    # グリッパ: 開、モータ: OFF -> 0
    result = _GripperOpenClose(api, motorCtrl=False, gripperCtrl=False)
    [value] = dType.GetEndEffectorGripper(api)
    print("Gripper OPEN Motor OFF: {}".format(value))
    # グリッパ: 閉、モータ: ON -> 1
    result = _GripperOpenClose(api, motorCtrl=True, gripperCtrl=True)
    time.sleep(5)
    [value] = dType.GetEndEffectorGripper(api)
    print("Gripper CLOSE Motor ON: {}".format(value))
    # グリッパ: 閉、モータ: OFF -> 1
    result = _GripperOpenClose(api, motorCtrl=False, gripperCtrl=True)
    [value] = dType.GetEndEffectorGripper(api)
    print("Gripper CLOSE Motor OFF: {}".format(value))
    """

    # GripperAutoCtrl(api)
    # GripperAutoCtrl(api)

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
    # ptpMoveMode = "MoveJCoordinate"

    SetPoseAct(api, pose, ptpMoveMode)

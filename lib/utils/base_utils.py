import datetime as dt
import os
import json
from typing import Dict, Literal

import cv2
import numpy as np


def ReadJsonToDict(r_json_pth: str) -> Dict:
    """
    Ndjson ファイルからデータを読みだす関数．
    json は，保存したい全てのデータを一度読みだしておく必要があるが ndjson は，1つ1つのデータを順次保存することができる．
    ndjson の使い方については以下を参照．
    REF: https://qiita.com/eg_i_eg/items/aff02f6057b476cb15fa

    Arg:
        r_json_pth (str): データを読みだす `JSON` ファイルへのパス．

    Return:
        data (Dict): 読みだしたデータ．
    """
    # path に .json が含まれていなければ追加
    fp = r_json_pth
    if ".json" not in os.path.splitext(fp)[-1]:
        fp = fp + ".json"
    # ファイルへの書き込み
    with open(fp, "r") as f:
        data = json.load(f)

    return data


def save_img(img: np.ndarray, dir_name: str, ext: Literal["png", "jpg", "bmp"] = "png") -> int:
    """
    画像保存用の関数．

    Args:
        img (np.ndarray): 保存用の画像．
        dir_name (str): 画像の保存先のディレクトリ．
        ext (Literal[, optional): 保存する際の拡張子. Defaults to "png".

    Returns:
        int: 保存のフラグ
    """
    # 画像が ndarray 配列か確認
    if type(img) != np.ndarray:
        return -1
    # ディレクトリが存在するか確認
    if not os.path.isdir(dir_name):
        return -2
    # 画像保存用の拡張子の選択
    if ext == "png": ext = ".png"
    elif ext == "jpg": ext = ".jpg"
    elif ext == "bmp": ext = ".bmp"
    else: return -3

    # 現在時刻を取得
    dt_now = dt.datetime.now()
    # フォルダ名用にyyyymmddの文字列を取得する
    today = dt_now.strftime("%Y%m%d%H%M")
    filepth = os.path.join(dir_name, today+ext)

    cv2.imwrite(filepth, img)


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


def WriteDataToJson(data: Dict, wt_json_pth: str):
    """
    データを追記する形で JSON ファイルに出力する関数．
    json は，保存したい全てのデータを一度読みだしておく必要があるが ndjson は，1つ1つのデータを順次保存することができる．
    ndjson の使い方については以下を参照．
    REF: https://qiita.com/eg_i_eg/items/aff02f6057b476cb15fa

    Args:
        data (Dict): json ファイルに出力するデータ．
        wt_json_pth (str): データを書き込む `JSON` ファイルへのパス．
    """
    # path に .json が含まれていなければ追加
    fp = wt_json_pth
    if ".json" not in os.path.splitext(fp)[-1]:
        fp = fp + ".json"
    # ファイルへの書き込み
    with open(fp, "a") as f:
        json.dump(data, f, ensure_ascii=False)
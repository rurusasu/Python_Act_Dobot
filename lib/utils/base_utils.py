import os
import json
import shutil
from typing import Dict

import cv2
import numpy as np


def makedir(filepth: str):
    """
    ディレクトリが存在しない場合に、深い階層のディレクトリまで再帰的に作成する関数。もし、ディレクトリが存在する場合は一度中身ごと削除してから再度作成。

    Arg:
        filepth (str): 作成するディレクトリのパス
    """
    if not os.path.isdir(filepth):
        os.makedirs(filepth, exist_ok=True)
    else:
        shutil.rmtree(filepth)
        os.makedirs(filepth, exist_ok=True)


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
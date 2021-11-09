import os
import json
from typing import Dict


def ReadNdjsonToDict(r_json_pth: str) -> Dict:
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


def WriteDataToNdjson(data: Dict, wt_json_pth: str):
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

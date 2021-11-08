import os
from typing import Dict

import ndjson


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
        writer = ndjson.writer(f)
        writer.writerow(data)

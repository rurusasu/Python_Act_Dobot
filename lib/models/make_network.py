import os
import sys
from typing import Tuple, Union

sys.path.append(".")
sys.path.append("..")
sys.path.append("../../")

import torch
import torch.nn as nn

from lib.models.cnns.get_cnn import GetCNN


_network_factory = {
    "cnns": GetCNN,
}


def make_network(net_name: str, num_classes: int) -> nn:
    """Network を読みだす関数

    Args:
        net_name (str): 読みだすネットワークの名前．
        num_classes (int): 分類するクラス数．

    Returns:
        torch.nn: ネットワークモデル
    """
    get_network_fun = _network_factory["cnns"]
    network = get_network_fun(net_name, num_classes)

    return network


def transfer_network(
    network, num_classes: int, replaced_layer_num: int = 1
) -> Tuple[nn.Sequential, int]:
    """
    転移学習用にモデルの全結合層を未学習のものに置き換える関数

    Args:
        network: make_network で読みだされたモデル
        num_classes (int): 転移学習時のクラス数．
        replaced_layer_num (int, optional): 置き換えたい全結合層の数．出力層のみ転移学習時のクラス数に置き換え，それ以外は weight を xavier の初期値で，bias を [0, 1] の一様分布の乱数で初期化する．Default to 1.

    Return:
        network (torch.nn.Sequential): 出力数と必要に応じて複数の全結合層の重みが初期化されたネットワーク．
        replaced_layer_num (int): 実際に重みが初期化された層の数．
    """
    if num_classes < 1:
        raise (
            "num_classes error. This value should be given as a positive integer greater than zero."
        )
    if replaced_layer_num < 1:
        raise (
            "relpaced layer number error. This value should be given as a positive integer greater than zero."
        )

    # 最終出力層の重みを初期化したかを判定するフラグ
    # False: 初期化した。
    out_layer_replaced = True
    iter = 0
    if hasattr(network, "classifier"):
        # classifier が複数の層から構成されている場合（例: AlexNet, VGG）
        if issubclass(type(network.classifier), nn.Sequential):
            for key, layer in reversed(network.classifier._modules.items()):
                if issubclass(type(layer), nn.Linear):
                    # 置き換えたい全結合層の数と実際に置き換えた全結合層の数が同数になった場合，break.
                    if iter > replaced_layer_num:
                        break
                    if out_layer_replaced:
                        in_features = layer.in_features  # 全結合層への既存の入力サイズを取得
                        fc = nn.Linear(
                            in_features, out_features=num_classes
                        )  # 訓練させたいクラス数に応じて出力サイズを変更
                        network.classifier._modules[key] = fc  # 全結合層を置き換え
                        out_layer_replaced = False  # フラグを変更
                        iter += 1
                    else:
                        nn.init.xavier_uniform_(layer.weight)
                        nn.init.uniform_(layer.bias, 0, 1)  # 一様分布
                        iter += 1

            return network, iter
        else:
            # EfficientNet の場合
            num_ftrs = network.classifier.in_features
            fc = nn.Linear(num_ftrs, out_features=num_classes)
            network.classifier = fc
            return network, 1
    # ResNet の場合
    elif hasattr(network, "fc"):
        num_ftrs = network.fc.in_features
        fc = nn.Linear(num_ftrs, out_features=num_classes)
        network.fc = fc
        return network, 1
    # IncResNetV2 の場合
    elif hasattr(network, "classif"):
        num_ftrs = network.classif.in_features
        fc = nn.Linear(num_ftrs, out_features=num_classes)
        network.classif = fc
        return network, 1


def ReadCNNWeights(network: torch.nn, cnn_pth: str):
    """事前学習したネットワークの重みを読みだす関数．

    Args:
        network ([type]): 重みを入れ込みたいネットワークの構造体
        cnn_pth (str): 保存した重みへのパス

    Returns:
        network (None | torch.nn):
            重みの読み出しに成功した場合: torch.nn
            失敗した場合: None
    """
    if not os.path.exists(cnn_pth):
        return None

    pretrained_model = torch.load(cnn_pth)
    network.load_state_dict(pretrained_model["net"])
    return network

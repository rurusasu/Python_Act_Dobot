import sys

sys.path.append(".")
sys.path.append("../../../")

from lib.models.cnns.alexnet import get_alex_net as get_alex
from lib.models.cnns.efficientnet import get_efficient_net as get_eff
from lib.models.cnns.inception import get_inception_net as get_inc
from lib.models.cnns.resnet import get_res_net as get_res
from lib.models.cnns.vgg import get_vgg_net as get_vgg

_network_factory = {
    "alex": get_alex,
    "eff": get_eff,
    "inc": get_inc,
    "res": get_res,
    "vgg": get_vgg,
}


def GetCNN(net_name, num_classes):

    model_num = -1
    arch = net_name
    if "_" in arch:
        model_num = str(arch[arch.find("_") + 1 :]) if "_" in arch else 0
        arch = arch[: arch.find("_")]

    if arch not in _network_factory:
        raise ValueError(f"The specified cfg.network={arch} does not exist.")
    if int(num_classes) <= 0:
        raise ValueError(f"The specified num_classes: {num_classes} does not exist.")
    get_model = _network_factory[arch]

    model = get_model(model_num, pretrained=False, num_classes=int(num_classes))

    return model

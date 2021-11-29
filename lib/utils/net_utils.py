import albumentations as albm
import numpy as np
import torch


def Numpy2Tensor(x: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(x.astype(np.float32)).clone()
    x = x.permute(2, 0, 1)
    x = x.unsqueeze(dim=0)
    return x


def Tensor2Int(x: torch.Tensor) -> int:
    x = x.to("cpu").detach().numpy().copy()
    x = x.item()
    return x


def CropImage(img: np.ndarray, mask: np.ndarray):
    transform = albm.Compose(
        [
            albm.CropNonEmptyMaskIfExists(
                height=227,
                width=227,
            )
        ]
    )
    transformd_img = transform(image=img, mask=mask)
    return transformd_img["image"]


def predict(img: np.ndarray, mask_img, network: torch.nn):
    # マスクされた部分の画像を切り取る．
    crop_img = CropImage(img, mask=mask_img)
    # numpy -> tensor
    input = Numpy2Tensor(x=crop_img.copy())

    # ネットワークで姿勢推定
    output = network(input)
    _, preds = torch.max(output, axis=1)

    preds = Tensor2Int(preds)

    return crop_img, preds

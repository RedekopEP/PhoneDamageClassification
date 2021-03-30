import segmentation_models_pytorch as smp
from losses import LossBinaryDice, FocalTverskyLoss, FocalLoss, TverskyLoss, GeneralizedDiceLoss, LossBinary
from UNet import UNet2D


def init(config):
    # ---- Model Initialization  ----
    if config["model"] == "UNet":
        model = smp.Unet(activation=None) #UNet2D(n_channels=3, n_classes=1) # #UNet2D(n_channels=1, n_classes=1) #smp.Unet(activation=None)
    elif config["model"] == "PSPNet":
        model = smp.PSPNet(activation=None)
    elif config["model"] == "FPN":
        model = smp.FPN(activation=None)
    elif config["model"] == "Linknet":
        model = smp.Linknet(activation=None)
    else:
        raise Exception('Incorrect model name!')

    # ---- Loss Initialization  ----
    if config["mode"] == 'train':
        if config["loss"] == "DiceBCE":
            loss = LossBinaryDice(dice_weight=config["dice_weight"])
        elif config["loss"] == "FocalTversky":
            loss = FocalTverskyLoss()
        elif config["loss"] == "Focal":
            loss = FocalLoss()
        elif config["loss"] == "Tversky":
            loss = TverskyLoss()
        else:
            raise Exception('Incorrect loss name!')

        return model, loss
    else:
        return model


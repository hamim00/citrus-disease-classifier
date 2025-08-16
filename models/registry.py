import os, json
import torch
import torch.nn as nn
from torchvision import models as tv
from .custom_cnn import CustomCNN
from .scnn import SCNN

def _replace_classifier(m, num_classes):
    # Works for most torchvision models
    if hasattr(m, "fc") and isinstance(m.fc, nn.Linear):
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, num_classes)
    elif hasattr(m, "classifier"):
        if isinstance(m.classifier, nn.Linear):
            in_f = m.classifier.in_features
            m.classifier = nn.Linear(in_f, num_classes)
        elif isinstance(m.classifier, nn.Sequential):
            for i in range(len(m.classifier)-1, -1, -1):
                if isinstance(m.classifier[i], nn.Linear):
                    in_f = m.classifier[i].in_features
                    m.classifier[i] = nn.Linear(in_f, num_classes)
                    break
    return m

def load_class_names(path="weights/class_names.json"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Run the split script to generate it.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_model(arch: str, num_classes: int, pretrained: bool = True):
    arch = arch.lower()
    if arch == "custom_cnn":
        model = CustomCNN(num_classes); target_layer = model.cam_target_layer
    elif arch == "s_cnn":
        model = SCNN(num_classes); target_layer = model.cam_target_layer
    elif arch == "resnet18":
        m = tv.resnet18(weights=tv.ResNet18_Weights.DEFAULT if pretrained else None)
        model = _replace_classifier(m, num_classes); target_layer = model.layer4[-1]
    elif arch == "mobilenet_v3_small":
        m = tv.mobilenet_v3_small(weights=tv.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        model = _replace_classifier(m, num_classes); target_layer = model.features[-1]
    elif arch == "efficientnet_b0":
        m = tv.efficientnet_b0(weights=tv.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        model = _replace_classifier(m, num_classes); target_layer = model.features[-1]
    else:
        raise ValueError(f"Unknown arch: {arch}")
    return model, target_layer

# Fill in your checkpoint filenames when ready
MODEL_CATALOG = [
    {"name": "Custom CNN",              "arch": "custom_cnn",        "ckpt": "weights/custom_cnn.pt",        "pretrained": False},
    {"name": "S-CNN (shallow)",         "arch": "s_cnn",             "ckpt": "weights/scnn.pt",              "pretrained": False},
    {"name": "ResNet18 (transfer)",     "arch": "resnet18",          "ckpt": "weights/resnet18_finetuned.pt","pretrained": True},
    {"name": "MobileNetV3-Small (IMN)", "arch": "mobilenet_v3_small","ckpt": None,                           "pretrained": True},
    {"name": "EfficientNet-B0 (IMN)",   "arch": "efficientnet_b0",   "ckpt": None,                           "pretrained": True},
]

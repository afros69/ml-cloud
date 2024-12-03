import cv2
import torch
from torch import nn
from torchvision import models, transforms
import albumentations as A

from consts import color_model_path

device = "cuda" if torch.cuda.is_available() else "cpu"

best_model_wts = models.efficientnet_b3(pretrained=True)
best_model_wts.classifier[1] = nn.Linear(in_features=1536, out_features=16, bias=True)
best_model_wts.load_state_dict(torch.load(color_model_path), strict=False)
best_model_wts = best_model_wts.to(device)
best_model_wts = best_model_wts.eval()

data_transforms_val = transforms.Compose([
        transforms.Resize((60, 60)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
# TODO: value ???
padding = A.PadIfNeeded(min_height=60, min_width=60, border_mode=cv2.BORDER_CONSTANT, value=cv2.BORDER_CONSTANT)

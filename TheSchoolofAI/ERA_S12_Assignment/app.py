# -*- coding: utf-8 -*-
"""app.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1EnkEPQ2DkOV1xYxsgIYBm9we2CHGAU_D
"""

# from google.colab import drive
# drive.mount('/content/drive')

# !pip install grad-cam

# !pip install torchsummary
# !pip install pytorch-lightning
# !pip install gradio

# !cp /content/drive/MyDrive/The\ School\ of\ AI/Session\ 12\ Assignment/model.py /content
# !cp /content/drive/MyDrive/The\ School\ of\ AI/Session\ 12\ Assignment/dataload.py /content
# !cp /content/drive/MyDrive/The\ School\ of\ AI/Session\ 12\ Assignment/utils.py /content

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np
import os
import albumentations as A
import cv2
import gradio as gr
import matplotlib.pyplot as plt
import model, dataload, utils

from io import BytesIO
from pathlib import Path
from random import shuffle

from __future__ import print_function
from albumentations.pytorch import ToTensorV2
from albumentations import Normalize

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.model_summary import ModelSummary
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torchmetrics import Accuracy
from torchsummary import summary

from model import LitCustomResNet
from dataload import *
from utils import *

from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms as T
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

mean = (0.49139968, 0.48215841, 0.44653091)
std = (0.24703223, 0.24348513, 0.26158784)
transforms = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])
classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
BATCH_SIZE = 512
softmax = torch.nn.Softmax(dim=0)

class Cifar10SearchDataset(torchvision.datasets.CIFAR10):
    def __init__(self, root="~/data/cifar10", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

## Load the model
LitCustomResNet = torch.load('/content/drive/MyDrive/The School of AI/Session 12 Assignment/LitCustomResNet_v1.pt',map_location=torch.device('cpu'))

LitCustomResNet

def get_traget_layer(block: str, layer: int):
    layer_num = 0 if layer == 0 else -1
    if block == "block1":
        return LitCustomResNet.resblock1[layer_num]
    if block == "block2":
        return LitCustomResNet.resblock2[layer_num]
    if block == "block3":
        return LitCustomResNet.resblock3[layer_num]
    if block == "block4":
        return LitCustomResNet.resblock4[layer_num]

#default_cam = GradCAM(model=model, target_layers=[get_traget_layer("block4", -1)])
default_cam = GradCAM(model=LitCustomResNet, target_layers=[get_traget_layer("block4", -1)])

def predict_img(img: np.ndarray, top_k: int = 10):
    preds = LitCustomResNet(img)
    preds = softmax(preds.flatten())
    preds = {classes[i]: float(preds[i]) for i in range(10)}
    preds = {
        k: v for k, v in sorted(preds.items(), key=lambda item: item[1], reverse=True)[:top_k]
    }

    return preds

def display_cam(cam: GradCAM, org_img: np.ndarray, img: torch.Tensor, transparency: float):
    grayscale_cam = cam(input_tensor=img, targets=None)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(
        org_img / 255, grayscale_cam, use_rgb=True, image_weight=transparency
    )
    return visualization

misclassified_images = []
import cv2
import numpy as np

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

def de_normalize(img):
    mean = (0.4914, 0.4822, 0.4471)
    std = (0.2469, 0.2433, 0.2615)
    img = img.numpy().astype(dtype=np.float32)

    for i in range(img.shape[0]):
         img[i] = (img[i]*std[i])+mean[i]

    return np.transpose(img, (1,2,0))

def save_misclassified(test_loader,model,device):
  counter = 0
  unique_check = {}
  if not os.path.exists("/mis_classified_images"):
    os.mkdir("mis_classified_images/")
  with torch.no_grad():
      for data, target in test_loader:
          data, target = data.to(device), target.to(device)
          output = model(data)
          _, pred = torch.max(output, 1)
          for i in range(len(pred)):
              if pred[i] != target[i]:
                  misclassified_images.append(data[i])
                  npimg = de_normalize(data[i].cpu())
                  op_img = (npimg*255).astype(np.uint8)
                  targeted_class = classes[target[i]]
                  if (targeted_class not in unique_check):
                    unique_check[targeted_class] = 1
                    file_name = f"Actual_{classes[target[i]]}_Pred_{classes[pred[i]]}_{counter}.jpg"
                    cv2.imwrite("/content/mis_classified_images/"+file_name,op_img)
                  elif (targeted_class in unique_check and unique_check[targeted_class] <= 4):
                    unique_check[targeted_class] += 1
                    file_name = f"Actual_{classes[target[i]]}_Pred_{classes[pred[i]]}_{counter}.jpg"
                    cv2.imwrite("/content/mis_classified_images/"+file_name,op_img)

          if len(unique_check)==10:
            break
  return unique_check

train_loader,test_loader = loader.load_data(BATCH_SIZE)
print(len(train_loader),len(test_loader))

device = "cpu"

save_misclassified(test_loader,LitCustomResNet,device)

misclf_path = "/content/mis_classified_images/"
mis_classified_imgs = list(Path(misclf_path).glob("*"))

from numpy.core.fromnumeric import shape
type(mis_classified_imgs), shape(mis_classified_imgs)

def make_image(p: Path | str, pred: str, label: str):
    im = cv2.imread(str(p))
    im = cv2.resize(im, (64, 64))

    plt.imshow(im)
    plt.title(f"{pred} / {label}")
    plt.axis("off")

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)

    img_array = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    buffer.close()

    # Decode the image array using OpenCV
    im = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return im

mean = (0.49139968, 0.48215841, 0.44653091)
std = (0.24703223, 0.24348513, 0.26158784)
transforms = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])

def inference(
    org_img: np.ndarray,
    top_k: int,
    show_cam: str,
    num_cam_imgs: int,
    cam_block: str,
    target_layer_num: int,
    transparency: float,
    show_misclf: str,
    num_misclf: int,
):
    input_img = transforms(org_img)
    input_img = input_img.unsqueeze(0)

    preds = predict_img(input_img, top_k)
    org_img = display_cam(default_cam, org_img, input_img, transparency)

    shuffle(mis_classified_imgs)
    #shuffle(misclassified_data)
    cam_outputs = []
    if show_cam:
        img_list = []

        target_layers = [get_traget_layer(cam_block, target_layer_num)]
        #cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
        cam = GradCAM(model=LitCustomResNet, target_layers=target_layers, use_cuda=False)
        for p in mis_classified_imgs[:num_cam_imgs]:
            im = cv2.imread(str(p))
            inp_im = transforms(im)
            inp_im = inp_im.unsqueeze(0)

            grayscale_cam = cam(input_tensor=inp_im, targets=None)

            grayscale_cam = grayscale_cam[0, :]
            visualization = show_cam_on_image(
                im / 255, grayscale_cam, use_rgb=True, image_weight=transparency
            )
            cam_outputs.append(visualization)

        del cam, img_list

    misclf_images_output = []
    if show_misclf:
        img_list = []
        gt = []
        for p in mis_classified_imgs[:num_misclf]:
            img_list.append(transforms(Image.open(p).convert("RGB")))
            gt.append(p.name.split("_")[0])

        misclf_out = softmax(LitCustomResNet(torch.stack(img_list))).argmax(dim=1).tolist()
        del img_list
        for imp, pred, label in zip(mis_classified_imgs[:num_misclf], misclf_out, gt):
            pred = classes[pred]
            misclf_images_output.append(make_image(imp, pred, label))

    return org_img, preds, cam_outputs, misclf_images_output

title = "CIFAR10 trained on Custom Model inspired by ResNet with GradCAM"
description = "A simple Gradio interface to infer on ResNet model, and get GradCAM results"
# examples = [["cat.jpg", 0.5, -1], ["dog.jpg", 0.5, -1]]
demo = gr.Interface(
    inference,
    inputs=[
        gr.Image(shape=(32, 32), label="Input Image"),
        gr.Slider(1, 10, value=3, step=1, label="Top K predictions"),
        gr.Checkbox(label="Show Grad Cam"),
        gr.Slider(1, 20, value=5, step=1, label="Number of images"),
        gr.Radio(label="Which Block?", choices=["block1", "block2", "block3","block4"]),
        gr.Slider(0, 1, value=1, step=1, label="Which Layer?"),
        gr.Slider(0, 1, value=0.5, label="Opacity of GradCAM"),
        gr.Checkbox(label="Show Misclassified Images"),
        gr.Slider(1, 20, value=5, step=5, label="Number of Misclassification Images"),
    ],
    outputs=[
        gr.Image(shape=(32, 32), label="Output", width=128, height=128),
        "label",
        gr.Gallery(label="GradCAM Output"),
        gr.Gallery(
            label="Misclassified Images Pred/G.T.",
            columns=[2],
            rows=[2],
            object_fit="contain",
            height="auto",
        ),
    ],
    title=title,
    description=description,
    # examples=examples,
)

demo.launch(debug=True)
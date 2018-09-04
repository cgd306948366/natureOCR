#coding:utf-8
import sys

import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable 
import numpy as np
import os
import util
import dataset
from PIL import Image
import models.crnn as crnn
import keys
from math import *
import cv2

# use affine to refine the boxes.
def dumpRotateImage(img,degree,pt1,pt2,pt3,pt4):
    height,width=img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
    matRotation=cv2.getRotationMatrix2D((width/2,height/2),degree,1)
    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2
    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    pt1 = list(pt1)
    pt3 = list(pt3)
    [[pt1[0]], [pt1[1]]] = np.dot(matRotation, np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(matRotation, np.array([[pt3[0]], [pt3[1]], [1]]))
    imgOut=imgRotation[int(pt1[1]):int(pt3[1]),int(pt1[0]):int(pt3[0])]
    height,width=imgOut.shape[:2]
    return imgOut

def crnnModel():
    alphabet = keys.alphabet
    # define a converter for convert rnn results to string
    converter = util.strLabelConverter(alphabet)
    model = crnn.CRNN(32, 1, len(alphabet)+1, 256, 1).cuda()
    path = './crnn/samples/CRNN.pth'
    model.load_state_dict(torch.load(path))
    return model,converter

def crnnRec(model,converter,im,text_recs):
   index = 0
   pred_all = ''
   for rec in text_recs:
       pt1 = (rec[0],rec[1])
       pt2 = (rec[2],rec[3])
       pt3 = (rec[6],rec[7])
       pt4 = (rec[4],rec[5])
       # refine the boxes so that the recognition model can more useful.
       partImg = dumpRotateImage(im,degrees(atan2(pt2[1]-pt1[1],pt2[0]-pt1[0])),pt1,pt2,pt3,pt4)
       #mahotas.imsave('%s.jpg'%index, partImg)
       # convert RGB to gray 
       image = Image.fromarray(partImg ).convert('L')
       scale = image.size[1]*1.0 / 32
       w = image.size[0] / scale
       w = int(w)
       #print(w)
       # notice the crnn input height is 32, width is any.
       transformer = dataset.resizeNormalize((w, 32))
       # use cuda
       image = transformer(image).cuda()
       # add batch axis
       image = image.view(1, *image.size())
       image = Variable(image)
       model.eval()
       preds = model(image)
       # easily know, the preds are batchs, steps, scores
       _, preds = preds.max(2)
       preds = preds.squeeze(1)
       preds = preds.transpose(-1, 0).contiguous().view(-1)
       preds_size = Variable(torch.IntTensor([preds.size(0)]))
       raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
       sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
       print(sim_pred)
       index = index + 1
       pred_all += sim_pred + '\n'
   return pred_all

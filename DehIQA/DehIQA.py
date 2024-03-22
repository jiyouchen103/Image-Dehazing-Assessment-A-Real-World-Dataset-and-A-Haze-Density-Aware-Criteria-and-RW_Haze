# This is a pytoch implementation of DISTS metric.
# Requirements: python >= 3.6, pytorch >= 1.0
import cv2
import math
import numpy as np
import os,sys
import torch
from torchvision import models,transforms
import torch.nn as nn
import torch.nn.functional as F
import dataloader

def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2,3],keepdim=keepdim)

def DarkChannel(im,sz):
    b,g,r = cv2.split(im)
    dc = cv2.min(cv2.min(r,g),b);
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    dark = cv2.erode(dc,kernel)
    return dark

def AtmLight(im,dark):
    [h,w] = im.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1))
    darkvec = dark.reshape(imsz);
    imvec = im.reshape(imsz,3);

    indices = darkvec.argsort();
    indices = indices[imsz-numpx::]

    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
       atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx;
    return A

def TransmissionEstimate(im,A,sz):
    omega = 0.95;
    im3 = np.empty(im.shape,im.dtype);

    for ind in range(0,3):
        im3[:,:,ind] = im[:,:,ind]/A[0,ind]

    transmission = 1 - omega*DarkChannel(im3,sz);
    return transmission

def Guidedfilter(im,p,r,eps):
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r));
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r));
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r));
    cov_Ip = mean_Ip - mean_I*mean_p;

    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r));
    var_I   = mean_II - mean_I*mean_I;

    a = cov_Ip/(var_I + eps);
    b = mean_p - a*mean_I;

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r));
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r));

    q = mean_a*im + mean_b;
    return q;

def TransmissionRefine(im,et):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);
    gray = np.float64(gray)/255;
    r = 60;
    eps = 0.0001;
    t = Guidedfilter(gray,et,r,eps);

    return t;

def Recover(im,t,A,tx = 0.1):
    res = np.empty(im.shape,im.dtype);
    t = cv2.max(t,tx);

    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]

    return res


class L2pooling(nn.Module):
    def __init__(self, filter_size=5, stride=2, channels=None, pad_off=0):
        super(L2pooling, self).__init__()
        self.padding = (filter_size - 2 )//2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        g = torch.Tensor(a[:,None]*a[None,:])
        g = g/torch.sum(g)
        self.register_buffer('filter', g[None,None,:,:].repeat((self.channels,1,1,1)))

    def forward(self, input):
        input = input**2
        out = F.conv2d(input, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
        return (out+1e-12).sqrt()

class DISTS(torch.nn.Module):
    def __init__(self, load_weights=True):
        super(DISTS, self).__init__()
        pretrained_features = models.resnet18(pretrained=True)
        num_ftrs = pretrained_features.fc.in_features
        pretrained_features.fc = torch.nn.Linear(num_ftrs, 2)
        pretrain_model = torch.load('./DehIQA.pth')
        new_state_dict = {}
        for k, v in pretrain_model.items():
            new_state_dict[k[7:]] = v
        pretrained_features.load_state_dict(new_state_dict)

        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()
        self.stage1.add_module(str(0), pretrained_features.conv1)
        self.stage1.add_module(str(1), pretrained_features.bn1)
        self.stage1.add_module(str(2), pretrained_features.relu)
        self.stage2.add_module(str(0), pretrained_features.maxpool)
        self.stage2.add_module(str(1), pretrained_features.layer1)
        self.stage3.add_module(str(0), pretrained_features.layer2)
        self.stage4.add_module(str(0), pretrained_features.layer3)
        self.stage5.add_module(str(0), pretrained_features.layer4)

        for param in self.parameters():
            param.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,-1,1,1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1,-1,1,1))
        self.weights = [1.0 / 3, 1.0 / 64, 1.0 / 128, 1.0 / 256, 1.0 / 512, 1.0 / 512]
        self.chns = [3,64,64,128,256,512]
        
    def forward_once(self, x):
        h = (x-self.mean)/self.std
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h
        return [x,h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]

    def forward(self, x, y, t, ratio, batch_average=True):

        score = 0
        for i in range(ratio):
            feats0 = self.forward_once(x[i])
            feats1 = self.forward_once(y[i])
            feats2 = self.forward_once(t[i])
            dist1 = 0
            dist2 = 0
            c1 = 1e-6
            c2 = 1e-6

            for k in range(len(self.chns)):
                feats0[k] = feats0[k] * (1-feats2[k])
                feats1[k] = feats1[k] * (1-feats2[k])
                x_mean = feats0[k].mean([2, 3], keepdim=True)
                y_mean = feats1[k].mean([2, 3], keepdim=True)
                S1 = (2 * x_mean * y_mean + c1) / (x_mean ** 2 + y_mean ** 2 + c1)
                dist1 = dist1 + (self.weights[k]*S1).sum(1, keepdim=True)

                x_var = ((feats0[k] - x_mean) ** 2).mean([2, 3], keepdim=True)
                y_var = ((feats1[k] - y_mean) ** 2).mean([2, 3], keepdim=True)
                xy_cov = (feats0[k] * feats1[k]).mean([2, 3], keepdim=True) - x_mean * y_mean
                S2 = (2 * xy_cov + c2) / (x_var + y_var + c2)
                dist2 = dist2 + (self.weights[k]*S2).sum(1, keepdim=True)

            score += 1 - ((dist1 + dist2).squeeze()/12.0)
        if batch_average:
            return score/ratio
        else:
            return score


def prepare_image(image, raio, device, resize=True):

    image_list = []
    for i in range(raio):
        image = transforms.functional.resize(image, [int(image.size[0] / (i+1)), int(image.size[1] / (i+1))])
        image2 = transforms.ToTensor()(image).unsqueeze(0).to(device)
        image_list.append(image2)
    return image_list


if __name__ == '__main__':

    from PIL import Image
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, default='./data/RW_Haze/2/gt/clear.jpg')
    parser.add_argument('--haze', type=str, default='./data/RW_Haze/2/hazy/57776003_C70674371_20181225070310209_TIMING.jpg')
    parser.add_argument('--dehaze', type=str, default='./data/D4/rw_old/results/reconstruct/57776003_C70674371_20181225070310209_TIMING.png')
    opts = parser.parse_args()

    image_scaling = 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image_dez = prepare_image(Image.open(opts.ref).convert("RGB"), image_scaling, device)
    image_ref = prepare_image(Image.open(opts.haze).convert("RGB"), image_scaling, device)
    image_trs = prepare_image(Image.open(opts.dehaze).convert("RGB"), image_scaling, device)

    model = DISTS().to(device)

    score = model(image_ref, image_dez, image_trs, image_scaling)

    print(score)


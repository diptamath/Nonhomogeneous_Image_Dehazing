import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os
import math
import argparse
import random
import models
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import time
from PIL import Image


#Hyper Parameters
METHOD = "DMSHN_1_2_4"

GPU = 0

SAMPLE_DIR = "./new_dataset/val/HAZY"       # store hazy images here

EXPDIR = 'DMSHN_results'

def save_images(images, name):
    filename = './test_results/' + EXPDIR + "/" + name
    torchvision.utils.save_image(images, filename)


def main():
    print("init data folders")

    encoder_lv1 = models.Encoder().cuda(GPU) 
    encoder_lv2 = models.Encoder().cuda(GPU) 
    encoder_lv3 = models.Encoder().cuda(GPU) 

    decoder_lv1 = models.Decoder().cuda(GPU) 
    decoder_lv2 = models.Decoder().cuda(GPU)     
    decoder_lv3 = models.Decoder().cuda(GPU) 



    # LOAD models here 

    saved_epoch = 130
    encoder_lv1.load_state_dict(torch.load(str('./checkpoints2/' + METHOD + "/ep" + str(saved_epoch)+ "_encoder_lv1.pkl")))
    encoder_lv2.load_state_dict(torch.load(str('./checkpoints2/' + METHOD + "/ep" + str(saved_epoch)+ "_encoder_lv2.pkl")))
    encoder_lv3.load_state_dict(torch.load(str('./checkpoints2/' + METHOD + "/ep" + str(saved_epoch)+ "_encoder_lv3.pkl")))

    decoder_lv1.load_state_dict(torch.load(str('./checkpoints2/' + METHOD + "/ep" + str(saved_epoch)+ "_decoder_lv1.pkl")))
    decoder_lv2.load_state_dict(torch.load(str('./checkpoints2/' + METHOD + "/ep" + str(saved_epoch)+ "_decoder_lv2.pkl")))
    decoder_lv3.load_state_dict(torch.load(str('./checkpoints2/' + METHOD + "/ep" + str(saved_epoch)+ "_decoder_lv3.pkl")))
    

    os.makedirs('./test_results/' + EXPDIR, exist_ok = True)

    iteration = 0.0
    test_time = 0.0
    
    for images_name in os.listdir(SAMPLE_DIR):
        # print (images_name )
        with torch.no_grad():             
            images_lv1 = transforms.ToTensor()(Image.open(SAMPLE_DIR + '/' + images_name).convert('RGB'))
            images_lv1 = Variable(images_lv1 - 0.5).unsqueeze(0).cuda(GPU)

            start = time.time()
            H = images_lv1.size(2)
            W = images_lv1.size(3)

            images_lv2 = F.interpolate(images_lv1, scale_factor = 0.5, mode = 'bilinear')
            images_lv3 = F.interpolate(images_lv2, scale_factor = 0.5, mode = 'bilinear')

            feature_lv3 = encoder_lv3(images_lv3)
            residual_lv3 = decoder_lv3(feature_lv3)


            residual_lv3 = F.interpolate(residual_lv3, scale_factor=2, mode= 'bilinear')
            feature_lv3 = F.interpolate(feature_lv3, scale_factor=2, mode= 'bilinear')
            feature_lv2 = encoder_lv2(images_lv2 + residual_lv3)
            residual_lv2 = decoder_lv2(feature_lv2 + feature_lv3)

            residual_lv2 = F.interpolate(residual_lv2, scale_factor=2, mode= 'bilinear')
            feature_lv2 = F.interpolate(feature_lv2, scale_factor=2, mode= 'bilinear')
            feature_lv1 = encoder_lv2(images_lv1 + residual_lv2)
            dehazed_image = decoder_lv2(feature_lv1 + feature_lv2)

        
            stop = time.time()
            test_time += stop-start
            print('RunTime:%.4f'%(stop-start), '  Average Runtime:%.4f'%(test_time/(iteration+1)))
            save_images(dehazed_image.data + 0.5, images_name) 



if __name__ == '__main__':
    main()

        

        


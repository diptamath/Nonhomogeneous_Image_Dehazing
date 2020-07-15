import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import math
import argparse
import random
import models
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from datasets import NH_HazeDataset
import time
from loss import CustomLoss_function

from tensorboardX import SummaryWriter
writer = SummaryWriter('runs/train2/run3')

from tqdm import tqdm

parser = argparse.ArgumentParser(description="Deep Multi-Scale Hierarchical Network")
parser.add_argument("-e","--epochs",type = int, default = 300)
parser.add_argument("-se","--start_epoch",type = int, default = 100)
parser.add_argument("-b","--batchsize",type = int, default = 8)
parser.add_argument("-s","--imagesize",type = int, default = 120)
parser.add_argument("-l","--learning_rate", type = float, default = 0.000001)
parser.add_argument("-g","--gpu",type=int, default=1)
args = parser.parse_args()

#Hyper Parameters
METHOD = "DMSHN_1_2_4"
LEARNING_RATE = args.learning_rate
EPOCHS = args.epochs
GPU = args.gpu
BATCH_SIZE = args.batchsize
IMAGE_SIZE = args.imagesize

start_epoch = args.start_epoch

def save_dehazed_images(images, iteration, epoch):
    filename = './checkpoints/' + METHOD + "/epoch" + str(epoch) + "/" + "Iter_" + str(iteration) + "_dehazed.png"
    torchvision.utils.save_image(images, filename)

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, 0.5*math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

def main():
    print("init data folders")

    encoder_lv1 = models.Encoder()
    encoder_lv2 = models.Encoder()    
    encoder_lv3 = models.Encoder()

    decoder_lv1 = models.Decoder()
    decoder_lv2 = models.Decoder()    
    decoder_lv3 = models.Decoder()
    
    encoder_lv1.apply(weight_init).cuda(GPU)    
    encoder_lv2.apply(weight_init).cuda(GPU)
    encoder_lv3.apply(weight_init).cuda(GPU)

    decoder_lv1.apply(weight_init).cuda(GPU)    
    decoder_lv2.apply(weight_init).cuda(GPU)
    decoder_lv3.apply(weight_init).cuda(GPU)
    
    encoder_lv1_optim = torch.optim.Adam(encoder_lv1.parameters(),lr=LEARNING_RATE)
    # encoder_lv1_scheduler = StepLR(encoder_lv1_optim,step_size=10,gamma=0.1)
    encoder_lv2_optim = torch.optim.Adam(encoder_lv2.parameters(),lr=LEARNING_RATE)
    # encoder_lv2_scheduler = StepLR(encoder_lv2_optim,step_size=10,gamma=0.1)
    encoder_lv3_optim = torch.optim.Adam(encoder_lv3.parameters(),lr=LEARNING_RATE)
    # encoder_lv3_scheduler = StepLR(encoder_lv3_optim,step_size=10,gamma=0.1)

    decoder_lv1_optim = torch.optim.Adam(decoder_lv1.parameters(),lr=LEARNING_RATE)
    # decoder_lv1_scheduler = StepLR(decoder_lv1_optim,step_size=10,gamma=0.1)
    decoder_lv2_optim = torch.optim.Adam(decoder_lv2.parameters(),lr=LEARNING_RATE)
    # decoder_lv2_scheduler = StepLR(decoder_lv2_optim,step_size=10,gamma=0.1)
    decoder_lv3_optim = torch.optim.Adam(decoder_lv3.parameters(),lr=LEARNING_RATE)
    # decoder_lv3_scheduler = StepLR(decoder_lv3_optim,step_size=10,gamma=0.1)

    # if os.path.exists(str('./checkpoints/' + METHOD + "/encoder_lv1.pkl")):
    #     encoder_lv1.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/encoder_lv1.pkl")))
    #     print("load encoder_lv1 success")
    # if os.path.exists(str('./checkpoints/' + METHOD + "/encoder_lv2.pkl")):
    #     encoder_lv2.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/encoder_lv2.pkl")))
    #     print("load encoder_lv2 success")
    # if os.path.exists(str('./checkpoints/' + METHOD + "/encoder_lv3.pkl")):
    #     encoder_lv3.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/encoder_lv3.pkl")))
    #     print("load encoder_lv3 success")

    # if os.path.exists(str('./checkpoints/' + METHOD + "/decoder_lv1.pkl")):
    #     decoder_lv1.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/decoder_lv1.pkl")))
    #     print("load encoder_lv1 success")
    # if os.path.exists(str('./checkpoints/' + METHOD + "/decoder_lv2.pkl")):
    #     decoder_lv2.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/decoder_lv2.pkl")))
    #     print("load decoder_lv2 success")
    # if os.path.exists(str('./checkpoints/' + METHOD + "/decoder_lv3.pkl")):
    #     decoder_lv3.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/decoder_lv3.pkl")))
    #     print("load decoder_lv3 success")

    # LOAD models here 

    saved_epoch = 99
    encoder_lv1.load_state_dict(torch.load(str('./checkpoints2/' + METHOD + "/ep" + str(saved_epoch)+ "_encoder_lv1.pkl")))
    encoder_lv2.load_state_dict(torch.load(str('./checkpoints2/' + METHOD + "/ep" + str(saved_epoch)+ "_encoder_lv2.pkl")))
    encoder_lv3.load_state_dict(torch.load(str('./checkpoints2/' + METHOD + "/ep" + str(saved_epoch)+ "_encoder_lv3.pkl")))

    decoder_lv1.load_state_dict(torch.load(str('./checkpoints2/' + METHOD + "/ep" + str(saved_epoch)+ "_decoder_lv1.pkl")))
    decoder_lv2.load_state_dict(torch.load(str('./checkpoints2/' + METHOD + "/ep" + str(saved_epoch)+ "_decoder_lv2.pkl")))
    decoder_lv3.load_state_dict(torch.load(str('./checkpoints2/' + METHOD + "/ep" + str(saved_epoch)+ "_decoder_lv3.pkl")))
    
    if os.path.exists('./checkpoints2/' + METHOD) == False:
        os.system('mkdir ./checkpoints2/' + METHOD)    
            
    for epoch in range(start_epoch, EPOCHS):
        # encoder_lv1_scheduler.step(epoch)
        # encoder_lv2_scheduler.step(epoch)
        # encoder_lv3_scheduler.step(epoch)

        # decoder_lv1_scheduler.step(epoch)
        # decoder_lv2_scheduler.step(epoch)
        # decoder_lv3_scheduler.step(epoch)     
        
        print("Training...")
        
        train_dataset = NH_HazeDataset(
            hazed_image_files = 'new_dataset/train_patch_hazy.txt',   # make changes here !
            dehazed_image_files = 'new_dataset/train_patch_gt.txt',
            root_dir = 'new_dataset/',
            crop = False,
            rotation = False,
            crop_size = IMAGE_SIZE,
            transform = transforms.Compose([transforms.Resize((128,160)),
                transforms.ToTensor()
                ]))
        train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True)
        start = 0
        iLoss = 0

        print('Epoch: ',epoch)

        # torch.save(encoder_lv1.state_dict(),str('./checkpoints2/' + METHOD + "/encoder_lv1.pkl"))
        
        for iteration, images in enumerate(train_dataloader):            
            # mse = nn.MSELoss().cuda(GPU)   
            # mae = nn.L1Loss().cuda(GPU)      
            custom_loss_fn = CustomLoss_function().cuda(GPU) 
            
            gt = Variable(images['dehazed_image'] - 0.5).cuda(GPU)            
            H = gt.size(2)
            W = gt.size(3)

            images_lv1 = Variable(images['hazed_image'] - 0.5).cuda(GPU)
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
            feature_lv1 = encoder_lv1(images_lv1 + residual_lv2)
            dehazed_image = decoder_lv1(feature_lv1 + feature_lv2)

            loss_lv1, loss_recn, loss_perc, loss_tv = custom_loss_fn(dehazed_image,gt)

            loss  = loss_lv1

            iLoss += loss.item()
            
            encoder_lv1.zero_grad()
            encoder_lv2.zero_grad()
            encoder_lv3.zero_grad()

            decoder_lv1.zero_grad()
            decoder_lv2.zero_grad()
            decoder_lv3.zero_grad()

            loss.backward()

            encoder_lv1_optim.step()
            encoder_lv2_optim.step()
            encoder_lv3_optim.step()

            decoder_lv1_optim.step()
            decoder_lv2_optim.step()
            decoder_lv3_optim.step() 

            writer.add_scalar('Loss',loss.item(),epoch*len(train_dataloader)+iteration)
            writer.add_scalar('Loss_recn',loss_recn.item(),epoch*len(train_dataloader)+iteration)
            writer.add_scalar('Loss_perc',loss_perc.item(),epoch*len(train_dataloader)+iteration)
            writer.add_scalar('Loss_tv',loss_tv.item(),epoch*len(train_dataloader)+iteration)
            
            if (iteration+1)%10 == 0:
                stop = time.time()
                print("epoch:", epoch, "iteration:", iteration+1, "loss:%.4f"%loss.item(), 'time:%.4f'%(stop-start))
                start = time.time()
        
                    
        torch.save(encoder_lv1.state_dict(),str('./checkpoints2/' + METHOD + "/ep" + str(epoch)+ "_encoder_lv1.pkl"))
        torch.save(encoder_lv2.state_dict(),str('./checkpoints2/' + METHOD + "/ep" + str(epoch)+"_encoder_lv2.pkl"))
        torch.save(encoder_lv3.state_dict(),str('./checkpoints2/' + METHOD + "/ep" + str(epoch)+"_encoder_lv3.pkl"))

        torch.save(decoder_lv1.state_dict(),str('./checkpoints2/' + METHOD + "/ep" + str(epoch)+"_decoder_lv1.pkl"))
        torch.save(decoder_lv2.state_dict(),str('./checkpoints2/' + METHOD + "/ep" + str(epoch)+"_decoder_lv2.pkl"))
        torch.save(decoder_lv3.state_dict(),str('./checkpoints2/' + METHOD + "/ep" + str(epoch)+"_decoder_lv3.pkl"))

        print ('Training Loss:', iLoss/len(train_dataloader))
                

if __name__ == '__main__':
    main()

        

        


import pandas as pd 
import glob
import cv2

train_inp_images = glob.glob('/media/newton/newton/nitre/dataset/train/HAZY/*.png')
train_out_images = glob.glob('/media/newton/newton/nitre/dataset/train/GT/*.png')

val_inp_images = glob.glob('/media/newton/newton/nitre/dataset/val/HAZY/*.png')
val_out_images = glob.glob('/media/newton/newton/nitre/dataset/val/GT/*.png')

#test_inp_images = glob.glob('/media/newton/newton/nitre/dataset/train/HAZY/*.png')

xcords = []
ycords = []

for img_path in train_inp_images:
	for i in range(0,1600-160,160):
		for j in range(0,1200-120,120):
			img_nm = img_path.split('/')[-1]
			frame1 = cv2.imread(img_path)
			frame2 = cv2.imread('/media/newton/newton/nitre/dataset/train/GT/' + img_nm)
			cropImg1 = frame1[j:j+120,i:i+160]
			cropImg2 = frame2[j:j+120,i:i+160]
			cv2.imwrite('/media/newton/newton/nitre/dataset/prepare_data/new_dataset/train/HAZY/'+str(i)+'_'+str(j)+'_'+img_nm,cropImg1)
			cv2.imwrite('/media/newton/newton/nitre/dataset/prepare_data/new_dataset/train/GT/'+str(i)+'_'+str(j)+'_'+img_nm,cropImg2)


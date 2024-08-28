"""
#General-purpose script for detecting fence anomalies
"""

import os
import cv2
import glob
import numpy as np
import torch

from inference.myViT import ViT
from torch.optim import Adam, RAdam
from torch.nn import CrossEntropyLoss
from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST
from tqdm import tqdm, trange

if __name__ == '__main__':
	
	batch_size = 128

	# Loading data
	transform = ToTensor()
	
	# Defining model and training options
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
	model = ViT((3, 256, 256), n_patches=4, n_blocks=3, hidden_d=128, n_heads=4, out_d=2).to(device)
	N_EPOCHS = 10
	LR = 0.006
	optimizer = RAdam(model.parameters(), lr=LR)
	criterion = CrossEntropyLoss()

	#General training loop
	path = 'data/train/'
	for epoch in trange(N_EPOCHS, desc="Training"):
		train_loss = 0.0
		
		#Load my own dataset, images will be 3x256x256
		file_list = [name for name in os.listdir(path + 'images')]
		number_of_batches = np.ceil(len(file_list) / batch_size)
		np.random.shuffle(file_list)
			
		#This is not an efficient data loader - could be improved but loading 128x3x256x256 images is heavy
		while(len(file_list) > 0):	#Grab 128 instances and corresponding masks
			if(len(file_list) >= batch_size):
				batch = file_list[0 : batch_size]
				file_list = file_list[batch_size : ]
			else:
				batch = file_list
				file_list = []
				
			print(len(file_list))
				
			images = []
			masks = []
			labels = []
			for i in range(len(batch)):
				image_name = path + 'images/' + batch[i]
				mask_name = path + 'masks/' + batch[i]
				
				images.append( np.swapaxes(cv2.resize(cv2.imread(image_name), (256,256) ), -1, 0))
				masks.append( np.swapaxes(cv2.resize(cv2.imread(mask_name), (256,256) ), -1, 0))
			
				label = batch[i].split('_')[1].split('.')[0]
				labels.append(int(label))
	
			images = torch.from_numpy(np.array(images))
			masks = torch.from_numpy(np.array(masks))
			labels = torch.from_numpy(np.array(labels))
			
			
			#Forward pass
			images, masks, labels = images.to(device), masks.to(device), labels.to(device)
			y_hat = model([images, masks])
			loss = criterion(y_hat, labels)

			#Track loss
			train_loss += loss.detach().cpu().item() / number_of_batches
			
			#Apply gradients
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")
		
	
	# Test loop
	path = 'data/test/'
	with torch.no_grad():
		file_list = [name for name in os.listdir(path + 'images')]
		number_of_batches = np.ceil(len(file_list) / batch_size)
		np.random.shuffle(file_list)
		
		correct, total = 0, 0
		test_loss = 0.0
		
		while(len(file_list) > 0):	#Grab 128 instances and corresponding masks
			if(len(file_list) >= batch_size):
				batch = file_list[0 : batch_size]
				file_list = file_list[batch_size : ]
			else:
				batch = file_list
				file_list = []
				
			print(len(file_list))
				
			images = []
			masks = []
			labels = []
			for i in range(len(batch)):
				image_name = path + 'images/' + batch[i]
				mask_name = path + 'masks/' + batch[i]
				
				images.append( np.swapaxes(cv2.resize(cv2.imread(image_name), (256,256) ), -1, 0))
				masks.append( np.swapaxes(cv2.resize(cv2.imread(mask_name), (256,256) ), -1, 0))
			
				label = batch[i].split('_')[1].split('.')[0]
				labels.append(int(label))
	
			images = torch.from_numpy(np.array(images))
			masks = torch.from_numpy(np.array(masks))
			labels = torch.from_numpy(np.array(labels))
		
			images, masks, labels = images.to(device), masks.to(device), labels.to(device)
			y_hat = model([images, masks])
			loss = criterion(y_hat, labels)
		
			test_loss += loss.detach().cpu().item() / number_of_batches
		
			correct += torch.sum(torch.argmax(y_hat, dim=1) == labels).detach().cpu().item()
			total += len(images)
			
		print(f"Test loss: {test_loss:.2f}")
		print(f"Test accuracy: {correct / total * 100:.2f}%")
	
	
			
	#Save full model - not advisable
	torch.save(model, './inference/transformer.pth')
	#Save state dictionary - better option
	torch.save(model.state_dict(), './inference/transformer_dict.pth')

	
	
	
	
	
	
	
	
	

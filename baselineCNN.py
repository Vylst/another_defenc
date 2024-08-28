import numpy as np
import tensorflow as tf
import os
import cv2

from tqdm import tqdm, trange

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
	
SEED_VALUE = 42
 
# Fix seed to make training deterministic.
#random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)

def cnn_model(input_shape=(256, 256, 3)):
     
	model = tf.keras.models.Sequential()

	#------------------------------------
	# Conv Block 1: 32 Filters, MaxPool.
	#------------------------------------
	model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=input_shape))
	model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
	model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

	#------------------------------------
	# Conv Block 2: 64 Filters, MaxPool.
	#------------------------------------
	model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
	model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
	model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

	#------------------------------------
	# Conv Block 3: 64 Filters, MaxPool.
	#------------------------------------
	model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
	model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
	model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

	#------------------------------------
	# Flatten the convolutional features.
	#------------------------------------
	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(512, activation='relu'))
	model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

	return model



if __name__ == '__main__':
	model = cnn_model()
	model.summary()
	N_EPOCHS = 10
	LR = 0.001
	batch_size = 128

	model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = LR), loss=tf.keras.losses.BinaryCrossentropy())

	#General training loop
	path = '../data/train/'
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
				
				images.append(cv2.resize(cv2.imread(image_name), (256,256) ))
				masks.append(cv2.resize(cv2.imread(mask_name), (256,256) ))
			
				label = batch[i].split('_')[1].split('.')[0]
				labels.append(int(label))
	
			images = np.array(images)
			masks = np.array(masks)
			labels = np.array(labels)

			logs = model.train_on_batch(images, labels) 
		






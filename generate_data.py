"""
#Script for generating naive-based masks of images with fences

IT_S TERRIBLE - THE NAIVE SEGMENTATION PROCESS IS VERY SUSCEPTIBLE TO THE NOISE OF THE SYNTHETIC IMAGES
"""

import cv2
import os
import numpy as np

from utils.ImTools import ImTools

	

if __name__ == '__main__':
	
	tools = ImTools()						#ImTools object


	path = 'data/train/'
	for image_path in os.listdir(path + 'images'):
		img = cv2.imread(path + 'images/' + image_path)
		
		naiveMask = tools.naiveSegmenter(img)
	
		cv2.imwrite(path + 'generated_naive_masks/' + image_path, naiveMask)
		
	
	
	
	

	
	
	

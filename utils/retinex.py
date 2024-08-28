import numpy as np
import cv2 

'''
# This class implements the Retinex algorithm for emulation of the color enhancement achieved by the human eye
# Refers to https://github.com/aravindskrishnan/Retinex-Image-Enhancement
'''
class Retinex:
	def __init__(self):
		self.variance_list = [15, 80, 250]

	def singleScaleRetinex(self, img, variance):
		retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), variance))
		return retinex

	def multiScaleRetinex(self, img):
		retinex = np.zeros_like(img)
		for variance in self.variance_list:
			retinex += self.singleScaleRetinex(img, variance)
		retinex = retinex / len(self.variance_list)
		return retinex
   
	def MSR(self, img):
		img = np.float64(img) + 1.0
		img_retinex = self.multiScaleRetinex(img)

		for i in range(img_retinex.shape[2]):
			unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)
			for u, c in zip(unique, count):
				if u == 0:
					zero_count = c
					break            
			low_val = unique[0] / 100.0
			high_val = unique[-1] / 100.0
			for u, c in zip(unique, count):
				if u < 0 and c < zero_count * 0.1:
					low_val = u / 100.0
				if u > 0 and c < zero_count * 0.1:
					high_val = u / 100.0
					break            
			img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)
			img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) * 255
		
		img_retinex = np.uint8(img_retinex)        
		return img_retinex

	def SSR(self, img, variance):
		img = np.float64(img) + 1.0
		img_retinex = self.singleScaleRetinex(img, variance)
		for i in range(img_retinex.shape[2]):
			unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)
			for u, c in zip(unique, count):
				if u == 0:
					zero_count = c
					break            
			low_val = unique[0] / 100.0
			high_val = unique[-1] / 100.0
			for u, c in zip(unique, count):
				if u < 0 and c < zero_count * 0.1:
					low_val = u / 100.0
				if u > 0 and c < zero_count * 0.1:
					high_val = u / 100.0
					break            
			img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)
		
			img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) * 255
		
		img_retinex = np.uint8(img_retinex)        
		return img_retinex	
		
		

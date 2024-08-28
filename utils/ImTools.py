import numpy as np
import cv2 

from utils.zhangSuen import zhangSuen
from utils.retinex import Retinex

'''
# This class implements a collection of image processing operations
'''
class ImTools:
	def __init__(self):
		self.retinex = Retinex()
		self.zhangSuen = zhangSuen()
		
	def bilateralFilter(self, src, d, sigmaColor, sigmaSpace):
		"""Apply bilateral filtering to the source image
		Wraps OpenCV's function: https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430737ed
		"""
		return cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace)

	def contrastEnhancerLAB(self, src):
		"""Enhance image contrast in LAB color space, via adaptive histogram equalization on the L-channel 
		L-channel: representing lightness in the image
		a-channel: representing change in color between red and green
		b-channel: representing change in color between yellow and blue
		CLAHE: Contrast Limited Adaptive Histogram Equalization
		"""
		lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)			# Conversion to LAB color space
		l_channel, a, b = cv2.split(lab)
		clahe = cv2.createCLAHE(clipLimit=0.1, tileGridSize=(10,10))	# Applying CLAHE to image L-channel
		cl = clahe.apply(l_channel)
		limg = cv2.merge((cl,a,b))					# Merge the CLAHE enhanced L-channel with the a and b channel
		enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)		#Regress to BGR color space

		return enhanced_img

	def binarizerOSTU(self, src, number=255):
		"""Binarize image according to OSTU method, requiring image to be grayscale
		Wraps OpenCV's function: https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57
		"""
		#gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
		_, binarized_img = cv2.threshold(src, 0, number, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

		return binarized_img
		
	def close(self, src, kernel_size=50):
		"""Perform a closing operation over the image, using OpenCV's morphology functions
		Wraps OpenCV's function: 
		"""
		kernel = np.ones((kernel_size, kernel_size), np.uint8)
		closed = cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel)
		return closed
		
	def retinexWrapper(self, src):
		"""Wrapper for retinex algorithm
		"""
		return self.retinex.MSR(src)
		
	def zhangSuenWrapper(self, src):
		"""Wrapper for retinex algorithm - It is very heavy and time consuming
		Image should be binarized to {0,1} values
		"""
		return self.zhangSuen.zhangSuen(src)
		
	def naiveSegmenter(self, src):
		"""A basic segmentation of foreground, based on the process described in
		the paper 'An adaptive method of damage detection for fishing nets based on image processing technology'
		with added Retinex, as suggested by another paper
		"""
		img = src		
		#img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
		img = self.bilateralFilter(img, 15, 75, 75)			#Remove noise through bilateral filtering
		img = self.contrastEnhancerLAB(img)		#Enhance image contrast
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = self.binarizerOSTU(img)				#Binarize Image through OSTU method
		img = cv2.bitwise_not(img)				#Invert binarized image
		img = self.close(img, 8)
		
		return img
		
	def dilate(self, src, kernel=15):
		"""Perform a dilation operation over the image, using OpenCV's morphology functions
		Wraps OpenCV's function: 
		"""
		kernel = np.ones((kernel, kernel), np.uint8)
		dilated = cv2.dilate(src, kernel, iterations = 1)
		return dilated
		
	def maxConvexHull(self, src):
		"""A function to extract the maximum convex hull encompassing a fence mask. A dilation operation is performed,
		followed by closing and binarization. The remaining hulls are disregarded except for the one with largest area
		"""
		gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
		#dilated =  self.dilate(gray)
		#closed = self.close(dilated)
		binarized = self.binarizerOSTU(gray)
		
		(cnts, _) = cv2.findContours(binarized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		c = max(cnts, key = cv2.contourArea)
		mask = np.zeros_like(binarized , dtype=np.uint8)
		convexHull = cv2.convexHull(c)
		cv2.fillPoly(mask, pts=[convexHull], color=255)
		
		return mask
		
		
	def connectedComponentAnalysis(self, src):
		'''
		Could be interesting to explore
		'''
		
		#output = cv2.connectedComponentsWithStats(naiveMask, 4, cv2.CV_32S)
		#(numLabels, labels, stats, centroids) = output
		
		return
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		

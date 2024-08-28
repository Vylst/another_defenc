"""
#General-purpose script for detecting fence anomalies
"""

import cv2
import numpy as np

from utils.retinex import Retinex
from utils.ImTools import ImTools
from inference import segmentMask
from inference.ViT import ViT


def video(source):
	"""
	#For processing a video, for instance segmenting mask from each frame
	"""
	retinex = Retinex()						#Retinex object
	tools = ImTools()						#ImTools object
	
	cap = cv2.VideoCapture(source)
	while not cap.isOpened():
		cap = cv2.VideoCapture(source)
		cv2.waitKey(1000)
		print("Wait for the header")
	
	count = 0
	while cap.isOpened():
		ret, frame = cap.read()
		
		small_img = cv2.resize(frame, (256,256), interpolation = cv2.INTER_AREA)
		
		virginMask = segmentMask(frame)						#Segment first tier virgin mask
		hullMask = tools.maxConvexHull(virginMask)					#Obtain hulled second tier mask
		rawCutOut = cv2.bitwise_and(small_img, small_img, mask = hullMask)		#Use 2nd tier mask to cut out full fence ROI from raw image
		maskCutOut = cv2.bitwise_and(virginMask, virginMask, mask = hullMask)	#Use 2nd tier mask to cut out full fence ROI from virgin mask
		hulledMask = segmentMask(rawCutOut)						#Segment first tier hulled mask from ROIed raw image
		orMask = cv2.bitwise_or(maskCutOut, hulledMask)				#OR the mask from the hulled raw image with the hulled virgin mask
		
		out = np.hstack((small_img, orMask))
		cv2.imwrite("output/frame%d.jpg" % count, out)
		count = count + 1
		
	cap.release()
	

if __name__ == '__main__':
	
	#video("./robot.avi")

	retinex = Retinex()							#Retinex object
	tools = ImTools()							#ImTools object

	org_img = cv2.imread('test3.png') #, cv2.IMREAD_GRAYSCALE)
	inferredMask = segmentMask(org_img)								#Segment mask by pix2pix inference
	hull = cv2.resize(tools.maxConvexHull(inferredMask), (org_img.shape[1], org_img.shape[0]))	#Obtain max convex hull
	cropped = cv2.bitwise_and(org_img, org_img, mask = hull)					#Crop hull from original image
	
	naiveMask = tools.naiveSegmenter(cropped)
	binarized = tools.binarizerOSTU(naiveMask, 1)
	small = cv2.resize(binarized, (256,256), interpolation = cv2.INTER_AREA)
	zs = 255*tools.zhangSuenWrapper(small)
	

	cv2.imshow('test', naiveMask)
	cv2.imshow('test2', zs)
	
	
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
	
	
	
	
	
	'''
	cap = cv2.VideoCapture("./robot.avi")
	while not cap.isOpened():
		cap = cv2.VideoCapture("./robot.avi")
		cv2.waitKey(1000)
		print("Wait for the header")
	
	count = 0
	while cap.isOpened():
		ret, frame = cap.read()
		
		#msr_img = retinex.MSR(frame)
		#naiveMask = tools.naiveSegmenter(msr_img)
		
		hull = segmentMask(frame)
		hull = tools.maxConvexHull(hull)
		hull = cv2.resize(hull, (np.shape(frame)[1], np.shape(frame)[0]), interpolation = cv2.INTER_AREA)
		
		print(np.shape(frame))
		print(np.shape(hull))
		
		out = np.hstack((frame, cv2.cvtColor(hull, cv2.COLOR_GRAY2BGR)))
		cv2.imwrite("output/frame" + str(count).zfill(5) + ".jpg", out)
		count = count + 1
		
	cap.release()
	'''
	
	
	
	'''
	small_img = cv2.resize(org_img, (256,256), interpolation = cv2.INTER_AREA)

	#msr_img = retinex.MSR(org_img)						#Apply MultiScale Retinex
	virginMask = segmentMask(org_img)						#Segment first tier virgin mask
	hullMask = tools.maxConvexHull(virginMask)					#Obtain hulled second tier mask
	rawCutOut = cv2.bitwise_and(small_img, small_img, mask = hullMask)		#Use 2nd tier mask to cut out full fence ROI from raw image
	maskCutOut = cv2.bitwise_and(virginMask, virginMask, mask = hullMask)	#Use 2nd tier mask to cut out full fence ROI from virgin mask
	hulledMask = segmentMask(rawCutOut)						#Segment first tier hulled mask from ROIed raw image
	orMask = cv2.bitwise_or(maskCutOut, hulledMask)				#OR the mask from the hulled raw image with the hulled virgin mask

	
	cv2.imshow('test1', virginMask)
	
	'''
	
	
	
	
	
	
	
	
	
	
	
	
	

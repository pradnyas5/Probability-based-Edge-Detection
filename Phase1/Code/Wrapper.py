#!/usr/bin/env python3

"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code

Colab file can be found at:
	https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute

Code adapted from CMSC733 at the University of Maryland, College Park.
"""

# Code starts here:

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, rotate
import math
import os
from sklearn.cluster import KMeans


base_path = '/home/pradnya/cv_course/homework/pshinde1_hw0/Phase1/Code'


##### Filter 1 ##### 
# Generate Difference of Gaussian Filter Bank: (DoG)
# Display all the filters in this filter bank and save image as DoG.png,
# use command "cv2.imwrite(...)"
# """

def get_gaussian_kernel_2D(k_size, sigma, elongation_factor):
	x = np.arange(-math.floor(k_size/2), math.floor(k_size/2) + 1, 1)
	y = x
	x, y = np.meshgrid(x, y)
	norm = 1/(np.sqrt(2.0*np.pi*sigma**2))
	g_kernel = norm * np.exp(-(x**2 + y**2)/(2*((sigma*elongation_factor)**2)))
	return g_kernel/np.sum(g_kernel)


def plot_DoG_filters(scales, orients, filters):
	fig, axes = plt.subplots(len(scales), len(orients), figsize = (11,5))
	for i in range(len(scales)):
		axs_i = axes[i]
		for j in range(len(orients)):
			axs_j = axs_i[j] 
			filter = filters[j + i*len(orients)]
			# norm_filter = cv2.normalize(filter, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
			axs_j.imshow(filter, cmap = 'gray')
			axs_j.axis('off')

	plt.savefig('./FilterBanks/DoG.png', dpi=500, bbox_inches = 'tight')
	plt.show()		  
			   
def generate_DoG(visualize):
	# First step is to generate a Gaussian Kernel, for that we define two primary parameters: kernel size K and sigma and then we move ahead to generate Oriented DoG
	# Rule of thumb K (approx)= 2*pi*sigma
	# We will set sigma as 1 
	sigma = [2, 3]
	kernel_size = [5, 7]   # scales/sizes of kernel 
	orient_angles= np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330])*np.pi/180 #Orientations at which the filters will be roatated 
	DoG_filters = []
	elongation_factor = 1
	for i, size in enumerate(kernel_size):
		for angle in orient_angles:
				  #Get a Gaussian Kernel
			g_kernel = get_gaussian_kernel_2D(size, sigma[i], elongation_factor)

			#DefineSObel filters in X-Y directions
			sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
			sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

			#Convolve Sobel and Gaussian Kernel to generate filters in X-Y directions
			DoG_x = convolve(g_kernel, sobel_x)
			DoG_y = convolve(g_kernel, sobel_y)
			
			# Calculate Orientation
			DoG_filter = np.cos(angle)*DoG_x + np.sin(angle)*DoG_y
			DoG_filters.append(DoG_filter)   

	if visualize:
		plot_DoG_filters(kernel_size, orient_angles, DoG_filters)
	
	return DoG_filters
				 
# generate_DoG(True)  # In total we will have 24 filters in the filter bank	

#### Filter 2 ####
# Generate Leung-Malik Filter Bank: (LM)
# Display all the filters in this filter bank and save image as LM.png,
# use command "cv2.imwrite(...)"
# """
def plot_LMS(LMS):
	for i in range(48):
		plt.subplot(4, 12, i+1)
		plt.axis('off')
		plt.imshow(LMS[i], cmap='gray')
	plt.savefig('./FilterBanks/LMS.png', dpi=500, bbox_inches = 'tight')
	plt.show()

def get_LoG(k_size, scale, factor):
	# x = np.arange(-math.floor(k_size/2), math.floor(k_size/2) + 1, 1)
	# y = x
	r = int((k_size -1)/2)
	y = np.linspace(-r, r, k_size).astype(int)
	x = y.reshape((k_size, 1))
	sigma = scale*factor
	log_kernel = 1/(np.sqrt(np.pi*sigma**2))*(((x**2 + y**2)/sigma**4-(1/sigma**2)))*np.exp(-(x**2 + y**2)/(2*sigma**2))
	return log_kernel
	

def get_gaussian_kernel_1D(k_size, sigma, order, elongation_factor):
	x = np.arange(-math.floor(k_size/2), math.floor(k_size/2) + 1, 1)
	sigma = sigma*elongation_factor
	g_kernel_1D = (1/(np.sqrt(2*np.pi*(sigma**2))))*np.exp(-0.5*(x**2)/(sigma**2))
	
	if order == 0:    # Zero^th derivative of gaussian
		return g_kernel_1D
	
	elif order == 1:  # First order derivative of gaussian
		g_kernel_1D = (-x/sigma**2)*g_kernel_1D
		return g_kernel_1D
	
	else:             # Second order derivative of gaussian
		g_kernel_1D = (x**2/(sigma**4) - 1/(sigma**2))*g_kernel_1D
		return g_kernel_1D


def generate_LMS(visualize_res):
	#Define the scales of LM Small anf LM Large filters
	LMS_scales = np.array([1, np.sqrt(2), 2, 2*np.sqrt(2)])
	# LML_scales = np.array([ np.sqrt(2), 2, 2*np.sqrt(2), 4])
	k_size = 49
	orientations = -np.arange(0 , 180, 180/6)
	derivatives_1 = []   #List to store first order derivatives
	derivatives_2 = []   #List to store second order derivatives
 
	for scale in LMS_scales[:3]:
		for angle in orientations:
			g_kernel_x1 = get_gaussian_kernel_1D(k_size, scale,1,1)
			g_kernel_y1 = get_gaussian_kernel_1D(k_size, scale,1,3)
			
			g_kernel_first = np.outer(g_kernel_x1, g_kernel_y1)

			sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
			g_kernel_first = convolve(g_kernel_first, sobel)
			center = (int(k_size / 2), int(k_size / 2))
			rotation_matrix = cv2.getRotationMatrix2D(center = center, angle = angle, scale = 1)
			g_kernel_first = cv2.warpAffine(src = g_kernel_first, M = rotation_matrix, dsize = (k_size, k_size))
			g_kernel_first = cv2.normalize(g_kernel_first, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

			# g_kernel_first = rotate(g_kernel_first, angle)
			derivatives_1.append(g_kernel_first)

			g_kernel_x2 = get_gaussian_kernel_1D(k_size, scale,2,1)
			g_kernel_y2 = get_gaussian_kernel_1D(k_size, scale,2,3)

			g_kernel_second = np.outer(g_kernel_x2, g_kernel_y2)

			g_kernel_second = convolve(g_kernel_second, sobel)
			center = (int(k_size / 2), int(k_size / 2))
			rotation_matrix = cv2.getRotationMatrix2D(center = center, angle = angle, scale = 1)
			g_kernel_second = cv2.warpAffine(src = g_kernel_second, M = rotation_matrix, dsize = (k_size, k_size))
			g_kernel_second = cv2.normalize(g_kernel_second, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
			# g_kernel_second = rotate(g_kernel_second, angle)
			derivatives_2.append(g_kernel_second)
	
	derivatives = derivatives_1 + derivatives_2

	gaussians = []
	for scale in LMS_scales:
		gaussian = get_gaussian_kernel_2D(k_size, scale, 1)
		gaussian = cv2.normalize(gaussian, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
		gaussians.append(gaussian)

	laplacians = []
	multipliers = [1, 3]
	for factor in multipliers:
		for scale in LMS_scales:
			laplacian = get_LoG(k_size, scale, factor)
			laplacian = cv2.normalize(laplacian, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
			laplacians.append(laplacian)
	
	LMS = derivatives + laplacians + gaussians
	if visualize_res:
		plot_LMS(LMS)
	
	return LMS
	
# generate_LMS(True)

def plot_LML(LML):
	for i in range(48):
		plt.subplot(4, 12, i+1)
		plt.axis('off')
		plt.imshow(LML[i], cmap='gray')
	plt.savefig('./FilterBanks/LML.png', dpi=500, bbox_inches = 'tight')
	plt.show()


def generate_LML(visualize_res):
	#Define the scales of LM Small anf LM Large filters
	LML_scales = np.array([np.sqrt(2), 2, 2*np.sqrt(2),4])
	# LML_scales = np.array([ np.sqrt(2), 2, 2*np.sqrt(2), 4])
	k_size = 49
	orientations = -np.arange(0 , 180, 180/6)
	derivatives_1 = []   #List to store first order derivatives
	derivatives_2 = []   #List to store second order derivatives
 
	for scale in LML_scales[:3]:
		for angle in orientations:
			g_kernel_x1 = get_gaussian_kernel_1D(k_size, scale,1,1)
			g_kernel_y1 = get_gaussian_kernel_1D(k_size, scale,1,3)
			
			g_kernel_first = np.outer(g_kernel_x1, g_kernel_y1)
			# g_kernel_first = g_kernel_x1*g_kernel_y1
			center = (int(k_size / 2), int(k_size / 2))
			rotation_matrix = cv2.getRotationMatrix2D(center = center, angle = angle, scale = 1)
			g_kernel_first = cv2.warpAffine(src = g_kernel_first, M = rotation_matrix, dsize = (k_size, k_size))
			g_kernel_first = cv2.normalize(g_kernel_first, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

			# g_kernel_first = rotate(g_kernel_first, angle)
			derivatives_1.append(g_kernel_first)

			g_kernel_x2 = get_gaussian_kernel_1D(k_size, scale,2,1)
			g_kernel_y2 = get_gaussian_kernel_1D(k_size, scale,2,3)

			g_kernel_second = np.outer(g_kernel_x2, g_kernel_y2)
			# g_kernel_second = g_kernel_x2*g_kernel_y2
			center = (int(k_size / 2), int(k_size / 2))
			rotation_matrix = cv2.getRotationMatrix2D(center = center, angle = angle, scale = 1)
			g_kernel_second = cv2.warpAffine(src = g_kernel_second, M = rotation_matrix, dsize = (k_size, k_size))
			g_kernel_second = cv2.normalize(g_kernel_second, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
			# g_kernel_second = rotate(g_kernel_second, angle)
			derivatives_2.append(g_kernel_second)
	
	derivatives = derivatives_1 + derivatives_2

	gaussians = []
	for scale in LML_scales:
		gaussian = get_gaussian_kernel_2D(k_size, scale, 1)
		gaussian = cv2.normalize(gaussian, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
		gaussians.append(gaussian)

	laplacians = []
	multipliers = [1, 3]
	for factor in multipliers:
		for scale in LML_scales:
			laplacian = get_LoG(k_size, scale, factor)
			laplacian = cv2.normalize(laplacian, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
			laplacians.append(laplacian)
	
	LML = derivatives + laplacians + gaussians
	if visualize_res:
		plot_LML(LML)
	
	return LML

# generate_LML(True)
#### Filter 3 ####	
# Generate Gabor Filter Bank: (Gabor)
	# Display all the filters in this filter bank and save image as Gabor.png,
	# use command "cv2.imwrite(...)"
	# """
		
def plot_gabor(gb_filter):
	for i in range(40):
		plt.subplot(5, 8, i+1)
		plt.axis('off')
		plt.imshow(gb_filter[i], cmap='gray')
	plt.savefig('./FilterBanks/Gabor.png', dpi=500, bbox_inches = 'tight')
	plt.show()


def get_gabor(k_size, sigma, theta, Lambda, psi, gamma):
	kernel_dim = np.linspace(-int(k_size/2), int(k_size/2), k_size)
	x, y = np.meshgrid(kernel_dim, kernel_dim)
	x = x*np.cos(theta) + y*np.sin(theta)
	y = -x*np.sin(theta) + y*np.cos(theta)
	multiplier_1 = np.exp(-0.5*((x**2 + (gamma**2*y**2))/sigma**2))
	multiplier_2 = np.cos(2*np.pi*(x/Lambda) + psi)
	gabor = multiplier_1*multiplier_2
	return gabor


def generate_Gabor(visualize): 
	k_size = 49
	sigma_params = np.array([9, 11, 13, 15, 17])
	# sigma_params = np.array([np.sqrt(2),1+np.sqrt(2), 2+np.sqrt(2), 2*np.sqrt(2), 3+np.sqrt(2)])
	theta_values = np.arange(0, 180, 180/8)*np.pi/180
	# Lambda = np.array([5, 7, 9, 11, 13])
	Lambda = 7
	gamma = 1
	psi = 0
	gb_filter = []
	for i, sigma in enumerate(sigma_params):
		for theta in theta_values:
			# Lambda = sigma
			gabor = get_gabor(k_size, sigma, theta, Lambda, psi, gamma)
			gabor = cv2.normalize(gabor,dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
			gb_filter.append(gabor)

	if visualize:
		plot_gabor(gb_filter)
	
	return gb_filter


# generate_Gabor(True)

def get_clusters(responses, type):
	if type == 'TextonMap':
		clusters = 64
		n_init = 2
	# elif type == 'BrightnessMap' : 
	# 	clusters = 32
	else:
		clusters = 16	
		n_init = 4
	img_params = np.shape(responses)
	#img_params = (number of filters, height, width)
	responses_2D = np.transpose(responses, axes=[1, 2, 0]).reshape(img_params[1]*img_params[2], img_params[0])
	#Generate a KMeans model
	model = KMeans(n_clusters=clusters, n_init=n_init)
	#Fit the model onto the filter responses
	clusters = model.fit_predict(responses_2D)

	return np.reshape(clusters, img_params[1:])

def plot_masks(masks, row, col):
	for i in range(len(masks)):
		plt.subplot(row, col, i+1)
		plt.axis('off')
		plt.imshow(masks[i], cmap='gray')
	
	plt.savefig('./FilterBanks/HDM.png', dpi=500, bbox_inches = 'tight')
	plt.show()

def get_masks(size, angle):
	kernel = np.linspace(-int((size-1)/2), int((size-1)/2), size)
	x, y = np.meshgrid(kernel, kernel)
	coords = [x.flatten(), y.flatten()]
	radius = size/2
	orientation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

	#Generate Left Masks
	vectors_left = np.dot(orientation, coords)
	x, y = vectors_left[0], vectors_left[1]
	mask1 = np.logical_and((x**2 + y**2)<radius**2, y>0)
	#Generate Right Masks
	vectors_right = np.dot(-orientation, coords)
	x, y = vectors_right[0], vectors_right[1]
	mask2 = np.logical_and((x**2 + y**2)<radius**2, y>0)

	mask1 = np.reshape(mask1, (size, size))
	mask2 = np.reshape(mask2, (size, size))

	return mask1, mask2

def generate_half_disc(visualize):
	kernel_size = [9, 19, 29] 
	num_orientations = 8
	orientations = np.arange(0, 180, 180/8)*np.pi/180
	half_discs = []
	for size in kernel_size:
		for i in range(num_orientations):
			m1, m2 = get_masks(size, orientations[i])
			half_discs.append(m1)
			half_discs.append(m2)

	if visualize:
		plot_masks(half_discs, 8, 6)
	
	return half_discs


# generate_half_disc(True)
		
def get_gradient(image, num_bins, half_disc_masks):
	left_masks = half_disc_masks[::2]
	right_masks = half_disc_masks[1::2]

	chi_sqr_vals = []

	for i in range(len(left_masks)):
		dst = (image*0).astype(float)
		for bin in range(num_bins):
			tmp = np.float32(image == bin)

			g_i = cv2.filter2D(tmp, ddepth=-1, kernel=np.float32(left_masks[i]))
			h_i = cv2.filter2D(tmp, ddepth=-1, kernel=np.float32(right_masks[i]))
			dst += (g_i - h_i)**2/(2*(g_i + h_i + 1e-10))

		chi_sqr_vals.append(dst)
	
	chi_sqr_vals = np.array(chi_sqr_vals)

	return np.mean(chi_sqr_vals, axis=0)

def main():
	img_folder = "./BSDS500/Images/"
	# images = os.listdir(img_folder)
	# images.append(images.pop(1))

	output_path = "./ImageRes/"
	
	sobel = "./BSDS500/SobelBaseline/"
	canny = "./BSDS500/CannyBaseline/"

	#Generate Filter Banks
	visualize = True
	#DoG
	DoG_bank = generate_DoG(visualize)

	#LM 
	LMS_bank = generate_LMS(visualize)
	LML_bank = generate_LML(visualize)
	LM_bank = LMS_bank + LML_bank 

	#Gabor
	Gabor_bank = generate_Gabor(visualize)

	filer_bank = DoG_bank + LM_bank + Gabor_bank


	# """
	# Generate Half-disk masks
	# Display all the Half-disk masks and save image as HDMasks.png,
	# use command "cv2.imwrite(...)"
	# """
	half_disc_masks = generate_half_disc(visualize)

	for i in range(10):
	
		img_path = img_folder + str(i+1) + '.jpg'
		img = cv2.imread(img_path)

		#Preprocessing grayscale conversion
		img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

		img_output = output_path + 'Image' + str(i+1) 

		#Create a list to store filter results and pass the image through the filter bank to retrieve results
		filter_res = []
		for filter in filer_bank:
			res = cv2.filter2D(img_gray, ddepth=-1, kernel=filter)
			filter_res.append(res)


		# Generate Texton Map
		# Filter image using oriented gaussian filter bank
		texton_map = get_clusters(filter_res, 'TextonMap')
		# texton_map = cv2.normalize(texton_map,dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
		t_output = img_output + '/TextonMap' + str(i+1) + '.png'
		plt.imshow(texton_map)
		# cmap='nipy_spectral'
		plt.axis('off')
		plt.savefig(t_output, dpi=300, bbox_inches="tight")

		# """
		# Generate Texton Gradient (Tg)
		# Perform Chi-square calculation on Texton Map
		# Display Tg and save image as Tg_ImageName.png,
		# use command "cv2.imwrite(...)"
		# """
		texton_gradient = get_gradient(texton_map, 64, half_disc_masks)
		# texton_gradient =cv2.n
		t_gradient = img_output + '/TextonGradient' + str(i+1) + '.png'
		plt.imshow(texton_gradient)
		plt.axis('off')
		plt.savefig(t_gradient, dpi=300, bbox_inches="tight")



		# """
		# Generate Brightness Map
		# Perform brightness binning 
		# "
		brightness_map = get_clusters(filter_res, 'BrightnessMap')
		# brightness_map = cv2.normalize(brightness_map,dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
		b_output = img_output + '/BrightnessMap' + str(i+1) + '.png'
		plt.imshow(brightness_map)
		plt.axis('off')
		plt.savefig(b_output, dpi=300, bbox_inches="tight")

		# """
		# Generate Brightness Gradient (Bg)
		# Perform Chi-square calculation on Brightness Map
		# Display Bg and save image as Bg_ImageName.png,
		# use command "cv2.imwrite(...)"
		# """

		brightness_gradient = get_gradient(brightness_map, 16, half_disc_masks)
		# texton_gradient =cv2.n
		b_gradient = img_output + '/BrightnessGradient' + str(i+1) + '.png'
		plt.imshow(brightness_gradient)
		plt.axis('off')
		plt.savefig(b_gradient, dpi=300, bbox_inches="tight")



		# """
		# Generate Color Map
		# Perform color binning or clustering
		# """
		color_map = get_clusters(filter_res, 'ColorMap')
		# color_map = cv2.normalize(color_map,dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
		c_output = img_output + '/ColorMap' + str(i+1) + '.png'
		plt.imshow(color_map)
		plt.axis('off')
		plt.savefig(c_output, dpi=300, bbox_inches="tight")

		# """
		# Generate Color Gradient (Cg)
		# Perform Chi-square calculation on Color Map
		# Display Cg and save image as Cg_ImageName.png,
		# use command "cv2.imwrite(...)"
		# """

		color_gradient = get_gradient(color_map, 16, half_disc_masks)
		# texton_gradient =cv2.n
		c_gradient = img_output + '/ColorGradient' + str(i+1) + '.png'
		plt.imshow(color_gradient)
		plt.axis('off')
		plt.savefig(c_gradient, dpi=300, bbox_inches="tight")


		sobel_img_path = sobel + str(i+1) + '.png'
		sobel_img = cv2.imread(sobel_img_path)
		sobel_img = cv2.cvtColor(sobel_img, cv2.COLOR_RGB2GRAY)	

		canny_img_path = canny + str(i+1) + '.png'
		canny_img = cv2.imread(canny_img_path)
		canny_img = cv2.cvtColor(canny_img, cv2.COLOR_RGB2GRAY)	

		pb_lite = np.multiply((0.33*texton_gradient + 0.33*brightness_gradient + 0.33*color_gradient), (0.5*sobel_img + 0.5*canny_img))
		pb_lite_save = img_output + '/pb_lite' + str(i+1) + '.png'
		plt.imshow(pb_lite, cmap='gray')
		plt.axis('off')
		plt.savefig(pb_lite_save, dpi=300, bbox_inches='tight')

	# """
	# Generate texture ID's using K-means clustering
	# Display texton map and save image as TextonMap_ImageName.png,
	# use command "cv2.imwrite('...)"
	# """
""
	# """
	# Read Sobel Baseline
	# use command "cv2.imread(...)"
	# """


	# """
	# Read Canny Baseline
	# use command "cv2.imread(...)"
	# """


	# """
	# Combine responses to get pb-lite output
	# Display PbLite and save image as PbLite_ImageName.png
	# use command "cv2.imwrite(...)"
# 	"""
	
if __name__ == '__main__':
	main()
 



#!/usr/bin/env python
import sys
import cv2
import numpy as np
import bayesplane
import plane
import transformation as tf

def main(args):
	# Declare Test Variables
	# Camera Intrinsics
	fx = 529.29
	fy = 531.28
	px = 466.96
	py = 273.26
	I = np.array([fx, 0 , px, 0, fy, py, 0, 0, 1]).reshape(3,3)
	
	x_start = 459
	x_end = 490
	y_start = 288
	y_end = 314
	rgb_image = cv2.imread("../data/rawData/rgb.jpg")
	depth_image = cv2.imread("../data/rawData/depth.jpg", cv2.IMREAD_ANYDEPTH)
	april_tag_rgb = rgb_image[y_start:y_end, x_start:x_end]
	april_tag_depth = depth_image[y_start:y_end, x_start:x_end]
	# cv2.imshow('april_tag', april_tag_rgb)
	# cv2.waitKey(0)
	all_pts = []
	print april_tag_depth
	for i in range(x_start, x_end):
		for j in range(y_start, y_end):
			depth = depth_image[i,j] / 1000.0
			if(depth != 0):
				x = (i - px) * depth / fx
				y = (i - py) * depth / fy
				all_pts.append([x,y,depth])
	sample_cov = 0.05
	samples = np.array(all_pts)
	print "samples Depth"
	print samples
	cov = np.asarray([sample_cov] * samples.shape[0])
	depth_plane_est = bayesplane.fit_plane_bayes(samples, cov)

	# For now hard code the test data x y values
	# Generate homogenous matrix for pose 
	x_r = 0.970411
	y_r = 0.013975
	z_r = -0.0396695
	w_r = -0.23777
	M = tf.quaternion_matrix([w_r,x_r,y_r,z_r]) 
	x_t = 0.0122108
	y_t = 0.045555
	z_t = 0.831281
	M[0, 3] = x_t
	M[1, 3] = y_t
	M[2, 3] = z_t
	M = np.delete(M, 3, 0)
	print "M"
	print M # pose extrinsics
	origin = np.array([0,0,0,1])
	np.transpose(origin)
	C = np.dot(I, M)
	coord = np.dot(C, origin)
	x_coord = coord[0] / coord[2]
	y_coord = coord[1] / coord[2]
	cv2.circle(rgb_image, (int(x_coord), int(y_coord)), 3, (255, 0,0))
	cv2.imshow('april_tag', rgb_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	x_samples = np.linspace(-0.02, 0.02, num = 10)
	y_samples = np.linspace(-0.02, 0.02, num = 10)
	sample_points = []
	for i in x_samples:
		for j in y_samples:
			sample_points.append([i,j,0,1])
	sample_points = np.transpose(np.array(sample_points))
	sample_points_viz = np.dot(C, sample_points)
	sample_points = np.transpose(np.dot(M, sample_points))
	for i in range(0, 50):
		x_coord = sample_points_viz[0, i] / sample_points_viz[2, i]
		y_coord = sample_points_viz[1, i] / sample_points_viz[2, i]
		cv2.circle(rgb_image, (int(x_coord), int(y_coord)), 3, (255, 0,0))
	cv2.imshow('april_tag', rgb_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	print "Sample_points RGB"
	print  sample_points[1:3, :]
	rgb_plane_est = bayesplane.fit_plane_bayes(sample_points, cov)

if __name__ == '__main__':
	main(sys.argv)
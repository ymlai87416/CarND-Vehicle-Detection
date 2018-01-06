import glob
import math
import os
import cv2
import sys
import traceback
from PIL import Image
import scipy.ndimage
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import skimage.filters
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from sklearn.svm import LinearSVC, SVC
from skimage.feature import hog
from scipy.ndimage.measurements import label
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg
import glob
from sklearn.preprocessing import StandardScaler
import time
from sklearn.svm import LinearSVC, SVC
from sklearn.cross_validation import train_test_split
import pickle

# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the RGB channels separately
    rhist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    ghist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    bhist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Generating bin centers
    bin_edges = rhist[1]
    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return rhist, ghist, bhist, bin_centers, hist_features
    
def bin_spatial(img, size=(32, 32)):
    new_img = cv2.resize(img, size)
    # Use cv2.resize().ravel() to create the feature vector
    features = new_img.ravel() # Remove this line!
    # Return the feature vector
    return features
	
# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        # Use skimage.hog() to get both features and a visualization
        features, hog_image = hog(img, orientations=orient,
                          pixels_per_cell=(pix_per_cell, pix_per_cell), 
                          cells_per_block=(cell_per_block, cell_per_block), 
                          transform_sqrt=False,
                          visualise=True, feature_vector=False)
                          
        return features, hog_image
    else:      
        features = hog(img, orientations=orient,
                          pixels_per_cell=(pix_per_cell, pix_per_cell), 
                          cells_per_block=(cell_per_block, cell_per_block), 
                          transform_sqrt=False,
                          visualise=False, feature_vector=feature_vec)
                          
        return features
		
# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            _,_,_,_,hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features

def evaluate_linear_svc_param(filename, color_space, orient, pix_per_cell, cell_per_block, hog_channel, 
												spatial_size, hist_bins, spatial_feat, hist_feat, hog_feat):
	car_features = extract_features(cars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
	notcar_features = extract_features(notcars, color_space=color_space, 
							spatial_size=spatial_size, hist_bins=hist_bins, 
							orient=orient, pix_per_cell=pix_per_cell, 
							cell_per_block=cell_per_block, 
							hog_channel=hog_channel, spatial_feat=spatial_feat, 
							hist_feat=hist_feat, hog_feat=hog_feat)

	car_features = [x.astype(np.float64) for x in car_features]
	notcar_features = [x.astype(np.float64) for x in notcar_features]

	X = np.vstack((np.array(car_features), np.array(notcar_features))).astype(np.float64)          
	# Fit a per-column scaler
	X_scaler = StandardScaler().fit(X)
	# Apply the scaler to X
	scaled_X = X_scaler.transform(X)

	# Define the labels vector
	y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

	# Split up data into randomized training and test sets
	rand_state = np.random.randint(0, 100)
	X_train, X_test, y_train, y_test = train_test_split(
		scaled_X, y, test_size=0.2, random_state=rand_state)
	
	import sklearn.metrics

	# Use a linear SVC 
	svc = LinearSVC()
	# Check the training time for the SVC
	t=time.time()
	svc.fit(X_train, y_train)
	t2 = time.time()
	print(round(t2-t, 2), 'Seconds to train SVC...')
	# Check the score of the SVC
	y_pred = svc.predict(X_test)
	
	print('Color space: ', color_space)
	print('Using:',orient,'orientations',pix_per_cell,
		'pixels per cell and', cell_per_block,'cells per block')
	print('Hog channels: ', hog_channel)
	print('Spatial size:', spatial_size)
	print('Histogram bin:', hist_bins)
	print('Use spatial feature?:', spatial_feat)
	print('Use histogram feature?:', hist_feat)
	print('Use hog feature?:', hog_feat)
	print('Feature vector length:', len(X_train[0]))
	print('Car example: ', len(cars))
	print('Non car example: ', len(notcars))
	acc_score = round(sklearn.metrics.accuracy_score(y_test, y_pred), 4)
	print('Test Accuracy of SVC = ', acc_score)
	
	with open(filename, 'a') as f:
		f.write("%s, %d, %d, %d, %s, %d, %d, %s, %s, %s, %.4f\n" % (color_space, orient, pix_per_cell, cell_per_block, str(hog_channel), spatial_size[0], hist_bins, spatial_feat, hist_feat, hog_feat, acc_score))
	f.closed

if __name__ == '__main__':

	color_spacex = sys.argv[1]

	# Divide up into cars and notcars
	images = glob.glob('../vehicle_dataset/*.jpeg')
	cars = []
	notcars = []
	for image in images:
		if 'image' in image or 'extra' in image:
			notcars.append(image)
		else:
			cars.append(image)

	# Reduce the sample size because HOG features are slow to compute
	# The quiz evaluator times out after 13s of CPU time
	sample_size = 1000
	cars = cars[0:sample_size]
	notcars = notcars[0:sample_size]	

	color_space = 'LUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
	orient = 9  # HOG orientations
	pix_per_cell = 8 # HOG pixels per cell
	cell_per_block = 2 # HOG cells per block
	hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
	spatial_size = (32, 32) # Spatial binning dimensions
	hist_bins = 32    # Number of histogram bins
	spatial_feat = True # Spatial features on or off
	hist_feat = True # Histogram features on or off
	hog_feat = True # HOG features on or off
	y_start_stop = [None, None] # Min and max in y to search in slide_window()
	
	color_space_list = [color_spacex]
	orient_list = [9, 5]
	pix_per_cell_list = [8, 16]
	cell_per_block_list = [2,3,4]
	hog_channel_list = [0, 1, 2, "ALL"]
	spatial_size_list = [(16,16), (32, 32)]
	hist_bins_list = [16, 32]
	spatial_feat_list = [True, False]
	hist_feat_list = [True, False]
	hog_feat_list = [True]
	
	for color_space in color_space_list:
		for orient in orient_list:
			for pix_per_cell in pix_per_cell_list:
				for cell_per_block in cell_per_block_list:
					for hog_channel in hog_channel_list:
						for spatial_size in spatial_size_list:
							for hist_bins in hist_bins_list:
								for spatial_feat in spatial_feat_list:
									for hist_feat in hist_feat_list:
										for hog_feat in hog_feat_list:
											evaluate_linear_svc_param('param_test'+color_space+'.txt', color_space, orient, pix_per_cell, cell_per_block, hog_channel, 
												spatial_size, hist_bins, spatial_feat, hist_feat, hog_feat) 
	



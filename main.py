import model as nn
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
import scipy.misc
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
import sys, getopt
import os  
import cv2
import warnings


figures = False
image = None 



# parse command line options
try:
      opts, args = getopt.getopt(sys.argv[1:],"hgi:",["Image="])
except getopt.GetoptError:
      print('USAGE: python3 main.py -i <imagefile.jpg>')
      sys.exit(2)


if len(opts) == 0 or '-i' not in opts[0] and '--Image' not in opts[0] and '-h' not in opts[0]:
	print('USAGE: python3 main.py -i <imagefile.jpg>')
	sys.exit(2)

for opt, arg in opts:
	if opt == '-h':
		print('USAGE: python3 main.py -i <imagefile.jpg>')
		print('Options:')
		print('-i <imagefile.jpg> or --Image=<imagefile.jpg>: the input image file to be')
		print('                                               classified')
		print('-g: plot the figures and graphs while training the model', end = '')
		print('-h: help menu')
		sys.exit(0)
	if opt in ('-i', '--Image'):
		image = arg	
	if opt in('-g'):
		figures = True

# checks if the file exists
if not os.path.isfile(image):
	print('Cannot open image: ' + image)
	sys.exit(3)


# load the data set
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()



# plot an image from the data set
# Example of a picture
index = 25
plt.imshow(train_set_x_orig[index])

if figures:
	print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")
	plt.show()


m_train = train_set_x_orig.shape[0]   # train set size
m_test = test_set_x_orig.shape[0]	  # test set size
num_px = train_set_x_orig.shape[1]	  # the number of pixels in the image


# flatten the data set into a (num_px * num_px * 3, m) matrix

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T



# Normalize (one common normalization would be (x[i] - mean) / (std_dev) )
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.



# create and train the neural network
layers_dims = [12288, 32, 16, 8, 1]
parameters = nn.nn_model(train_set_x, train_set_y, layers_dims, num_iterations = 3000, print_cost=figures)
nn.calculate_accuracy(test_set_x, test_set_y, parameters)



# We preprocess the image to fit your algorithm.
fname = image

image = cv2.imread(fname)
image = cv2.resize(image, dsize=(num_px,num_px), interpolation=cv2.INTER_CUBIC)
my_image = image.reshape((1, num_px*num_px*3)).T

plt.imshow(image)
plt.show()



my_predicted_image = nn.predict(my_image, parameters)



print("\nThis looks like a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")




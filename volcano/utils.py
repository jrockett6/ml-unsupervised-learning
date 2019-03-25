import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import pickle
import numpy as np
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from PIL import Image
import os


def get_picture_resized(df_data, row_loc, pic_res):
	data = (np.array((df_data.iloc[row_loc]), dtype=np.int8))
	data.resize(110, 110)
	data = data/130
	img = Image.fromarray(data)
	img = img.resize((pic_res, pic_res), resample=Image.BILINEAR)
	new_arr = np.array(img)
	new_arr.resize((1, pic_res*pic_res))

	return new_arr

def resize_images(df_data, pic_res):
	new_arr = np.array([])

	for i in range(len(df_data)):
		new_row = get_picture_resized(df_data, i, pic_res)

		if not new_arr.size:
			new_arr = new_row
		else:
			new_arr = np.vstack((new_arr, new_row))

	return new_arr


def load_data(dtree=False, knn=False):
	image_res = 55

	train_data_labels = read_csv("data/train_labels.csv")
	train_data_images = read_csv("data/train_images.csv", header=None)

	test_data_labels = read_csv("data/test_labels.csv")
	test_data_images = read_csv("data/test_images.csv", header=None)

	y_train = train_data_labels['Volcano?']
	y_test = test_data_labels['Volcano?']

	# If you want to view/save picture
	# view_picture(train_data_images, 2)

	x_train = resize_images(train_data_images, image_res)
	x_test = resize_images(test_data_images, image_res)

	train_test_data = {'x_train': x_train, 
					   'y_train': y_train,
					   'x_test': x_test, 
					   'y_test': y_test}	


	with open('train_test_data.pkl', 'wb') as output:
		pickle.dump(train_test_data, output, -1)	   

	print('(Training data for images loaded successfully)\n')


def read_data():
	with open('train_test_data.pkl', 'rb') as input:
		train_test_data = pickle.load(input)

	x_train = train_test_data['x_train']
	y_train = train_test_data['y_train']

	x_test = train_test_data['x_test']
	y_test = train_test_data['y_test']

	print('(Training data read successfully)\n')

	return x_train, y_train, x_test, y_test



def view_picture(df_data, row_loc, pic_res=110):
	data = (np.array((df_data.iloc[row_loc]), dtype=np.int8))
	data.resize(110, 110)
	img = Image.fromarray(data)
	# img = img.resize((pic_res, pic_res), resample=Image.BILINEAR)
	# img.save("out_data/volcano_res_" + str(pic_res) + ".png", format="PNG")
	img.show()


def test_metrics(y_predict, y_test):
	conf_mat = confusion_matrix(y_test, y_predict)

	true_pos = conf_mat[1][1]
	false_pos = conf_mat[0][1]
	false_neg = conf_mat[1][0]
	if (true_pos + false_pos) == 0:
		print('Entirely negative guesses')
		return 0

	precision = true_pos/(true_pos + false_pos)
	recall = true_pos/(true_pos + false_neg)
	f1 = 2 * (precision*recall) / (precision+recall)

	print('Recall: ', str(recall))
	print('Precision: ', str(precision))
	print('F-measure: ', str(f1))
	print('Accuracy: ', str(accuracy_score(y_test, y_predict)*100))
	print('Confusion matrix:')
	print(conf_mat)
	print()
	return f1


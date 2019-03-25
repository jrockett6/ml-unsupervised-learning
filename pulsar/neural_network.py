import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os

import numpy as np 
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU
from keras.optimizers import Adam
from utils import *
from dr import *
from math import sqrt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class f1_calc(Callback):
	def __init__(self, x_train, y_train):
		self.f1_trains = []
		self.f1_tests =[]
		self.x_train = x_train
		self.y_train = y_train

	def on_epoch_end(self, epoch, logs={}):
		y_predict_test = np.array(self.model.predict(self.validation_data[0]))
		y_predict_test = np.argmax(y_predict_test, axis=1)
		y_predict_train = np.array(self.model.predict(self.x_train))
		y_predict_train = np.argmax(y_predict_train, axis=1)
		y_test = self.validation_data[1]

		f1_train = f1_score(self.y_train, y_predict_train)
		f1_test = f1_score(y_test, y_predict_test)

		print("F1 test: {}, F1 train: {}".format(f1_test, f1_train))

		self.f1_trains.append(f1_train)
		self.f1_tests.append(f1_test)

def get_dr_features(x_data):
	x_thresh = variance_threshold(x_data)
	x_rand = randomized_projection(x_data)
	x_princ = principal_component(x_data)
	x_ind = indpendent_component(x_data)

	return [x_thresh, x_rand, x_princ, x_ind]

def train_model(x_train, y_train, x_test, y_test, history, iterats=1, inputs=4):
	model = Sequential()
	model.add(Dense(units=8, input_dim=inputs))
	model.add(Dense(units=8))
	model.add(LeakyReLU(alpha = 0.1))
	model.add(Dense(units=8))
	model.add(LeakyReLU(alpha = 0.1))
	model.add(Dense(units=2, activation='softmax'))

	model.compile(Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

	model.fit(x_train, y_train, epochs=iterats, validation_data=(x_test, y_test), callbacks=[history])

	return model

def test_metrics_nn(x_test, y_test, model):
	y_predict = model.predict(x_test)
	y_predict = np.argmax(y_predict, axis=1)
	conf_mat = confusion_matrix(y_test, y_predict)

	true_pos = conf_mat[1][1]
	false_pos = conf_mat[0][1]
	false_neg = conf_mat[1][0]
	if (true_pos + false_pos) == 0:
		return 0

	precision = true_pos/(true_pos + false_pos)
	print('Precision: ', str(precision))
	recall = true_pos/(true_pos + false_neg)
	print('Recall: ', str(recall))
	f1 = 2 * (precision*recall) / (precision+recall)
	print('F-measure: ', str(f1))
	print('Accuracy: ', str(accuracy_score(y_test, y_predict)*100))
	print('Confusion matrix:')
	print(conf_mat)
	print()
	return conf_mat

def plot_learning_curves(x_train, y_train, x_test, y_test, trials=1, epochs=20):
	# f1_train = [0 for i in range(epochs)]
	# f1_test = [0 for i in range(epochs)]
	labels = ['thresholded','rand proj','PCA', 'ICA', 'Original']
	f1s = [[0 for i in range(epochs)] for i in range(len(x_train))]

	for j in range(len(x_train)):
		for i in range(trials):
			history = f1_calc(x_train[j], y_train)
			model = train_model(x_train[j], y_train, x_test[j], y_test, history, epochs, len(x_train[j][0]))
			# f1_train = [sum(x) for x in zip(f1_train, history.f1_trains)]
			f1s[j] = [sum(x) for x in zip(f1s[j], history.f1_tests)]
		print('\n' + str(labels[j]))
		test_metrics_nn(x_test[j], y_test, model)

	# f1_train = [x/trials for x in f1_train]
	f1s = [[x/trials for x in f1_test] for f1_test in f1s]

	# plt.plot(f1s[0], color='#9CBA7F', linewidth=2)
	colors = ["salmon","dodgerblue", "springgreen", "darkorange", "mediumorchid", "aqua"]
	for i in range(len(f1s)):
		plt.plot([i+1 for i in range(epochs)], f1s[i], color=colors[i], linewidth=2)
	plt.xlabel('Epoch')
	plt.ylabel('F-measure')
	plt.title('Model F-measure')
	plt.grid(True)
	plt.legend(labels, loc='lower right')
	plt.savefig("out_data/nn_learning_curve")
	plt.show()

	return model

def plot_learning_curves_cluster(x_train, y_train, x_test, y_test, trials=1, epochs=150):
	labels = ['kmeans', 'EM']
	f1s = [[0 for i in range(epochs)] for i in range(len(x_train))]

	for j in range(len(x_train)):
		for i in range(trials):
			history = f1_calc(x_train[j], y_train)
			model = train_model(x_train[j], y_train, x_test[j], y_test, history, epochs, len(x_train[j][0]))
			f1s[j] = [sum(x) for x in zip(f1s[j], history.f1_tests)]
		print('\n' + str(labels[j]))
		test_metrics_nn(x_test[j], y_test, model)

	f1s = [[x/trials for x in f1_test] for f1_test in f1s]

	colors = ["salmon", "dodgerblue", "springgreen", "darkorange", "mediumorchid", "aqua"]
	for i in range(len(f1s)):
		plt.plot([i+1 for i in range(epochs)], f1s[i], color=colors[i], linewidth=2)
	plt.xlabel('Epoch')
	plt.ylabel('F-measure')
	plt.title('Model F-measure')
	plt.grid(True)
	plt.legend(labels, loc='lower right')
	plt.savefig("out_data/nn_learning_curve")
	plt.show()

	return model

def run_dr_nn(x_train, y_train, x_test, y_test):
	x_train_new = get_dr_features(x_train)
	x_test_new = get_dr_features(x_test)
	x_train_new.append(x_train)
	x_test_new.append(x_test)

	model = plot_learning_curves(x_train_new, y_train, x_test_new, y_test)

def run_cluster_nn(x_train, y_train, x_test, y_test, x_data, y_data):
	kmeans = KMeans(n_clusters=7).fit(x_data)
	labels = (kmeans.labels_).reshape(-1, 1)
	enc = OneHotEncoder(categories='auto').fit(labels)
	x_kmeans = enc.transform(labels).toarray()
	x_train_kmeans, x_test_kmeans, nan1, nan2 = train_test_split(x_kmeans, y_data, test_size=0.25, random_state=0)

	em = GaussianMixture(n_components=10,covariance_type='full').fit(x_train)
	labels = em.predict(x_train).reshape(-1, 1)
	labels_test = em.predict(x_test).reshape(-1, 1)
	enc = OneHotEncoder(categories='auto').fit(labels)
	x_train_em = enc.transform(labels).toarray()
	x_test_em = enc.transform(labels_test).toarray()

	x_train = [x_train_kmeans, x_train_em]
	x_test = [x_test_kmeans, x_test_em]

	model = plot_learning_curves_cluster(x_train, y_train, x_test, y_test)

def main():
	x_train, y_train, x_test, y_test, x_data, y_data, column_scales, column_min = read_data()

	#RUN NN ON DR DATA
	# run_dr_nn(x_train, y_train, x_test, y_test)

	#RUN NN ON CLUSTERED DR DATA
	# run_cluster_nn(x_train, y_train, x_test, y_test, x_data, y_data)



if __name__ == '__main__':
	main()

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import pickle
import numpy as np
from pandas import read_csv
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def read_data():
	data = read_csv("data/pulsar_stars.csv")
	y_data = np.array(data['target_class'])
	x = data.drop(['target_class'], axis=1)
	column_scales = [(np.max(x[column]) - np.min(x[column])) for column in x]
	column_min = [np.min(x[column]) for column in x]
	x_data = np.array([(x[column] - np.min(x[column]))/(np.max(x[column]) - np.min(x[column])) for column in x]).transpose() #scale between 0/1

	x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=0)
	return x_train, y_train, x_test, y_test, x_data, y_data, column_scales, column_min


def prediction_mapped(x_data, labels_mapped):
	predict_noise = []
	predict_pulsar = []

	for i in range(len(labels_mapped)):
		if labels_mapped[i] == 0:
			predict_noise.append(x_data[i])
		else:
			predict_pulsar.append(x_data[i])

	return np.array(predict_noise).transpose(), np.array(predict_pulsar).transpose()

def prediction_clusters(x_data, labels):
	cluster_labels = [[] for i in range(7)]

	for i in range(len(labels)):
		cluster_labels[labels[i]].append(x_data[i])

	for i in range(len(cluster_labels)):
		cluster_labels[i] = np.array(cluster_labels[i]).transpose()

	return cluster_labels


def view_data(column_scales, column_min, x_data, labels, labels_mapped):
	data = read_csv("data/pulsar_stars.csv")
	keys = list(data)

	predict_noise, predict_pulsar = prediction_mapped(x_data, labels_mapped)
	cluster_labels = prediction_clusters(x_data, labels)

	for i in range(len(predict_pulsar)):
		predict_pulsar[i] = predict_pulsar[i]*column_scales[i] + column_min[i]
		predict_noise[i] = predict_noise[i]*column_scales[i] + column_min[i]

	for i in range(len(cluster_labels)):
		for j in range(len(cluster_labels[i])):
			cluster_labels[i][j] = cluster_labels[i][j]*column_scales[j] + column_min[j]

	noise_points = data[data.target_class==0]
	pulsar_points = data[data.target_class==1]

	fig, ax = plt.subplots(4, 3)
	alph = 0.1
	
	#Original/True Labels
	ax[0, 0].scatter(noise_points[keys[1]], noise_points[keys[2]],label="RF Noise",color="red",alpha=alph,s=10)
	ax[0, 0].scatter(pulsar_points[keys[1]], pulsar_points[keys[2]],label="Pulsar",color="purple",alpha=alph,s=10)
	ax[0, 0].set_xlabel(keys[1])
	ax[0, 0].set_ylabel(keys[2])
	ax[0, 0].set_title('Actual Labels')
	leg = ax[0, 0].legend()
	for lh in leg.legendHandles: 
		lh.set_alpha(1)

	ax[1, 0].scatter(noise_points[keys[0]], noise_points[keys[3]],label="RF Noise",color="red",alpha=alph,s=10)
	ax[1, 0].scatter(pulsar_points[keys[0]], pulsar_points[keys[3]],label="Pulsar",color="purple",alpha=alph,s=10)
	ax[1, 0].set_xlabel(keys[0])
	ax[1, 0].set_ylabel(keys[3])

	ax[2, 0].scatter(noise_points[keys[4]], noise_points[keys[7]],label="RF Noise",color="red",alpha=alph,s=10)
	ax[2, 0].scatter(pulsar_points[keys[4]], pulsar_points[keys[7]],label="Pulsar",color="purple",alpha=alph,s=10)
	ax[2, 0].set_xlabel(keys[4])
	ax[2, 0].set_ylabel(keys[7])

	ax[3, 0].scatter(noise_points[keys[5]], noise_points[keys[6]],label="RF Noise",color="red",alpha=alph,s=10)
	ax[3, 0].scatter(pulsar_points[keys[5]], pulsar_points[keys[6]],label="Pulsar",color="purple",alpha=alph,s=10)
	ax[3, 0].set_xlabel(keys[5])
	ax[3, 0].set_ylabel(keys[6])


	#Clusters
	colors = ["dodgerblue", "salmon", "springgreen", "yellow", "darkorange", "aqua", "mediumorchid"]
	for i in range(len(cluster_labels)):
		label = "Cluster " + str(i+1)
		ax[0, 1].scatter(cluster_labels[i][1], cluster_labels[i][2],label=label,color=colors[i],alpha=alph,s=10)
	ax[0, 1].set_xlabel(keys[1])
	ax[0, 1].set_title('Cluster Labels')
	leg = ax[0, 1].legend()
	for lh in leg.legendHandles: 
		lh.set_alpha(1)

	colors = ["dodgerblue", "salmon", "springgreen", "yellow", "darkorange", "aqua", "mediumorchid"]
	for i in range(len(cluster_labels)):
		label = "Cluster " + str(i+1)
		ax[1, 1].scatter(cluster_labels[i][0], cluster_labels[i][3],label=label,color=colors[i],alpha=alph,s=10)
	ax[1, 1].set_xlabel(keys[0])

	colors = ["dodgerblue", "salmon", "springgreen", "yellow", "darkorange", "aqua", "mediumorchid"]
	for i in range(len(cluster_labels)):
		label = "Cluster " + str(i+1)
		ax[2, 1].scatter(cluster_labels[i][4], cluster_labels[i][7],label=label,color=colors[i],alpha=alph,s=10)
	ax[2, 1].set_xlabel(keys[4])

	colors = ["dodgerblue", "salmon", "springgreen", "yellow", "darkorange", "aqua", "mediumorchid"]
	for i in range(len(cluster_labels)):
		label = "Cluster " + str(i+1)
		ax[3, 1].scatter(cluster_labels[i][5], cluster_labels[i][6],label=label,color=colors[i],alpha=alph,s=10)
	ax[3, 1].set_xlabel(keys[5])


	#Predicted Labels
	ax[0, 2].scatter(predict_noise[1], predict_noise[2],label="Predicted Noise",color="dodgerblue",alpha=alph,s=10)
	ax[0, 2].scatter(predict_pulsar[1], predict_pulsar[2],label="Predicted Pulsar",color="forestgreen",alpha=alph,s=10)
	ax[0, 2].set_xlabel(keys[1])
	ax[0, 2].set_title('Predicted Labels')
	leg = ax[0, 2].legend()
	for lh in leg.legendHandles: 
		lh.set_alpha(1)

	ax[1, 2].scatter(predict_noise[0], predict_noise[3],label="Predicted Noise",color="dodgerblue",alpha=alph,s=10)
	ax[1, 2].scatter(predict_pulsar[0], predict_pulsar[3],label="Predicted Pulsar",color="forestgreen",alpha=alph,s=10)
	ax[1, 2].set_xlabel(keys[0])

	ax[2, 2].scatter(predict_noise[4], predict_noise[7],label="Predicted Noise",color="dodgerblue",alpha=alph,s=10)
	ax[2, 2].scatter(predict_pulsar[4], predict_pulsar[7],label="Predicted Pulsar",color="forestgreen",alpha=alph,s=10)
	ax[2, 2].set_xlabel(keys[4])

	ax[3, 2].scatter(predict_noise[5], predict_noise[6],label="Predicted Noise",color="dodgerblue",alpha=alph,s=10)
	ax[3, 2].scatter(predict_pulsar[5], predict_pulsar[6],label="Predicted Pulsar",color="forestgreen",alpha=alph,s=10)
	ax[3, 2].set_xlabel(keys[5])


	# ax[0, 0].scatter(cluster_centers[0][1], cluster_centers[0][2],label="Noise Centroid (predicted)",color="sandybrown",alpha=1.0,marker="*",s=100)
	# ax[0, 0].scatter(cluster_centers[1][1], cluster_centers[1][2],label="Pulsar Centroid (predicted)",color="royalblue",alpha=1.0,marker="*",s=100)
	# ax[0, 1].scatter(cluster_centers[0][1], cluster_centers[0][2],label="Noise Centroid (predicted)",color="aqua",alpha=1.0,marker="*",s=100)
	# ax[0, 1].scatter(cluster_centers[1][1], cluster_centers[1][2],label="Pulsar Centroid (predicted)",color="darkgreen",alpha=1.0,marker="*",s=100)
	# ax[1, 0].scatter(cluster_centers[0][0], cluster_centers[0][3],label="Cluster 2 Centroid",color="sandybrown",alpha=1.0,marker="*",s=100)
	# ax[1, 0].scatter(cluster_centers[1][0], cluster_centers[1][3],label="Cluster 1 Centroid",color="royalblue",alpha=1.0,marker="*",s=100)
	# ax[1, 1].scatter(cluster_centers[0][0], cluster_centers[0][3],label="Noise Centroid",color="aqua",alpha=1.0,marker="*",s=100)
	# ax[1, 1].scatter(cluster_centers[1][0], cluster_centers[1][3],label="Pulsar Centroid",color="darkgreen",alpha=1.0,marker="*",s=100)
	# ax[2, 0].scatter(cluster_centers[0][4], cluster_centers[0][7],label="Cluster 2 Centroid",color="sandybrown",alpha=1.0,marker="*",s=100)
	# ax[2, 0].scatter(cluster_centers[1][4], cluster_centers[1][7],label="Cluster 1 Centroid",color="royalblue",alpha=1.0,marker="*",s=100)
	# ax[2, 1].scatter(cluster_centers[0][4], cluster_centers[0][7],label="Noise Centroid",color="aqua",alpha=1.0,marker="*",s=100)
	# ax[2, 1].scatter(cluster_centers[1][4], cluster_centers[1][7],label="Pulsar Centroid",color="darkgreen",alpha=1.0,marker="*",s=100)
	# ax[3, 0].scatter(cluster_centers[0][5], cluster_centers[0][6],label="Cluster 2 Centroid",color="sandybrown",alpha=1.0,marker="*",s=100)
	# ax[3, 0].scatter(cluster_centers[1][5], cluster_centers[1][6],label="Cluster 1 Centroid",color="royalblue",alpha=1.0,marker="*",s=100)
	# ax[3, 1].scatter(cluster_centers[0][5], cluster_centers[0][6],label="Noise Centroid",color="aqua",alpha=1.0,marker="*",s=100)
	# ax[3, 1].scatter(cluster_centers[1][5], cluster_centers[1][6],label="Pulsar Centroid",color="darkgreen",alpha=1.0,marker="*",s=100)

	fig.suptitle('Pulsar Star vs RF Noise Attributes')
	plt.show()


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

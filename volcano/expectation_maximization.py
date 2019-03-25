from utils import *
from dr import *
from sklearn.metrics import accuracy_score, f1_score, homogeneity_score, completeness_score, v_measure_score
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import SparseRandomProjection
from scipy.stats import kurtosis

def elbow_curve(x_train):
	k_range = range(1,11)
	em = [GaussianMixture(n_components=i) for i in k_range]
	score = [-em[i].fit(x_train).bic(x_train) for i in range(len(em))]

	plt.plot(k_range, score,color='#9CBA7F',linewidth=2)
	# plt.scatter(10, score[9],facecolors='none',edgecolors='r',s=200,linewidth=2)
	plt.xlabel('K (# of clusters)')
	plt.ylabel('Score (BIC)')
	plt.title('EM Elbow Curve')
	plt.grid(True)
	plt.show()

def classifier_accuracy(x_train, y_train, x_test, y_test):
	em = GaussianMixture(n_components=4,covariance_type='full').fit(x_train)
	prediction_clusters_rounded = map_labels(y_train, em.predict(x_train))
	y_predict = em.predict(x_test)
	labels_mapped = [prediction_clusters_rounded[label] for label in y_predict]
	test_metrics(labels_mapped, y_test)

def get_scores(x_data, y_data):
	em = GaussianMixture(n_components=4,covariance_type='full').fit_predict(x_data)
	map_labels(y_data, em)
	print("Homogeneity score : " + str(homogeneity_score(y_data, em)))
	print("Completeness score : " + str(completeness_score(y_data, em)))
	print("V-measure score : " + str(v_measure_score(y_data, em)))	

def map_labels(y_data, em):
	num_clusters = 4
	labels = em
	tot_samples = [0 for i in range(num_clusters)]
	predicted_samples = [0 for i in range(num_clusters)]

	for i in range(len(labels)):

		tot_samples[labels[i]] += 1
		predicted_samples[labels[i]] += y_data[i]

	prediction_clusters = [predicted_samples[i]/tot_samples[i] for i in range(num_clusters)]
	prediction_clusters_rounded = [round(float(val)) for val in prediction_clusters]
	labels_mapped = [prediction_clusters_rounded[label] for label in labels]

	# print('Cluster ratios:')
	# for val in prediction_clusters:
	# 	if val < 0.5:
	# 		print(1-val)
	# 	else:
	# 		print(val)

	return prediction_clusters_rounded

def elbow_curve_dr(x_new):
	score = []

	for x_data in x_new:
		k_range = range(1,8)
		em = [GaussianMixture(n_components=i) for i in k_range]
		score.append([em[i].fit(x_data).bic(x_data) for i in range(len(em))])

	print(score[3])
	colors = ["salmon","dodgerblue", "springgreen", "darkorange", "aqua", "mediumorchid"]
	for i in range(len(x_new)):
		plt.plot(k_range, score[i],color=colors[i],linewidth=2)
	plt.xlabel('K (# of clusters)')
	plt.ylabel('Score (BIC)')
	plt.title('EM Elbow Curve')
	plt.legend(['thresholded','rand proj','PCA', 'ICA'])
	plt.grid(True)
	plt.show()


def get_dr_features(x_data):
	x_thresh = variance_threshold(x_data)
	x_rand = randomized_projection(x_data)
	x_princ = principal_component(x_data)
	x_ind = indpendent_component(x_data)

	return [x_thresh, x_rand, x_princ, x_ind]


def dim_reduction_exp(x_train, y_train, x_test, y_test, x_data, y_data):
	dr_names = ['x_thresh', 'x_rand', 'x_pca', 'x_ica']
	x_data_new = get_dr_features(x_data)
	x_train_new = get_dr_features(x_train)
	x_test_new = get_dr_features(x_test)

	elbow_curve_dr(x_train_new)

	print('--------------------------------------------------')
	for i in range(len(x_data_new)):
		print(dr_names[i])
		get_scores(x_data_new[i], y_data)
		classifier_accuracy(x_train_new[i], y_train, x_test_new[i], y_test)	

def main():
	if not os.path.isfile("train_test_data.pkl"):
		load_data()
	x_train, y_train, x_test, y_test = read_data()
	x_data = np.vstack((x_train, x_test))
	y_train = np.array(y_train).reshape(-1,1)
	y_test = np.array(y_test).reshape(-1,1)
	y_data = np.vstack((y_train, y_test)).reshape(-1,)

	# #PLOT ELBOW CURVE
	# elbow_curve(x_train)

	# #GET SCORES
	# get_scores(x_data, y_data)

	# #GET CLASSFIER METRICS
	# classifier_accuracy(x_train, y_train, x_test, y_test)

	# #ON DIMENSIONALITY REDUCED DATA
	# dim_reduction_exp(x_train, y_train, x_test, y_test, x_data, y_data)

if __name__ == '__main__':
	main()


# https://colab.research.google.com/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/05.12-Gaussian-Mixtures.ipynb#scrollTo=_57oq2ZnnSbq
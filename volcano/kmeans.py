from utils import *
from dr import *
from sklearn.metrics import accuracy_score, f1_score, homogeneity_score, completeness_score, v_measure_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import SparseRandomProjection
from scipy.stats import kurtosis

# def plot_learning_curve(x_train, y_train, x_test, y_test, number_components):
# 	f1_train_pca = []
# 	f1_train_ica = []
# 	f1_train_rca = []

# 	for i in range(1, number_components+1):
# 		pca = PCA(n_components=i)
# 		ica = FastICA(n_components=8,whiten=True,random_state=4)
# 		rca = SparseRandomProjection(n_components=8,random_state=0)


# 		x_train_pca = pca.fit_transform(x_train)
# 		x_test_pca = pca.fit_transform(x_test)
# 		x_train_ica = (ica.fit_transform(x_train))
# 		x_test_ica = (ica.fit_transform(x_test))
# 		x_train_rca = (rca.fit_transform(x_train))
# 		x_test_rca = (rca.fit_transform(x_test))

# 		x_train_ica = x_train_ica[:,i-1].reshape(-1, 1)
# 		x_test_ica = x_test_ica[:,i-1].reshape(-1, 1)
# 		x_train_rca = x_train_rca[:,i-1].reshape(-1, 1)
# 		x_test_rca = x_test_rca[:,i-1].reshape(-1, 1)
# 		print(np.var(x_train_rca))


# 		kmeans_pca = KMeans(n_clusters=2, init='random', random_state=0, max_iter=1000).fit(x_train_pca)
# 		kmeans_ica = KMeans(n_clusters=2, init='random', random_state=0, max_iter=1000).fit(x_train_ica)
# 		kmeans_rca = KMeans(n_clusters=2, init='random', random_state=0, max_iter=1000).fit(x_train_rca)

# 		y_predict_pca = kmeans_pca.predict(x_test_pca)
# 		y_predict_ica = kmeans_ica.predict(x_test_ica)
# 		y_predict_rca = kmeans_rca.predict(x_test_rca)

# 		f1_train_pca.append(f1_score(y_test, y_predict_pca))
# 		f1_train_ica.append(f1_score(y_test, y_predict_ica))
# 		f1_train_rca.append(f1_score(y_test, y_predict_rca))

# 	rng = [i for i in range(1,9)]
# 	plt.plot(rng, f1_train_pca, color='#9CBA7F', linewidth=2)
# 	plt.plot(rng, f1_train_ica, color='#8A2BE2', linewidth=2)
# 	plt.plot(rng, f1_train_rca, color='salmon', linewidth=2)
# 	plt.xlabel('Component')
# 	plt.ylabel('F-measure')
# 	plt.title('Kmeans F-measure')
# 	plt.legend(['PCA', 'ICA', 'RCA'], loc='upper right')
# 	plt.grid(True)
# 	plt.show()


def elbow_curve(x_train):
	k_range = range(1,11)
	k_means = [KMeans(n_clusters=i) for i in k_range]
	score = [-k_means[i].fit(x_train).score(x_train) for i in range(len(k_means))]

	plt.plot(k_range, score,color='#9CBA7F',linewidth=2)
	plt.scatter(4, score[3],facecolors='none',edgecolors='r',s=200,linewidth=2)
	plt.xlabel('K (# of clusters)')
	plt.ylabel('Score (SSE)')
	plt.title('Kmeans Elbow Curve')
	plt.grid(True)
	plt.show()

def get_scores(x_data, y_data):
	kmeans = KMeans(n_clusters=4).fit(x_data)
	print("Homogeneity score : " + str(homogeneity_score(y_data, kmeans.labels_)))
	print("Completeness score : " + str(completeness_score(y_data, kmeans.labels_)))
	print("V-measure score : " + str(v_measure_score(y_data, kmeans.labels_)))


def classifier_accuracy(x_train, y_train, x_test, y_test):
	kmeans = KMeans(n_clusters=4).fit(x_train)
	labels_mapped, prediction_clusters, prediction_clusters_rounded = map_labels(y_train, kmeans)
	y_predict = kmeans.predict(x_test)
	labels_mapped = [prediction_clusters_rounded[label] for label in y_predict]
	test_metrics(labels_mapped, y_test)

def map_labels(y_data, clf):
	labels = clf.labels_
	num_clusters = 4

	tot_samples = [0 for i in range(num_clusters)]
	predicted_samples = [0 for i in range(num_clusters)]

	for i in range(len(labels)):
		tot_samples[labels[i]] += 1
		predicted_samples[labels[i]] += y_data[i]

	prediction_clusters = [predicted_samples[i]/tot_samples[i] for i in range(num_clusters)]
	prediction_clusters_rounded = [round(float(val)) for val in prediction_clusters]
	labels_mapped = [prediction_clusters_rounded[label] for label in labels]	

	return labels_mapped, prediction_clusters, prediction_clusters_rounded

def elbow_curve_dr(x_new):
	score = []

	for x_data in x_new:
		k_range = range(1,21)
		k_means = [KMeans(n_clusters=i) for i in k_range]
		score.append([-k_means[i].fit(x_data).score(x_data) for i in range(len(k_means))])

	colors = ["mediumorchid","dodgerblue", "springgreen", "darkorange", "aqua", "mediumorchid"]
	for i in range(len(x_new)):
		plt.plot(k_range, score[i],color=colors[i],linewidth=2)
	plt.xlabel('K (# of clusters)')
	plt.ylabel('Score (SSE)')
	plt.title('Kmeans Elbow Curve')
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

	# elbow_curve_dr(x_train_new)

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

	# # PLOT ELBOW CURVE
	# elbow_curve(x_train)

	# # GET SCORES
	# get_scores(x_train, y_train)

	# # GET CLASSIFICATION METRICS
	# classifier_accuracy(x_train, y_train, x_test, y_test)

	# # ON DIMENSIONALITY REDUCED DATA
	# dim_reduction_exp(x_train, y_train, x_test, y_test, x_data, y_data)




if __name__ == '__main__':
	main()



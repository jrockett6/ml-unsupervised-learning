from utils import *
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from scipy.stats import kurtosis


def variance_threshold(x_data):
	x_data = x_data.transpose()
	explained_variance = [np.var(x_data[i]) for i in range(len(x_data))]
	explained_variance_ratio = [var/sum(explained_variance) for var in explained_variance]
	print('Threshold variance per feature : ')
	print(explained_variance)
	print('Threshold explained variance : ')
	print(explained_variance_ratio)
	print()
	x_data = np.vstack((x_data[0], x_data[4], x_data[5], x_data[6])).transpose()

	return x_data

def randomized_projection(x_data):
	avg_explained_variance = [0 for i in range(8)]
	iters = 1
	for x in range(iters):
		rca = GaussianRandomProjection(n_components=8)
		x_rca = rca.fit_transform(x_data)
		for i in range(8):
			avg_explained_variance[i] += np.var(x_rca[:,i].reshape(-1, 1))

	avg_explained_variance = [val/iters for val in avg_explained_variance]
	print('RCA avg explained variance : ' + str(avg_explained_variance))

	x_rca = GaussianRandomProjection(n_components=8,random_state=0).fit_transform(x_data)
	variance = [np.var(x_rca[:,i].reshape(-1, 1)) for i in range(8)]
	print('RCA var for experiments : ')
	print(variance)
	x_rca = np.vstack((x_rca[:,2], x_rca[:,3], x_rca[:,4], x_rca[:,6])).transpose()
	print()

	return x_rca

def principal_component(x_data):
	pca = PCA(n_components=8).fit(x_data)
	print('PCA explained_variance : ' + str(pca.explained_variance_))
	print('PCA explained_variance ratio: ' + str(pca.explained_variance_ratio_))
	print()
	x_pca = pca.fit_transform(x_data)
	x_pca = x_pca[:,0:3]

	return x_pca

def indpendent_component(x_data):
	ica = FastICA(n_components=8,whiten=True,random_state=0)
	x_ica = ica.fit_transform(x_data)
	kurt = [kurtosis(x_ica[:,i]) for i in range(len(x_ica[0]))]
	print('ICA kurt : ')
	print(kurt)
	ind = []
	for i in range(3):
		new_ind = np.argmax(kurt)
		ind.append(new_ind)
		kurt[new_ind] = 0
	x_ica = np.vstack((x_ica[:,ind[0]], x_ica[:, ind[1]], x_ica[:, ind[2]])).transpose()
	print()

	return x_ica


def main():
	x_train, y_train, x_test, y_test, x_data, y_data, column_scales, column_min = read_data()

	# #VARIANCE THRESHOLD
	# variance_threshold(x_data)

	# #RCA
	# randomized_projection(x_data)

	# #PCA
	# principal_component(x_data)

	# #ICA
	# indpendent_component(x_data)


if __name__ == '__main__':
	main()
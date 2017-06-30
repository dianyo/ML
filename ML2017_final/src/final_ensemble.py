import numpy as np
import csv
import sys
from sklearn.feature_extraction import DictVectorizer
from collections import OrderedDict
from sklearn.feature_selection import VarianceThreshold, SelectKBest
from sklearn.feature_selection import mutual_info_classif, chi2, mutual_info_regression
from sklearn import preprocessing
from sklearn.svm import SVC 
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn import datasets
from sklearn.decomposition import PCA
import xgboost as xg
from sklearn.grid_search import GridSearchCV
train_X_path = sys.argv[1]
train_y_path = sys.argv[2]
test_X_path = sys.argv[3]
sample_path = sys.argv[4]
answer_path = sys.argv[5]
DV = DictVectorizer(sparse=False)
le = preprocessing.LabelEncoder()
val=1000
def is_num(s):
	try:
		return float(s)
	except ValueError:
		return s

def read_X_data(path, data_type):
	data = []
	with open(path) as f:
		reader = csv.DictReader(f)
		k = 0
		for row in reader:
			row_data = {}
			for name in reader.fieldnames:
				if name == 'date_recorded':
					row[name] = row[name][5:7]
				if name == 'permit':
					row[name] = 1 if row[name] == True else 0
				if name not in [
					'id', 'recorded_by', 
					'wpt_name', 'subvillage',
					'ward', 'scheme_name', 'installer', 'funder',
					'waterpoint_type_group'] :
					row_data[name] = is_num(row[name])
			data.append(row_data)
	if data_type == 'train':
		X = DV.fit_transform(data)
	else:
		X = DV.transform(data)
	return X
def read_y_data(path):
	data = []
	with open(path) as f:
		reader = csv.reader(f)
		data = np.array(list(reader))
	data = np.delete(data, 0, axis=0)
	data = np.delete(data, 0, axis=1).ravel()
	
	le.fit(data)
	y = le.fit_transform(data)
	return y
			
def feature_select(X, y):
	# dimension reduction
	print("orgin shape = " + str(X.shape))
	# remove low variance
	# selector = VarianceThreshold(threshold=(.95 * (1 - .95)))
	# low_variance_X = selector.fit_transform(X)
	# print("low variance shape = " + str(low_variance_X.shape))

	# k best
	# selector = SelectKBest(chi2, k=200)
	# k_best_X = selector.fit_transform(X, y)
	

	# PCA
	print('doing PCA...')
	np.random.seed = 10
	pca = PCA(n_components=20)
	new_X = pca.fit_transform(X)
	print('new shape = ' + str(new_X.shape))
	return new_X

def write_answer(y, path):
	with open(path) as f:
		reader = csv.reader(f)
		answer = list(reader)
	answer = np.array(answer)
	with open(answer_path, 'w+') as f:
	 	for i in range(len(answer)):
	 		if i == 0:
	 			f.write('id,status_group\n')
	 		else:
	 			f.write('{},{}\n'.format(answer[i][0], y[i - 1]))


# read training data
print('reading train data ...')
train_X = read_X_data(train_X_path, 'train')
origin_train_y = read_y_data(train_y_path)


scaler = preprocessing.StandardScaler().fit(train_X)
X_transformed = scaler.transform(train_X)

train_X = X_transformed[:45000]
train_y = origin_train_y[:45000]

validation_X = X_transformed[45000:]
validation_y = origin_train_y[45000:]

print('reading validation predict data1')
validation_predict = np.load('predict1.npy')
for i in range(2,8):
	print('reading validation predict data' + str(i))
	tmp_validation_predict = np.load('predict' + str(i) + '.npy')
	validation_predict = np.vstack((validation_predict, tmp_validation_predict))
validation_predict = np.transpose(validation_predict)
print(validation_predict.shape)
print(train_X.shape)
# # start training
# clf = SVC(C=1, gamma='auto', )
print('training...')
xgb1 = xg.XGBClassifier(random_state=12345) #gamma=0.1, min_child_weight=0.5, 
# clf.fit(validation_predict, validation_y)


#0.8197: n_estimators=100, max_depth=14, learning_rate=0.2, colsample_bytree=0.4
#0.8173: n_estimators=500,
#0.8201:n_estimators=300, max_depth=15
#0.8188:n_estimators=256, max_depth=15, colsample_bytree=0.5
#0.8190:n_estimator=312, max_depth=16, colsample_bytree=0.4, min_child_weight=0.5
#:n_estimator=312, max_depth=16, colsapmle_bytree=0.4, min_child_weight=1.5
#0.8236:n_estimators=312, learning_rate=0.1, max_depth=14

# validation
# print('Validating...')
# print("Ein using RFC: " + str(clf.score(X_transformed, train_y)))
# scores = cross_val_score(clf, X_transformed, train_y, cv=5)
# print('Eout using RFC: ' + str(scores.mean()))

#grid
print('CV grid...')
param = {
    'n_estimators':[10, 100, 250],
    'max_depth':[5,7,9]
}

CV_grid = GridSearchCV(estimator=xgb1, param_grid=param, verbose=1,
	n_jobs=-1, cv=5)
CV_grid.fit(validation_predict, validation_y)
clf = CV_grid.best_estimator_

print('best params: ' + str(CV_grid.best_params_))

# read testing data
# print('reading test data ...')
# test_X = read_X_data(test_X_path, 'test')
# test_X_transformed = scaler.transform(test_X)

print('reading test X predict')
test_predict_X = np.load('X1.npy')
for i in range(2,8):
	print('reading test X predict data' + str(i))
	tmp_test_predict_X = np.load('X' + str(i) + '.npy')
	test_predict_X = np.vstack((test_predict_X, tmp_test_predict_X))
test_predict_X = np.transpose(test_predict_X)

print('predicting ... ')
predict_y = clf.predict(test_predict_X)

# vote:0.8214, with answer4 = n_estimator=312, max_depth=16, colsample_bytree=0.4, min_child_weight=1.5,
# and model n_estimator=128, max_depth=14, no min_child_weight
"""test_1 = read_y_data("answer_8201.csv")
test_2 = read_y_data("answer_8188.csv")
test_3 = read_y_data("answer_8190.csv")
test_4 = read_y_data("answer4.csv")

predicts = np.vstack((test_1, test_2, test_3, test_4, predict_y))
print(predicts.shape)
ans = []
for i in range(predicts.shape[1]):
    count = np.bincount(predicts[:, i])
    ans.append(np.argmax(count))"""

decoded_y = le.inverse_transform(predict_y)
write_answer(decoded_y, sample_path)

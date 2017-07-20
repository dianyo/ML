import numpy as np
import csv
import sys
import sklearn
from sklearn.feature_extraction import DictVectorizer
from collections import OrderedDict
from sklearn.feature_selection import VarianceThreshold, SelectKBest
from sklearn.feature_selection import mutual_info_classif, chi2, mutual_info_regression
from sklearn import preprocessing
from sklearn.svm import SVC 
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
from xgboost.sklearn import XGBClassifier  
train_X_path = sys.argv[1]
train_y_path = sys.argv[2]
test_X_path = sys.argv[3]
sample_path = sys.argv[4]
DV = DictVectorizer(sparse=False)
le = preprocessing.LabelEncoder()
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
	selector = SelectKBest(chi2, k=200)
	k_best_X = selector.fit_transform(X, y)
	# 

def write_answer(y, path):
	with open(path) as f:
		reader = csv.reader(f)
		answer = list(reader)
	answer = np.array(answer)
	with open('answer.csv', 'w+') as f:
	 	for i in range(len(answer)):
	 		if i == 0:
	 			f.write('id,status_group\n')
	 		else:
	 			f.write('{},{}\n'.format(answer[i][0], y[i - 1]))


# read training data
print('reading train data ...')
train_X = read_X_data(train_X_path, 'train')
train_y = read_y_data(train_y_path)

# selected_train_X = feature_select(train_X)
scaler = preprocessing.StandardScaler().fit(train_X)
X_transformed = scaler.transform(train_X)

# # start training
# clf = SVC(C=1, gamma='auto', )
print('training...')
clf = XGBClassifier(learning_rate =0.1, n_estimators=1000, max_depth=11,
	gamma=1, min_child_weight = 1, n_job=-1)
clf.fit(X_transformed, train_y)
# rfc_model = RandomForestClassifier(n_estimators=200,  max_depth=24,
# 	random_state=12345, verbose=1)

# param = {
# 	 'max_depth':range(3,10,2),
#  'min_child_weight':range(1,6,2)
# }
# CV_grid = GridSearchCV(estimator=rfc_model, param_grid=param, verbose=1)
# CV_grid.fit(X_transformed, train_y)
# clf = CV_grid.best_estimator_

# validation
# print('Validating...')
# print("Ein using RFC: " + str(clf.score(X_transformed, train_y)))
# scores = cross_val_score(clf, X_transformed, train_y, cv=5)
# print('Eout using RFC: ' + str(scores.mean()))


# read testing data
print('reading test data ...')
test_X = read_X_data(test_X_path, 'test')
test_X_transformed = scaler.transform(test_X)

print('predicting ... ')
predict_y = clf.predict(test_X_transformed)
decoded_y = le.inverse_transform(predict_y)
write_answer(decoded_y, sample_path)
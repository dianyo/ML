import numpy as np
import sys
import csv
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
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
def write_answer(y, path):
	with open(path) as f:
		reader = csv.reader(f)
		answer = list(reader)
	answer = np.array(answer)
	with open('answer_vote.csv', 'w+') as f:
	 	for i in range(len(answer)):
	 		if i == 0:
	 			f.write('id,status_group\n')
	 		else:
	 			f.write('{},{}\n'.format(answer[i][0], y[i - 1]))


predicts = np.zeros((14850,), dtype=int)
for i in range(len(sys.argv) - 1):
	print(sys.argv[i + 1])
	test = read_y_data(sys.argv[i + 1])
	predicts = np.vstack((predicts, test))

predicts = np.delete(predicts, 0, 0)

ans = []
for i in range(predicts.shape[1]):
    count = np.bincount(predicts[:, i])
    ans.append(np.argmax(count))

decoded_y = le.inverse_transform(ans)
write_answer(decoded_y, 'data/Submission_format.csv')
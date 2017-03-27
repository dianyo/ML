import csv
import numpy as np
import sys
test_file = sys.argv[1]
ouput_file = sys.argv[2]
def func_star(b, w, data):
	return b + w.dot(data)

def get_param():
	f = open("param.txt", "r+")
	b = float(f.readline())
	tmp_string = ''
	for line in f:
		tmp_string = tmp_string + line
	tmp_string = tmp_string.replace('[', '')
	tmp_string = tmp_string.replace(']', '')
	w = [(i) for i in tmp_string.split()]
	w = np.array(w, dtype=float)
	f.close()
	return (b,w)

def write_submission(data):
	global ouput_file
	with open(ouput_file, 'w', newline='\n') as f:
		writer = csv.writer(f)
		writer.writerow(['id','value'])
		for i in range(240):
			writer.writerow(['id_'+str(i), data[i]])

f = open(test_file, encoding='big5')
reader = csv.reader(f)
test_data = list(reader)
f.close
for i in range(len(test_data)):
	tmp = 2
	while tmp:
		test_data[i].pop(0)
		tmp = tmp - 1
	if (i - 10) % 18 == 0:
		for j in range(0, len(test_data[i])):
			if test_data[i][j] == 'NR':
				test_data[i][j] = '0'
test_data = np.array(test_data, dtype=float)
b,w = get_param()
garbage_feature = []
with open("delete_weights", "r+") as f:
	for line in f:
		garbage_feature.append(int(line))
PM25 = []
# with open("garbage_feature", "r+") as f:
# 	garbage_feature = f.readline()
# garbage_feature = garbage_feature.replace('[', '')
# garbage_feature = garbage_feature.replace(']', '')
# garbage_feature = garbage_feature.replace(',', '')
# garbage_feature = [int((i)) for i in garbage_feature.split(' ')]
for i in range(0, 4320, 18):
	test = test_data[i:i+18, 0:9].ravel()
	test = np.delete(test, garbage_feature, 0)
	# test = np.concatenate((test,test*test), axis=0)
	PM25.append(func_star(b, w, test))
write_submission(PM25)


	
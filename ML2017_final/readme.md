**Dependencies**
1. xgboost : http://xgboost.readthedocs.io/ 
2. scikit-learn: http://scikit-learn.org/stable/

</br>
</br>

**Get Answer**
voting by `python3 src/vote.py candidate/*.csv` and will get the answer file `answer_vote.csv`

</br>
</br>

**Train Model**
traing a xgboost model by `python3 src/Final_xboost_pass_strong.py data/Training_set_values.csv data/Training_set_labels.csv data/Test_set_labels.csv data/Submission_format.csv answer.csv` where `answer.csv` can be replace by any file name you want.
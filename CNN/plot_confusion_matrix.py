from keras.models import load_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import sys
import itertools
import numpy as np
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

model_path = sys.argv[1]
emotion_classifier = load_model(model_path)
np.set_printoptions(precision=2)
val_x = np.load('validation2000_x.npy')
val_y = np.load('validation2000_y.npy')
val_x = val_x / 255
predict_y = emotion_classifier.predict_classes(val_x)
conf_mat = confusion_matrix(val_y, predict_y)

plt.figure()
plot_confusion_matrix(conf_mat, classes=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"])
plt.savefig(sys.argv[2])
plt.close()
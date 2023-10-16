from matplotlib import pyplot as plt
import matplotlib
from languages_list import Languages   
import numpy as np 


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    log_cm = np.log(1 + cm)
    
    plt.figure(figsize=(30, 30), dpi=200)
    im = plt.imshow(cm, interpolation='nearest', cmap=cmap, norm=matplotlib.colors.LogNorm())
    plt.title(title)
    cbar = plt.colorbar(im,fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=24)

    if classes is not None:
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90, fontsize=12)
        plt.yticks(tick_marks, classes, fontsize=12)
        plt.tick_params(labeltop=True, labelright=True)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = log_cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] == 0:
                continue


            fontsize = 10
            text = str(cm[i, j])

            if cm[i, j] >= 100:
                fontsize = 8
            if cm[i, j] >= 1000:
                text = f"{cm[i, j]/1000:.1f}K"
                fontsize = 7

            plt.text(j, i+0.2, text, fontsize=fontsize,
                    horizontalalignment="center",
                    color="white" if log_cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cm = np.load("../artifacts/confusion_matrix.npy")

classes = np.array([
    Languages.to_string(i)
    for i in range(len(Languages))
])

order = np.argsort(-cm.sum(axis=1))

plot_confusion_matrix(cm[order, :][:, order], classes[order])
plt.savefig("../images/confusion_matrix.png")

for language in Languages:
    i = language.value
    print(Languages.to_string(language), cm[i, i] / cm[i].sum())

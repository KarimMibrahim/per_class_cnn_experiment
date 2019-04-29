import numpy as np
import pandas as pd
import os

PATH_TO_RESULTS = '/cluster/storage/kibrahim/per_class_classifier/experiments/'
IDS_PATH = '/cluster/storage/kibrahim/per_class_classifier/GroundTruth/'
SPECTROGRAM_PATH = "/cluster/storage/kibrahim/mel_specs/"
directories = ['car_Per_class_/2019-04-15_13-30-13','chill_Per_class_/2019-04-15_14-25-36',
               'club_Per_class_/2019-04-15_15-44-42','dance_Per_class_/2019-04-15_16-27-11',
              'gym_Per_class_/2019-04-15_17-57-34','happy_Per_class_/2019-04-15_18-34-12',
               'morning_Per_class_/2019-04-15_18-51-05',
              'night_Per_class_/2019-04-15_19-03-58','park_Per_class_/2019-04-15_19-21-25',
               'party_Per_class_/2019-04-15_19-47-24','relax_Per_class_/2019-04-15_21-09-31',
              'running_Per_class_/2019-04-15_21-55-19','sad_Per_class_/2019-04-15_22-25-30',
              'shower_Per_class_/2019-04-15_22-47-12 ','sleep_Per_class_/2019-04-15_22-58-28',
              'summer_Per_class_/2019-04-15_23-22-40','train_Per_class_/2019-04-15_23-58-12',
              'training_Per_class_/2019-04-16_00-10-45','work_Per_class_/2019-04-16_00-38-513',
              'workout_Per_class_/2019-04-16_01-18-23']
labels = ['car','chill','club','dance','gym','happy','morning','night','park','party','relax','running',
         'sad','shower','sleep','summer','train','training','work','workout']



# Plotting confusion matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax,fig


def get_predictions_classes(path):
    predictions_probs = np.loadtxt(os.path.join(path , 'predictions.out'))
    test_classes = np.loadtxt(os.path.join(path , 'test_ground_truth_classes.txt'))
    predictions = np.round(predictions_probs)
    return predictions,test_classes

# extracting TP,TN,FP,FN
def get_TP_TN_FP_FN(songs_ID,predictions,test_classes):
    true_positives = [x for x in range(len(songs_ID)) if predictions[x] == test_classes[x] and predictions[x] == 1]
    true_negatives = [x for x in range(len(songs_ID)) if predictions[x] == test_classes[x] and predictions[x] == 0]
    false_positives = [x for x in range(len(songs_ID)) if predictions[x] != test_classes[x] and predictions[x] == 1]
    false_negatives = [x for x in range(len(songs_ID)) if predictions[x] != test_classes[x] and predictions[x] == 0]
    return true_positives,true_negatives,false_positives,false_negatives

# get song_ids
def get_song_ids(label):
    test_ground_truth = pd.read_csv(IDS_PATH+label+'_test_groundtruth.csv')
    songs_ID = np.zeros([len(test_ground_truth), 1])
    for idx, filename in enumerate(list(test_ground_truth.song_id)):
        songs_ID[idx] = filename
    songs_ID = songs_ID.astype(int)
    return songs_ID


# reading predictions and groundtruth
for path,label in zip(directories,labels):
    predictions,test_classes = get_predictions_classes(os.path.join(PATH_TO_RESULTS,path))
    songs_ID = get_song_ids(label)
    true_positives,true_negatives,false_positives,false_negatives = get_TP_TN_FP_FN(songs_ID,
                                                                                    predictions,test_classes)
    # saving tracks for TP,TN..
    with open(os.path.join(PATH_TO_RESULTS,path,label+"_true_positives.txt"),'w')as f:
        for x in true_positives:
            f.writelines("https://www.deezer.com/en/track/"+str(int(songs_ID[x])) + "\n")

    with open(os.path.join(PATH_TO_RESULTS,path,label+"_true_negatives.txt"),'w')as f:
        for x in true_negatives:
            f.writelines("https://www.deezer.com/en/track/"+str(int(songs_ID[x])) + "\n")

    with open(os.path.join(PATH_TO_RESULTS,path,label+"_false_positives.txt"),'w')as f:
        for x in false_positives:
            f.writelines("https://www.deezer.com/en/track/"+str(int(songs_ID[x])) + "\n")

    with open(os.path.join(PATH_TO_RESULTS,path,label+"_false_negatives.txt"),'w')as f:
        for x in false_negatives:
            f.writelines("https://www.deezer.com/en/track/"+str(int(songs_ID[x])) + "\n")
            
    with open(os.path.join(PATH_TO_RESULTS,path,label+"_sample_tracks.txt"),'w')as f:
        for x in false_negatives:
            f.writelines("https://www.deezer.com/en/track/"+str(int(songs_ID[x])) + "\n")

    with open(os.path.join(PATH_TO_RESULTS,path,"dataset_split_size.txt"),'w') as f:
        f.writelines("Train datset size: " + str(int(np.ceil(len(songs_ID) * 4.0 * 0.9))) + "\n")
        f.writelines("validation datset size: " + str(int(np.floor(len(songs_ID) * 4.0 * 0.1))) + "\n")
        f.writelines("Test datset size: " + str(len(songs_ID)) + "\n")  
        
    ax,fig =plot_confusion_matrix(test_classes,predictions,["Negative","Positive"],title="Confusion Matrix for Class : " + label)
    fig.savefig(os.path.join(PATH_TO_RESULTS,path,label + "_confusion_matrix.png"))
    fig.savefig(os.path.join(PATH_TO_RESULTS,path,label + "_confusion_matrix.pdf"), format='pdf')


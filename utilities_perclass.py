# General Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':22})

# Machine Learning preprocessing and evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, hamming_loss, f1_score,confusion_matrix

SPECTROGRAMS_PATH = "/cluster/storage/kibrahim/mel_specs/"

    
def load_train_set_raw(TRAIN_PATH,FRAMES_NUMBER,SEGMENT_START,SPECTROGRAM_PATH=SPECTROGRAMS_PATH):
    # Loading testset groundtruth
    train_ground_truth = pd.read_csv(TRAIN_PATH)
    train_classes = train_ground_truth.binary_label.values
    train_classes = train_classes.astype(int)
    spectrograms = np.zeros([len(train_ground_truth), FRAMES_NUMBER, 96])
    songs_ID = np.zeros([len(train_ground_truth), 1])
    counter = 0
    for idx, filename in enumerate(list(train_ground_truth.song_id)):
        try:
            spect = np.load(os.path.join(SPECTROGRAM_PATH, str(filename) + '.npz'))['feat']
        except:
            counter +=1
            continue
        if (spect.shape == (1, 1292, 96)):
            spectrograms[idx] = spect[:,SEGMENT_START:SEGMENT_START + FRAMES_NUMBER,:]
            songs_ID[idx] = filename
    print("Failed to load " + str(counter) + "tracks")
    non_empty_rows = ~np.all(spectrograms == 0, axis=(1,2))
    spectrograms = spectrograms[non_empty_rows]
    train_classes = train_classes[non_empty_rows]
    spectrograms = np.expand_dims(spectrograms, axis=3)
    return spectrograms, train_classes
    
    
def load_test_set_raw(TEST_PATH,FRAMES_NUMBER,SEGMENT_START,SPECTROGRAM_PATH=SPECTROGRAMS_PATH):
    # Loading testset groundtruth
    test_ground_truth = pd.read_csv(TEST_PATH)
    test_classes = test_ground_truth.binary_label.values
    test_classes = test_classes.astype(int)
    spectrograms = np.zeros([len(test_ground_truth), FRAMES_NUMBER, 96])
    songs_ID = np.zeros([len(test_ground_truth), 1])
    for idx, filename in enumerate(list(test_ground_truth.song_id)):
        try:
            spect = np.load(os.path.join(SPECTROGRAM_PATH, str(filename) + '.npz'))['feat']
        except:
            continue
        if (spect.shape == (1, 1292, 96)):
            spectrograms[idx] = spect[:,SEGMENT_START:SEGMENT_START + FRAMES_NUMBER,:]
            songs_ID[idx] = filename
    non_empty_rows = ~np.all(spectrograms == 0, axis=(1,2))
    spectrograms = spectrograms[non_empty_rows]
    test_classes = test_classes[non_empty_rows]
    songs_ID = songs_ID[non_empty_rows]
    spectrograms = np.expand_dims(spectrograms, axis=3)
    return spectrograms, test_classes,songs_ID
    
    
def evaluate_model(test_pred_prob,test_pred, spectrograms, test_classes, saving_path):
    """
    Evaluates a given model using accuracy, area under curve and hamming loss
    :param model: model to be evaluated
    :param spectrograms: the test set spectrograms as an np.array
    :param test_classes: the ground truth labels
    :return: accuracy, auc_roc
    """
    # Accuracy
    accuracy = 100 * accuracy_score(test_classes, test_pred)
    print("Exact match accuracy is: " + str(accuracy) + "%")
    # Area Under the Receiver Operating Characteristic Curve (ROC AUC)
    auc_roc = roc_auc_score(test_classes, test_pred_prob)
    print("Area Under the Curve (AUC) is: " + str(auc_roc))
    recall = recall_score(test_classes, test_pred)
    print("recall is: " + str(recall))
    precision = precision_score(test_classes, test_pred)
    print("precision is: " + str(precision))
    f1 = f1_score(test_classes, test_pred)
    print("f1 is: " + str(f1)) 
    with open(os.path.join(saving_path, "evaluation_results.txt"), "w") as f:
        f.write("Exact match accuracy is: " + str(accuracy) + "%\n" + "Area Under the Curve (AUC) is: " + str(auc_roc)
         + "\n" + "recall is: " + str(recall) + "\n" + "precision is: " + str(precision) + "\n" + "f1 is: " + str(f1))
    print("saving prediction to disk")
    np.savetxt(os.path.join(saving_path, 'predictions.out'), test_pred_prob, delimiter=',')
    np.savetxt(os.path.join(saving_path, 'test_ground_truth_classes.txt'), test_classes, delimiter=',')
    return accuracy, auc_roc, recall, precision, f1   
    
    
def plot_loss_acuracy(history, path, label):
    # Plot training & validation accuracy values
    plt.figure(figsize=(10, 10))
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy for ' + label + " class")
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(os.path.join(path,label + "_model_accuracy.png"))
    plt.savefig(os.path.join(path,label  + "_model_accuracy.pdf"), format='pdf')
    #plt.savefig(os.path.join(path,label + "_model_accuracy.eps"), format='eps', dpi=900)
    #Plot training & validation loss values
    plt.figure(figsize=(10, 10))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss ' + label + " class")
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(os.path.join(path,label + "_model_loss.png"))
    plt.savefig(os.path.join(path,label + "_model_loss.pdf"), format='pdf')
    #plt.savefig(os.path.join(path,label + "_model_loss.eps"), format='eps', dpi=900)
    
def plot_confusion_matrix(y_true, y_pred, classes, path, label,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    title = "Confusion Matrix for Class : " + label

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
    fig.savefig(os.path.join(path,label + "_confusion_matrix.png"))
    fig.savefig(os.path.join(path,label + "_confusion_matrix.pdf"), format='pdf') 
    
    
# extracting TP,TN,FP,FN
def get_TP_TN_FP_FN(songs_ID,predictions,test_classes,path,label):
    true_positives = [x for x in range(len(songs_ID)) if predictions[x] == test_classes[x] and predictions[x] == 1]
    true_negatives = [x for x in range(len(songs_ID)) if predictions[x] == test_classes[x] and predictions[x] == 0]
    false_positives = [x for x in range(len(songs_ID)) if predictions[x] != test_classes[x] and predictions[x] == 1]
    false_negatives = [x for x in range(len(songs_ID)) if predictions[x] != test_classes[x] and predictions[x] == 0]
    # saving tracks for TP,TN..
    with open(os.path.join(path,label+"_true_positives.txt"),'w')as f:
        for x in true_positives:
            f.writelines("https://www.deezer.com/en/track/"+str(int(songs_ID[x])) + "\n")
    with open(os.path.join(path,label+"_true_negatives.txt"),'w')as f:
        for x in true_negatives:
            f.writelines("https://www.deezer.com/en/track/"+str(int(songs_ID[x])) + "\n")
    with open(os.path.join(path,label+"_false_positives.txt"),'w')as f:
        for x in false_positives:
            f.writelines("https://www.deezer.com/en/track/"+str(int(songs_ID[x])) + "\n")
    with open(os.path.join(path,label+"_false_negatives.txt"),'w')as f:
        for x in false_negatives:
            f.writelines("https://www.deezer.com/en/track/"+str(int(songs_ID[x])) + "\n")
    return true_positives,true_negatives,false_positives,false_negatives
    


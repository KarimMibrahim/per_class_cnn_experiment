# General Imports
import os
import numpy as np
import pandas as pd
from time import strftime, localtime
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':22})

# Deep Learning
import tensorflow as tf
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, TimeDistributed, Flatten, GRU, Dropout, Dense
from keras.models import Sequential
from keras.layers import InputLayer, Conv2D, MaxPooling2D, TimeDistributed, Flatten, GRU, Dropout, Dense,BatchNormalization
import dzr_ml_tf.data_pipeline as dp
from dzr_ml_tf.label_processing import tf_multilabel_binarize
#from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from keras.callbacks import ModelCheckpoint, TensorBoard


# Machine Learning preprocessing and evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, roc_auc_score, hamming_loss, f1_score
from sklearn.model_selection import train_test_split

# Reproducible results
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

SOURCE_PATH = "/cluster/storage/kibrahim/per_class_classifier/"
SPECTROGRAMS_PATH = "/cluster/storage/kibrahim/mel_specs/"

INPUT_SHAPE = (1292, 96, 1)
LABELS_LIST = ['car', 'chill', 'club', 'dance', 'gym', 'happy', 'morning', 'night', 'park', 'party', 'relax', 'running',
               'sad',
               'shower', 'sleep', 'summer', 'train', 'training', 'work', 'workout']

def get_model():
    # Define model architecture
    model = Sequential(
        [
            InputLayer(input_shape=INPUT_SHAPE, name="input_layer"),

            BatchNormalization(),

            Conv2D(activation="relu", filters=32, kernel_size=[3, 3], name="conv_1", padding="same"),
            MaxPooling2D(name="max_pool_1", padding="valid", pool_size=[2, 2]),

            Conv2D(activation="relu", filters=64, kernel_size=[3, 3], name="conv_2", padding="same", use_bias=True),
            MaxPooling2D(name="max_pool_2", padding="valid", pool_size=[2, 2]),

            Conv2D(activation="relu", filters=128, kernel_size=[3, 3], name="conv_3", padding="same", use_bias=True),
            MaxPooling2D(name="max_pool_3", padding="valid", pool_size=[2, 2]),

            Conv2D(activation="relu", filters=256, kernel_size=[3, 3], name="conv_4", padding="same", use_bias=True),
            MaxPooling2D(name="max_pool_4", padding="valid", pool_size=[2, 2]),

            # TimeDistributed(layer=Flatten(name="Flatten"), name="TD_Flatten"),
            # GRU(activation="tanh", dropout=0.1, name="gru_1", recurrent_activation="hard_sigmoid", recurrent_dropout=0.1,
            #        return_sequences=False, trainable=True, units=512, use_bias=True),

            # Dropout(name="dropout_1", rate=0.3),
            # Dense(activation="sigmoid", name="dense_1", trainable=True, units=20),

            Flatten(),
            Dense(256, activation='sigmoid', name="dense_1"),
            Dropout(name="dropout_1", rate=0.3),
            Dense(1, activation='sigmoid', name="dense_2"),
        ]
    )
    return model


def compile_model(model, loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy']):
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    
def load_train_set_raw(TRAIN_PATH,SPECTROGRAM_PATH=SPECTROGRAMS_PATH):
    # Loading testset groundtruth
    train_ground_truth = pd.read_csv(TRAIN_PATH)
    train_classes = train_ground_truth.binary_label.values
    train_classes = train_classes.astype(int)
    spectrograms = np.zeros([len(train_ground_truth), 1292, 96])
    songs_ID = np.zeros([len(train_ground_truth), 1])
    for idx, filename in enumerate(list(train_ground_truth.song_id)):
        try:
            spect = np.load(os.path.join(SPECTROGRAM_PATH, str(filename) + '.npz'))['feat']
        except:
            continue
        if (spect.shape == (1, 1292, 96)):
            spectrograms[idx] = spect
            songs_ID[idx] = filename
    spectrograms = np.expand_dims(spectrograms, axis=3)
    return spectrograms, train_classes


def load_test_set_raw(TEST_PATH,SPECTROGRAM_PATH=SPECTROGRAMS_PATH):
    # Loading testset groundtruth
    test_ground_truth = pd.read_csv(TEST_PATH)
    test_classes = test_ground_truth.binary_label.values
    test_classes = test_classes.astype(int)
    spectrograms = np.zeros([len(test_ground_truth), 1292, 96])
    songs_ID = np.zeros([len(test_ground_truth), 1])
    for idx, filename in enumerate(list(test_ground_truth.song_id)):
        try:
            spect = np.load(os.path.join(SPECTROGRAM_PATH, str(filename) + '.npz'))['feat']
        except:
            continue
        if (spect.shape == (1, 1292, 96)):
            spectrograms[idx] = spect
            songs_ID[idx] = filename
    spectrograms = np.expand_dims(spectrograms, axis=3)
    return spectrograms, test_classes


def evaluate_model(model, spectrograms, test_classes, saving_path):
    """
    Evaluates a given model using accuracy, area under curve and hamming loss
    :param model: model to be evaluated
    :param spectrograms: the test set spectrograms as an np.array
    :param test_classes: the ground truth labels
    :return: accuracy, auc_roc
    """
    test_pred_prob = model.predict(spectrograms)
    test_pred = np.round(test_pred_prob)
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



def save_model(model, path):
    # serialize model to JSON
    model_json = model.to_json()
    with open(os.path.join(path, "model.json"), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(os.join.path(path, "model.h5"))
    print("Saved model to disk")


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
    plt.title('Model loss ' + label + "class")
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(os.path.join(path,label + "_model_loss.png"))
    plt.savefig(os.path.join(path,label + "_model_loss.pdf"), format='pdf')
    #plt.savefig(os.path.join(path,label + "_model_loss.eps"), format='eps', dpi=900)


def main():
    # splitting datasets
    # split_dataset()
    labels_results = np.zeros([20,6])
    # Loading datasets
    for idx,label in enumerate(LABELS_LIST):
        print("training for class : " + label)
        training_dataset,training_classes = load_train_set_raw(os.path.join(SOURCE_PATH, "GroundTruth/", 
                                                                label+"_train_groundtruth.csv"))
        exp_dir = os.path.join(SOURCE_PATH, "experiments/")
        experiment_name = os.path.join(label + "_Per_class_", strftime("%Y-%m-%d_%H-%M-%S", localtime()))
        fit_config = {
            #"steps_per_epoch": 1000,
            "batch_size":64,
            "epochs": 20,
            "initial_epoch": 0,
            #"validation_steps": 100,
            "validation_split":0.1,
            "callbacks": [
                TensorBoard(log_dir=os.path.join(exp_dir, experiment_name)),
                ModelCheckpoint(os.path.join(exp_dir, experiment_name, "last_iter.h5"),
                                save_weights_only=False),
                ModelCheckpoint(os.path.join(exp_dir, experiment_name, "best_eval.h5"),
                                save_best_only=True,
                                monitor="val_loss",
                                save_weights_only=False)
            ]
        }
        # Printing the command to run tensorboard [Just to remember]
        print("Execute the following in a terminal:\n" + "tensorboard --logdir=" + os.path.join(exp_dir, experiment_name))
        model = get_model()
        compile_model(model)
        history = model.fit(training_dataset,training_classes, **fit_config);
        spectrograms, test_classes = load_test_set_raw(os.path.join(SOURCE_PATH, "GroundTruth/", 
                                                                label+"_test_groundtruth.csv"))
        accuracy, auc_roc, recall, precision, f1 = evaluate_model(model, spectrograms, test_classes,
                                                      saving_path=os.path.join(exp_dir, experiment_name))
        labels_results[idx,:] = [len(training_classes),accuracy, auc_roc, recall, precision, f1]
        plot_loss_acuracy(history,os.path.join(exp_dir, experiment_name),label)
        
    labels_results_df = pd.DataFrame(labels_results,index = LABELS_LIST , columns  = [ "Training Size","Accuracy", "AUC_ROC", "Recall", "Precision", "f1"])
    labels_results_df.to_csv(os.path.join(SOURCE_PATH, 'all_results.csv'))
     
     
if __name__ == "__main__":
    main()

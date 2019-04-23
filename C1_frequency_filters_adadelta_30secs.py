# General Imports
import os
import numpy as np
import pandas as pd
from time import strftime, localtime
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':22})

# my code
from utilities_perclass import load_train_set_raw,load_test_set_raw,evaluate_model,plot_loss_acuracy,plot_confusion_matrix,get_TP_TN_FP_FN

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
from keras import optimizers


# Machine Learning preprocessing and evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, roc_auc_score, hamming_loss, f1_score
from sklearn.model_selection import train_test_split

# Reproducible results
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

SOURCE_PATH = "/cluster/storage/kibrahim/per_class_cnn_experiment/"
SPECTROGRAMS_PATH = "/cluster/storage/kibrahim/mel_specs/"

FRAMES_NUMBER = 646
SEGMENT_START = 323 # starting frame for segmentation, i.e. for a 30 seconds segment, start at frame 323 = ~15 seconds
RESULTS_SAVING_Path = "C2_frequency_30secs_Results"
optimization = optimizers.Adadelta()


INPUT_SHAPE = (FRAMES_NUMBER, 96, 1)
#LABELS_LIST = ['car', 'chill', 'club', 'dance', 'gym', 'happy', 'morning', 'night', 'park', 'party', 'relax', 'running',
#               'sad','shower', 'sleep', 'summer', 'train', 'training', 'work', 'workout']
LABELS_LIST = ['car', 'chill', 'club', 'dance', 'night', 'relax','sad','sleep']


def get_model():
    # Define model architecture
    model = Sequential(
        [
            InputLayer(input_shape=INPUT_SHAPE, name="input_layer"),

            BatchNormalization(),

            Conv2D(activation="relu", filters=32, kernel_size=[32, 1], name="conv_1", padding="same"),
            MaxPooling2D(name="max_pool_1", padding="valid", pool_size=[1, 80]),

            #Conv2D(activation="relu", filters=64, kernel_size=[3, 3], name="conv_2", padding="same", use_bias=True),
            #MaxPooling2D(name="max_pool_2", padding="valid", pool_size=[2, 2]),


            # TimeDistributed(layer=Flatten(name="Flatten"), name="TD_Flatten"),
            # GRU(activation="tanh", dropout=0.1, name="gru_1", recurrent_activation="hard_sigmoid", recurrent_dropout=0.1,
            #        return_sequences=False, trainable=True, units=512, use_bias=True),

            # Dropout(name="dropout_1", rate=0.3),
            # Dense(activation="sigmoid", name="dense_1", trainable=True, units=20),

            Flatten(),
            Dense(200, activation='sigmoid', name="dense_1"),
            Dropout(name="dropout_1", rate=0.5),
            Dense(1, activation='sigmoid', name="dense_2"),
        ]
    )
    return model


def compile_model(model, loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy']):
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)


def main():
    # initialize results array 
    labels_results = np.zeros([len(LABELS_LIST),6])
    # Loading datasets
    for idx,label in enumerate(LABELS_LIST):
        print("training for class : " + label)
        training_dataset,training_classes = load_train_set_raw(os.path.join(SOURCE_PATH, "GroundTruth/", 
                                                                label+"_train_groundtruth.csv"),FRAMES_NUMBER,SEGMENT_START)
        X_train, X_val, y_train, y_val = train_test_split(training_dataset, training_classes, test_size=0.2, random_state=0, stratify =training_classes)
        # Defining saving paths
        exp_dir = os.path.join(SOURCE_PATH, "experiments/",RESULTS_SAVING_Path)
        experiment_name = os.path.join(label + "_Per_class_", strftime("%Y-%m-%d_%H-%M-%S", localtime()))
        Model_save_path = os.path.join(SOURCE_PATH, "Saved_models/", label + "_Per_class_", strftime("%Y-%m-%d_%H-%M-%S", localtime()))
        fit_config = {
            "batch_size":32,
            "epochs": 20,
            "initial_epoch": 0,
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
        compile_model(model,optimizer= optimization)
        model.summary()
        history = model.fit(X_train,y_train, validation_data=(X_val, y_val), **fit_config);
        spectrograms, test_classes,songs_ID = load_test_set_raw(os.path.join(SOURCE_PATH, "GroundTruth/", 
                                                                label+"_test_groundtruth.csv"),FRAMES_NUMBER,SEGMENT_START)
        test_pred_prob = model.predict(spectrograms)
        test_pred = np.round(test_pred_prob)  
        accuracy, auc_roc, recall, precision, f1 = evaluate_model(test_pred_prob,test_pred, spectrograms, test_classes,
                                                      saving_path=os.path.join(exp_dir, experiment_name))
        get_TP_TN_FP_FN(songs_ID,test_pred,test_classes,os.path.join(exp_dir, experiment_name),label) # saving samples of true negative, false negativest etc..
        plot_loss_acuracy(history,os.path.join(exp_dir, experiment_name),label)
        plot_confusion_matrix(test_classes,test_pred,["Negative","Positive"],os.path.join(exp_dir, experiment_name),label)
        labels_results[idx,:] = [len(training_classes),accuracy, auc_roc, recall, precision, f1]       
    labels_results_df = pd.DataFrame(labels_results,index = LABELS_LIST , columns  = [ "Training Size","Accuracy", "AUC_ROC", "Recall", "Precision", "f1"])
    labels_results_df.to_csv(os.path.join(exp_dir, RESULTS_SAVING_Path + '.csv'),float_format='%.3f')
     
     
if __name__ == "__main__":
    main()

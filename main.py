# General Imports
import os
import numpy as np
import pandas as pd
from time import strftime, localtime
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':22})

# my code
from utilities_perclass import load_train_set_raw,load_test_set_raw,evaluate_model,plot_loss_acuracy,plot_confusion_matrix,get_TP_TN_FP_FN
from models import get_model,compile_model
# Deep Learning
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
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

"""
for deezer cluster, uncomment
"""
#SOURCE_PATH = "/srv/workspace/research/git_repo"
#SPECTROGRAMS_PATH = "/srv/workspace/research/melspectrograms"

FRAMES_NUMBER = 646
SEGMENT_START = 323 # starting frame for segmentation, i.e. for a 30 seconds segment, start at frame 323 = ~15 seconds

INPUT_SHAPE = (FRAMES_NUMBER, 96, 1)
LABELS_LIST = ['car', 'chill', 'club', 'dance', 'gym', 'happy', 'morning', 'night' , 'party', 'relax', 'running',
               'sad', 'sleep', 'summer', 'training', 'work', 'workout']

models_list = ['C2_square', 'C1_time', 'C1_frequency', 'C4_square', 'C1_square']

def main():
    # initialize results array 
    #labels_results = np.zeros([len(LABELS_LIST)*len(models_list),8])
    labels_results= []
    # Loading datasets
    for idx,label in enumerate(LABELS_LIST):
        print("training for class : " + label)
        training_dataset,training_classes = load_train_set_raw(os.path.join(SOURCE_PATH, "GroundTruth/", 
                                                                label+"_train_groundtruth.csv"),FRAMES_NUMBER,SEGMENT_START)
        X_train, X_val, y_train, y_val = train_test_split(training_dataset, training_classes, test_size=0.2, random_state=0, stratify =training_classes)
        spectrograms, test_classes,songs_ID = load_test_set_raw(os.path.join(SOURCE_PATH, "GroundTruth/", 
                                                                label+"_test_groundtruth.csv"),FRAMES_NUMBER,SEGMENT_START)
        # iterate through models
        for model_name in models_list:
            # Defining saving paths
            exp_dir = os.path.join(SOURCE_PATH, "experiments/",label,model_name)
            if not os.path.exists(os.path.join(SOURCE_PATH, "experiments/",label)):
                os.mkdir(os.path.join(SOURCE_PATH, "experiments/",label))
            if not os.path.exists(os.path.join(SOURCE_PATH, "experiments/",label,model_name)):
                os.mkdir(os.path.join(SOURCE_PATH, "experiments/",label,model_name))
            #experiment_name = os.path.join(label + "_Per_class_", strftime("%Y-%m-%d_%H-%M-%S", localtime()))
            fit_config = {
                "batch_size":32,
                "epochs": 20,
                "initial_epoch": 0,
                "callbacks": [
                    ModelCheckpoint(os.path.join(exp_dir, "last_iter.h5"),
                                    save_weights_only=False),
                    ModelCheckpoint(os.path.join(exp_dir, "best_eval.h5"),
                                    save_best_only=True,
                                    monitor="val_loss",
                                    save_weights_only=False),
                    EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1,mode='min', restore_best_weights=True)
                ]
            }
            optimization = optimizers.Adadelta()
            model = get_model(model_name,INPUT_SHAPE)
            compile_model(model,optimizer= optimization)
            # Save model architecture and # of parameters to disc
            history = model.fit(X_train,y_train, validation_data=(X_val, y_val),verbose = 2, **fit_config);
            with open(os.path.join(exp_dir, "model_summary.txt"),'w+') as fh:
                model.summary(print_fn=lambda x: fh.write(x + '\n'))
            test_pred_prob = model.predict(spectrograms)
            test_pred = np.round(test_pred_prob)  
            accuracy, auc_roc, recall, precision, f1 = evaluate_model(test_pred_prob,test_pred, spectrograms, test_classes,
                                                          saving_path=os.path.join(exp_dir))
            get_TP_TN_FP_FN(songs_ID,test_pred,test_classes,os.path.join(exp_dir),label) # saving samples of true negative, false negativest etc..
            plot_loss_acuracy(history,os.path.join(exp_dir),label)
            plot_confusion_matrix(test_classes,test_pred,["Negative","Positive"],os.path.join(exp_dir),label)
            #labels_results[idx,:] = [label,model_name,len(training_classes),accuracy, auc_roc, recall, precision, f1]
            labels_results.append([label,model_name,model.count_params(),len(training_classes),accuracy, auc_roc, recall, precision, f1])    
    labels_results = np.asarray(labels_results)
    labels_results_df = pd.DataFrame(labels_results, columns  = [ "Class","Model","# paramters" , "Training Size","Accuracy", "AUC_ROC", "Recall", "Precision", "f1"])
    labels_results_df.to_csv(os.path.join(SOURCE_PATH, "experiments/", strftime("%Y-%m-%d_%H-%M-%S", localtime()) + '.csv'),float_format='%.3f')
     
     
if __name__ == "__main__":
    main()


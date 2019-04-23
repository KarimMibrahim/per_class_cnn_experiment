#! /bin/bash
#$ -cwd
#$ -N 'cluster_test'
#$ -M karim.m.ibraheem@gmail.com
#$ -m abes
#$ -S /bin/bash
#$ -j y
#$ -l hostname=tsicluster15
#$ -l gpu=1
#$ -v CUDA_VISIBLE_DEVICES=0
#$ -o /ldaphome/kibrahim/per_class_cnn_experiment/C2_adadelta_30secs15.txt
#$ -e /ldaphome/kibrahim/per_class_cnn_experiment/C2_adadelta_30secs.log

source /ldaphome/kibrahim/context_classification_cnn/env_tf/bin/activate
python ./C2_adadelta_30secs.py
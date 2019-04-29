#! /bin/bash
#$ -cwd
#$ -N 'cluster_test'
#$ -M karim.m.ibraheem@gmail.com
#$ -m abes
#$ -S /bin/bash
#$ -j y
#$ -l hostname=tsicluster12
#$ -l gpu=1
#$ -o /ldaphome/kibrahim/per_class_cnn_experiment/AD_EXP1.txt
#$ -e /ldaphome/kibrahim/per_class_cnn_experiment/AD_EXP1.log

source /ldaphome/kibrahim/context_classification_cnn/env_tf/bin/activate
python ./per_class_cnn_model.py 
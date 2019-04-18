#! /bin/bash
#$ -cwd
#$ -N 'cluster_test'
#$ -M karim.m.ibraheem@gmail.com
#$ -m abes
#$ -S /bin/bash
#$ -j y
#$ -l hostname=tsicluster12
#$ -l gpu=1
#$ -o /ldaphome/kibrahim/per_class_cnn_experiment/C2_sgd_60secs.txt
#$ -e /ldaphome/kibrahim/per_class_cnn_experiment/C2_sgd_60secss.log

source /ldaphome/kibrahim/context_classification_cnn/env_tf/bin/activate
python ./C2_sgd_60secs.py
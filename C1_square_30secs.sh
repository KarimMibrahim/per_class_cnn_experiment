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
#$ -o /ldaphome/kibrahim/per_class_cnn_experiment/C1_square_30secs.txt
#$ -e /ldaphome/kibrahim/per_class_cnn_experiment/C1_square_30secs.log

source /ldaphome/kibrahim/context_classification_cnn/env_tf/bin/activate
python ./C1_square_30secs.py
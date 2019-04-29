#! /bin/bash
#$ -cwd
#$ -N 'cluster_test'
#$ -M karim.m.ibraheem@gmail.com
#$ -m abes
#$ -S /bin/bash
#$ -j y
#$ -l hostname=tsicluster13
#$ -l gpu=1
#$ -v CUDA_VISIBLE_DEVICES=0
#$ -o /ldaphome/kibrahim/per_class_cnn_experiment/C4_adaldelta_30secs13.txt
#$ -e /ldaphome/kibrahim/per_class_cnn_experiment/C4_adaldelta_30secs.log

source /ldaphome/kibrahim/context_classification_cnn/env_tf_1.2/bin/activate
python ./C4_adaldelta_30secs.py
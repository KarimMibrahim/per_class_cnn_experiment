#! /bin/bash
#$ -cwd
#$ -N 'results_analysis'
#$ -M karim.m.ibraheem@gmail.com
#$ -m abes
#$ -S /bin/bash
#$ -j y
#$ -l hostname=tsicluster11
#$ -l gpu=1
#$ -o /ldaphome/kibrahim/per_class_classifier/results_analysis.txt
#$ -e /ldaphome/kibrahim/per_class_classifier/results_analysis.log

source /ldaphome/kibrahim/context_classification_cnn/env_tf/bin/activate
python ./per_class_results_analysis.py
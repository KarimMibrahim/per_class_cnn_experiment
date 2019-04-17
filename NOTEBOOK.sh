#! /bin/bash
#$ -cwd
#$ -N 'Notebook_test'
#$ -M karim.m.ibraheem@gmail.com
#$ -m abes
#$ -S /bin/bash
#$ -j y
#$ -l hostname=tsicluster11
#$ -l gpu=1
#$ -o /ldaphome/kibrahim/per_class_classifier/NOTEBOOK.txt
#$ -e /ldaphome/kibrahim/per_class_classifier/NOTEBOOK.log

source /ldaphome/kibrahim/context_classification_cnn/env_tf/bin/activate
jupyter notebook --no-browser --port=8888
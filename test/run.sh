#!/bin/sh

python keras_nn_test.py classification
python keras_nn_score_test.py classification

python keras_nn_test.py regression
python keras_nn_score_test.py regression

python keras_nn_test.py text_classification
python keras_nn_score_test.py text_classification

python keras_nn_test.py time_series
python keras_nn_score_test.py time_series

python keras_nn_test.py unsupervised
python keras_nn_score_test.py unsupervised
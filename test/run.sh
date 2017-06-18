#!/bin/bash

if [ ! -z "$1" ]; 
then
  echo running test $1
  echo
  python keras_nn_test.py $1
  python keras_nn_score_test.py $1
else
  for test in classification regression autoregression text_classification time_series unsupervised
  do
    echo running test $test
    echo
    python keras_nn_test.py $test
    python keras_nn_score_test.py $test
  done
fi


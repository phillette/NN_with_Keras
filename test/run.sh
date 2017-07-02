#!/bin/bash

(cd ../examples; ./unzip.sh)

if [ ! -z "$1" ]; 
then
  echo running test $1
  echo
  python keras_nn_test.py $1
  python keras_nn_score_test.py $1
else
  python testdriver.py
fi


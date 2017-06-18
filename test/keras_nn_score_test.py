
import sys

order_field = ''
backend = 'theano'

vocabulary_size = 0
word_limit = 0
window_size = 0
look_ahead = 0
num_unsupervised_outputs = 0

testname = sys.argv[1]
modelpath = "/tmp/dnn.model"
modelmetadata_path = "/tmp/dnn.metadata"
override_output_layer1 = '0'

if testname == "regression" or testname == "autoregression":
    input_type = 'predictor_fields'
    fields = "sepal_length,sepal_width,petal_length"
    target = "petal_width"
    objective = "regression"
    datafile = "~/Datasets/iris.csv"
elif testname == "text_classification":
    input_type = 'text'
    text_field = 'text'
    datafile = "~/Datasets/movie-pang02.csv"
    target = "class"
    vocabulary_size = 20000
    word_limit = 200
    objective = "classification"
elif testname == "time_series":
    fields = "wp2,wp3"
    input_type = 'predictor_fields'
    target = 'wp1'
    datafile = "~/Datasets/wind_forecasting.csv"
    objective = "time_series"
    window_size = 50
    look_ahead = 10
elif testname == "unsupervised":
    input_type = 'predictor_fields'
    fields = "sepal_length,sepal_width,petal_length,petal_width"
    target = ''
    datafile = "~/Datasets/iris.csv"
    objective = "unsupervised"
    num_unsupervised_outputs = 3
    override_output_layer1 = "1"
elif testname == "classification":
    input_type = 'predictor_fields'
    fields = "sepal_length,sepal_width,petal_length,petal_width"
    target = "species"
    objective = "classification"
    datafile = "~/Datasets/iris.csv"
else:
    raise Exception("unknown testname %s" % (testname))

from testrunner import run
run("keras_nn_score.py", "/tmp/test.py",globals())
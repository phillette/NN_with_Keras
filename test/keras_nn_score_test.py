
import sys

order_field = ''
backend = 'theano'

vocabulary_size = 0
word_limit = 0
window_size = 0
look_ahead = 0
num_unsupervised_outputs = 0

testname = sys.argv[1]

if testname == "regression":
    input_type = 'predictor_fields'
    fields = "sepal_length,sepal_width,petal_length"
    target = "petal_width"
    modelpath = "/tmp/dnn.model.reg"
    modelmetadata_path = "/tmp/dnn.metadata.reg"
    objective = "regression"
    datafile = "~/Datasets/iris.csv"
elif testname == "text_classification":
    input_type = 'text'
    text_field = 'text'
    datafile = "~/Datasets/movie-pang02.csv"
    target = "class"
    modelpath = "/tmp/dnn.model.txtclass"
    modelmetadata_path = "/tmp/dnn.metadata.txtclass"
    vocabulary_size = 20000
    word_limit = 200
    objective = "classification"
elif testname == "time_series":
    input_type = 'predictor_fields'
    target = 'value'
    datafile = "~/Datasets/sinwave.csv"
    modelpath = "/tmp/dnn.model.timeseries"
    modelmetadata_path = "/tmp/dnn.metadata.timeseries"
    objective = "time_series"
    window_size = 50
    look_ahead = 10
elif testname == "unsupervised":
    input_type = 'predictor_fields'
    fields = "sepal_length,sepal_width,petal_length,petal_width"
    target = ''
    datafile = "~/Datasets/iris.csv"
    modelpath = "/tmp/dnn.model.unsupervised"
    modelmetadata_path = "/tmp/dnn.metadata.unsupervised"
    objective = "unsupervised"
    num_unsupervised_outputs = 3
    override_output_layer = "1"
elif testname == "classification":
    input_type = 'predictor_fields'
    fields = "sepal_length,sepal_width,petal_length,petal_width"
    target = "species"
    modelpath = "/tmp/dnn.model.class"
    modelmetadata_path = "/tmp/dnn.metadata.class"
    objective = "classification"
    datafile = "~/Datasets/iris.csv"
else:
    raise Exception("unknown testname %s" % (testname))

from testrunner import run
run("keras_nn_score.py", "/tmp/test.py",globals())
import os
import os.path
import sys
wd = os.getcwd()

backend = 'theano'
num_epochs = 15
batch_size = 32
target_scaling = 'minmax'
validation_split = 0.2
input_type = 'predictor_fields'
datafile = "~/Datasets/iris.csv"
verbose = 'Y'
redirect_output = False
output_path = "/tmp/test.log"

record_stats = True
stats_output_path = '/tmp/metrics.csv'
order_field = ''
override_output_layer = "none"
network_configuration = "manual"

layer_0_type = 'none'
layer_0_parameter = ""
layer_1_type = 'none'
layer_1_parameter = ""
layer_2_type = 'none'
layer_2_parameter = ""
layer_3_type = 'none'
layer_3_parameter = ""
layer_4_type = 'none'
layer_4_parameter = ""
layer_5_type = 'none'
layer_5_parameter = ""
layer_6_type = 'none'
layer_6_parameter = ""
layer_7_type = 'none'
layer_7_parameter = ""
layer_8_type = 'none'
layer_8_parameter = ""
layer_9_type = 'none'
layer_9_parameter = ""
window_size=0
look_ahread=0
vocabulary_size = 0
word_limit = 0
num_unsupervised_outputs = 0
image_width = 0
image_height = 0
image_depth = 0
look_ahead = 0
text_field = ''

testname = sys.argv[1]

if testname == "regression":
    fields = "sepal_length,sepal_width,petal_length"
    target = "petal_width"
    loss_function = 'mean_squared_error'
    optimizer = 'adam'
    modelpath = "/tmp/dnn.model.reg"
    modelmetadata_path = "/tmp/dnn.metadata.reg"
    objective = "regression"
    layer_0_type = 'dense'
    layer_0_parameter = '16, activation="tanh", W_regularizer=l2(0.001)'
elif testname == "time_series":
    fields = []
    target = "value"
    loss_function = 'mean_squared_error'
    optimizer = 'adam'
    modelpath = "/tmp/dnn.model.timeseries"
    modelmetadata_path = "/tmp/dnn.metadata.timeseries"
    objective = "time_series"
    window_size=50
    look_ahread=10
    layer_0_type = 'lstm'
    layer_0_parameter = "50, return_sequences=True"
    layer_1_type = 'dropout'
    layer_1_parameter = "0.2"
    layer_2_type = 'lstm'
    layer_2_parameter = "100"
    layer_3_type = 'dropout'
    layer_3_parameter = "0.2"
    datafile = "~/Datasets/sinwave.csv"
elif testname == "text_classification":
    num_epochs = 4
    objective = "classification"
    input_type = 'text'
    datafile = "~/Datasets/movie-pang02.csv"
    text_field = "text"
    target = "class"
    modelpath = "/tmp/dnn.model.txtclass"
    modelmetadata_path = "/tmp/dnn.metadata.txtclass"
    loss_function = 'categorical_crossentropy'
    optimizer = 'adam'
    vocabulary_size = 20000
    word_limit = 200
    layer_0_type = 'embedding'
    layer_0_parameter = '20000, 128'
    layer_1_type = 'lstm'
    layer_1_parameter = '128, dropout=0.2, recurrent_dropout=0.2'
elif testname == "unsupervised":
    fields = "sepal_length,sepal_width,petal_length,petal_width"
    target = ""
    loss_function = 'mean_squared_error'
    optimizer = 'adam'
    modelpath = "/tmp/dnn.model.unsupervised"
    modelmetadata_path = "/tmp/dnn.metadata.unsupervised"
    objective = "unsupervised"
    num_epochs=100
    layer_0_type = 'dense'
    layer_0_parameter = "3, activation='softmax'"
    override_output_layer = "1"
elif testname == "classification":
    fields = "sepal_length,sepal_width,petal_length,petal_width"
    target = "species"
    loss_function = 'categorical_crossentropy'
    optimizer = 'adam'
    modelpath = "/tmp/dnn.model.class"
    modelmetadata_path = "/tmp/dnn.metadata.class"
    objective = "classification"
    layer_0_type = 'dense'
    layer_0_parameter = '16, activation="tanh", W_regularizer=l2(0.001)'
    layer_1_type = '_custom_'
    layer_1_parameter = 'Dropout(0.5)'
else:
    raise Exception("unknown testname %s"%(testname))

from testrunner import run
run("keras_nn.py","/tmp/test.py",globals())
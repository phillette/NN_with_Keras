# encoding=utf-8
script_details = ("keras_nn.py",0.6)

import sys
import os
import pandas as pd
import json

class RedirectStream(object):

  def __init__(self, fname):
    try:
        self.dupf = open(fname,"wb")
    except:
        self.dupf = None

  def write(self, x):
    if self.dupf:
        self.dupf.write(x)
        self.dupf.flush()

  def flush(self):
    pass

ascontext=None

layer_types = []
layer_parameters = []

MAX_LAYERS=10
for layer_index in range(0,MAX_LAYERS):
    layer_types.append("none")
    layer_parameters.append("")

if len(sys.argv) > 1 and sys.argv[1] == "-test":
    import os
    wd = os.getcwd()

    backend = 'theano'
    num_epochs = 15
    batch_size = 32
    target_scaling = 'minmax'
    validation_split = 0.2
    input_type = 'predictor_fields'
    datafile = "~/Datasets/iris.csv"
    verbose = 1
    redirect_output = False
    output_path = "/tmp/test.log"

    record_stats = True
    stats_output_path = '/tmp/metrics.csv'

    if len(sys.argv) > 2 and sys.argv[2] == "regression":
        fields = ["sepal_length", "sepal_width", "petal_length"]
        target = "petal_width"
        loss_function = 'mean_squared_error'
        optimizer = 'adam'
        modelpath = "/tmp/dnn.model.reg"
        modelmetadata_path = "/tmp/dnn.metadata.reg"
        objective = "regression"
        layer_types[0] = 'dense'
        layer_parameters[0] = "16, activation='tanh', W_regularizer=l2(0.001)"
    elif len(sys.argv) > 2 and sys.argv[2] == "text_classification":
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
        layer_types[0] = 'embedding'
        layer_parameters[0] = '20000, 128'

        layer_types[1] = 'lstm'
        layer_parameters[1] = '128, dropout=0.2, recurrent_dropout=0.2'
    else:
        fields = ["sepal_length","sepal_width","petal_length","petal_width"]
        target = "species"
        loss_function = 'categorical_crossentropy'
        optimizer = 'adam'

        modelpath = "/tmp/dnn.model.class"
        modelmetadata_path = "/tmp/dnn.metadata.class"
        objective = "classification"
        layer_types[0] = 'dense'
        layer_parameters[0] = "16, activation='tanh', W_regularizer=l2(0.001)"
        layer_types[1] = '_custom_'
        layer_parameters[1] = 'Dropout(0.5)'

    df = pd.read_csv(datafile)

    import shutil
    try:
        shutil.rmtree(modelpath)
    except:
        pass
    os.mkdir(modelpath)
else:
    import spss.pyspark.runtime
    ascontext = spss.pyspark.runtime.getContext()
    sc = ascontext.getSparkContext()
    df = ascontext.getSparkInputData()
    fields = map(lambda x: x.strip(),"%%fields%%".split(","))
    target = '%%target%%'
    num_epochs = int('%%num_epochs%%')
    backend = '%%backend%%'
    batch_size = int('%%batch_size%%')
    input_type = '%%input_type%%'
    text_field = '%%text_field%%'
    vocabulary_size = int('%%vocabulary_size%%')
    word_limit = int('%%word_limit%%')
    validation_split = float('%%validation_split%%')
    verbose = 0
    if '%%verbose%%' == 'Y':
        verbose = 1

    redirect_output = False
    if '%%redirect_output%%' == 'Y':
        redirect_output = True
    output_path = '%%output_path%%'

    record_stats = False
    if '%%record_stats%%' == 'Y':
        record_stats = True
    stats_output_path = '%%stats_output_path%%'

    loss_function = '%%loss_function%%'
    optimizer = '%%optimizer%%'
    objective = '%%objective%%'
    target_scaling = '%%target_scaling%%'

    layer_types[0] = '%%layer_0_type%%'
    layer_parameters[0] = '%%layer_0_parameter%%'

    layer_types[1] = '%%layer_1_type%%'
    layer_parameters[1] = '%%layer_1_parameter%%'

    layer_types[2] = '%%layer_2_type%%'
    layer_parameters[2] = '%%layer_2_parameter%%'

    layer_types[3] = '%%layer_3_type%%'
    layer_parameters[3] = '%%layer_3_parameter%%'

    layer_types[4] = '%%layer_4_type%%'
    layer_parameters[4] = '%%layer_4_parameter%%'

    layer_types[5] = '%%layer_5_type%%'
    layer_parameters[5] = '%%layer_5_parameter%%'

    layer_types[6] = '%%layer_6_type%%'
    layer_parameters[6] = '%%layer_6_parameter%%'

    layer_types[7] = '%%layer_7_type%%'
    layer_parameters[7] = '%%layer_7_parameter%%'

    layer_types[8] = '%%layer_8_type%%'
    layer_parameters[8] = '%%layer_8_parameter%%'

    layer_types[9] = '%%layer_9_type%%'
    layer_parameters[9] = '%%layer_9_parameter%%'

    from os import tempnam
    modelpath = tempnam()
    os.mkdir(modelpath)
    df = df.toPandas()

if redirect_output:
    r = RedirectStream(output_path)
    sys.stdout = r
    sys.stderr = r

if backend != "default":
    os.environ["KERAS_BACKEND"] = backend

from keras.models import Sequential
from keras.layers import *
from keras.regularizers import *

layerNames = {
    'dense':'Dense',
    'conv1d':'Conv1D',
    'conv2d':'Conv2D',
    'conv3d':'Conv3D',
    'flatten':'Flatten',
    'activation':'Activation',
    'reshape':'Reshape',
    'dropout':'Dropout',
    'lambda':'Lambda',
    'maxPooling1d':'MaxPooling1D',
    'maxPooling2d':'MaxPooling2D',
    'maxPooling3d':'MaxPooling3D',
    'embedding':'Embedding',
    'lstm':'LSTM'
}


class LayerFactory(object):

    def __init__(self,predictor_count):
        self.predictor_count = predictor_count
        self.first_layer = True

    def createLayer(self,layer_type,layer_parameter):
        if layer_type == "_custom_":
            self.first_layer = False
            return eval(layer_parameter)
        if self.first_layer:
            if layer_parameter:
                layer_parameter += ", "
            layer_parameter += 'input_shape=('+str(self.predictor_count)+',)'
            self.first_layer = False
        if layer_type not in layerNames:
            raise Exception("Invalid layer type:" + layer_type)
        ctr = layerNames[layer_type]+"("
        if layer_parameter:
            ctr += layer_parameter
        ctr += ")"
        return eval(ctr)

if input_type == "text":
    from keras.preprocessing.text import one_hot
    from keras.preprocessing.sequence import pad_sequences
    X = pad_sequences(df.apply(lambda row:one_hot(row[text_field].encode("utf-8"),vocabulary_size),axis=1),word_limit)
    fields = [text_field]
    num_features = word_limit
else:
    X = df.as_matrix(fields)
    num_features = len(fields)

y = pd.DataFrame()

model_metadata = { "predictors": fields, "target":target }

target_values = []

if objective == "classification":

    # get the list of unique target values
    target_values = list(df[target].unique())

    # one hot encoding step
    def encode(c, target_value):
        if c == target_value:
            return 1
        else:
            return 0

    for target_value in target_values:
        y[target_value] = df.apply(lambda row:encode(row[target],target_value),axis=1).astype(int)

    model_metadata["target_values"] = target_values

if objective == "regression":

    if target_scaling == 'minmax':
        min = df[target].min()
        max = df[target].max()

        y["target"] = df.apply(lambda row:(row[target] - min)/(max-min),axis=1).astype(float)

        model_metadata["target_max"] = max
        model_metadata["target_min"] = min
    else:
        y["target"] = df.apply(lambda row: row[target], axis=1).astype(float)



model = Sequential()
lf = LayerFactory(num_features)

for layer_index in range(0,MAX_LAYERS):
    if layer_types[layer_index] != 'none':
        model.add(lf.createLayer(layer_types[layer_index],layer_parameters[layer_index].strip()))

if objective == "classification":
    model.add(Dense(len(target_values), activation="softmax"))
else:
    model.add(Dense(1))

model.compile(loss=loss_function,
              metrics=['accuracy'],
              optimizer=optimizer)

# Build the model

h = model.fit(X, y.as_matrix(), verbose=verbose, batch_size=batch_size, nb_epoch=num_epochs, validation_split=validation_split)
if record_stats:
    keys = ["epoch"]
    keys += sorted(key for key in h.history if isinstance(h.history[key],list))
    sf = open(stats_output_path,"w")
    sf.write(",".join(keys)+chr(10))
    for e in range(0,num_epochs):
        vals = []
        for k in keys:
            if k == "epoch":
                vals.append(str(e+1))
            else:
                vc = h.history[k]
                if e < len(vc):
                    vals.append(str(vc[e]))
                else:
                    vals.append("?")
        sf.write(",".join(vals)+chr(10))

model.save(os.path.join(modelpath,"model"))

s_metadata = json.dumps(model_metadata)

if ascontext:
    ascontext.setModelContentFromPath("model",modelpath)
    ascontext.setModelContentFromString("model.metadata",s_metadata)
else:
    open(modelmetadata_path,"w").write(s_metadata)


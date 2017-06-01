# encoding=utf-8
script_details = ("keras_nn.py",0.5)

import sys
import os
import pandas as pd
import json

ascontext=None

layer_types = []
layer_parameters = []
layer_extras = []

MAX_LAYERS=10
for layer_index in range(0,MAX_LAYERS):
    layer_types.append("none")
    layer_parameters.append("")
    layer_extras.append("")

if len(sys.argv) > 1 and sys.argv[1] == "-test":
    import os
    wd = os.getcwd()
    df = pd.read_csv("~/Datasets/iris.csv")
    backend = 'theano'
    num_epochs = 200
    batch_size = 32
    target_scaling = 'minmax'

    if len(sys.argv) > 2 and sys.argv[2] == "regression":
        fields = ["sepal_length", "sepal_width", "petal_length"]
        target = "petal_width"
        loss_function = 'mean_squared_error'
        optimizer = 'adam'
        verbose = 1
        modelpath = "/tmp/dnn.model.reg"
        modelmetadata_path = "/tmp/dnn.metadata.reg"
        objective = "regression"
    else:
        fields = ["sepal_length","sepal_width","petal_length","petal_width"]
        target = "species"
        loss_function = 'categorical_crossentropy'
        optimizer = 'adam'
        verbose = 1
        modelpath = "/tmp/dnn.model.class"
        modelmetadata_path = "/tmp/dnn.metadata.class"
        objective = "classification"

    layer_types[0] = 'dense'
    layer_parameters[0] = '16'
    layer_extras[0] = "activation='tanh', W_regularizer=l2(0.001)"
    layer_types[1] = 'dropout'
    layer_parameters[1] = '0.5'
    layer_extras[1] = ''


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
    verbose = 0
    if '%%verbose%%' == 'Y':
        verbose = 1
    loss_function = '%%loss_function%%'
    optimizer = '%%optimizer%%'
    objective = '%%objective%%'
    target_scaling = '%%target_scaling%%'

    layer_types[0] = '%%layer_0_type%%'
    layer_parameters[0] = '%%layer_0_parameter%%'
    layer_extras[0] = '%%layer_0_extras%%'
    layer_types[1] = '%%layer_1_type%%'
    layer_parameters[1] = '%%layer_1_parameter%%'
    layer_extras[1] = '%%layer_1_extras%%'
    layer_types[2] = '%%layer_2_type%%'
    layer_parameters[2] = '%%layer_2_parameter%%'
    layer_extras[2] = '%%layer_2_extras%%'
    layer_types[3] = '%%layer_3_type%%'
    layer_parameters[3] = '%%layer_3_parameter%%'
    layer_extras[3] = '%%layer_3_extras%%'
    layer_types[4] = '%%layer_4_type%%'
    layer_parameters[4] = '%%layer_4_parameter%%'
    layer_extras[4] = '%%layer_4_extras%%'
    layer_types[5] = '%%layer_5_type%%'
    layer_parameters[5] = '%%layer_5_parameter%%'
    layer_extras[5] = '%%layer_5_extras%%'
    layer_types[6] = '%%layer_6_type%%'
    layer_parameters[6] = '%%layer_6_parameter%%'
    layer_extras[6] = '%%layer_6_extras%%'
    layer_types[7] = '%%layer_7_type%%'
    layer_parameters[7] = '%%layer_7_parameter%%'
    layer_extras[7] = '%%layer_7_extras%%'
    layer_types[8] = '%%layer_8_type%%'
    layer_parameters[8] = '%%layer_8_parameter%%'
    layer_extras[8] = '%%layer_8_extras%%'
    layer_types[9] = '%%layer_9_type%%'
    layer_parameters[9] = '%%layer_9_parameter%%'
    layer_extras[9] = '%%layer_9_extras%%'

    from os import tempnam
    modelpath = tempnam()
    os.mkdir(modelpath)
    df = df.toPandas()

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
    'maxPooling3d':'MaxPooling3D'
}


class LayerFactory(object):

    def __init__(self,predictor_count):
        self.predictor_count = predictor_count
        self.first_layer = True

    def createLayer(self,layer_type,layer_parameter,layer_extras):
        if self.first_layer:
            if layer_extras:
                layer_extras += ", "
            layer_extras += 'input_shape=('+str(self.predictor_count)+',)'
            self.first_layer = False
        if layer_type not in layerNames:
            raise Exception("Invalid layer type:" + layer_type)
        ctr = layerNames[layer_type]+"("
        if layer_parameter:
            ctr += layer_parameter
        if layer_extras:
            if layer_parameter:
                ctr += ","
            ctr += layer_extras
        ctr += ")"
        return eval(ctr)



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

X = df.as_matrix(fields)

model = Sequential()
lf = LayerFactory(len(fields))

for layer_index in range(0,MAX_LAYERS):
    if layer_types[layer_index] != 'none':
        model.add(lf.createLayer(layer_types[layer_index],layer_parameters[layer_index].strip(),layer_extras[layer_index].strip()))

if objective == "classification":
    model.add(Dense(len(target_values), activation="softmax"))
else:
    model.add(Dense(1))

model.compile(loss=loss_function,
              metrics=['accuracy'],
              optimizer=optimizer)

# Build the model

model.fit(X, y.as_matrix(), verbose=verbose, batch_size=batch_size, nb_epoch=num_epochs)

model.save(os.path.join(modelpath,"model"))

s_metadata = json.dumps(model_metadata)

if ascontext:
    ascontext.setModelContentFromPath("model",modelpath)
    ascontext.setModelContentFromString("model.metadata",s_metadata)
else:
    open(modelmetadata_path,"w").write(s_metadata)


script_details = ("keras_nn.py",0.1)

import sys
import os
import pandas as pd
import json

ascontext=None

layer_types = []
layer_parameters = []
layer_extras = []

MAX_LAYERS=2
for layer_index in range(0,MAX_LAYERS):
    layer_types.append("none")
    layer_parameters.append("")
    layer_extras.append("")

if len(sys.argv) > 1 and sys.argv[1] == "-test":
    import os
    wd = os.getcwd()
    df = pd.read_csv("~/Datasets/iris.csv")
    # specify predictors and target
    fields = ["sepal_length","sepal_width","petal_length","petal_width"]
    target = "species"
    backend = 'theano'
    num_epochs = 1000
    learning_rate = 0.01
    modelpath = "/tmp/dnn.model"

    layer_types[0] = 'dense'
    layer_parameters[0] = '16'
    layer_extras[0] = "activation='tanh' W_regularizer=l2(0.001)"
    layer_types[1] = 'dropout'
    layer_parameters[1] = '0.5'
    layer_extras[1] = ''

    modelmetadata_path = "/tmp/dnn.metadata"
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

    layer_types[0] = '%%layer_0_type%%'
    layer_parameters[0] = '%%layer_0_parameter%%'
    layer_extras[0] = '%%layer_0_extras%%'
    layer_types[1] = '%%layer_1_type%%'
    layer_parameters[1] = '%%layer_1_parameter%%'
    layer_extras[1] = '%%layer_1_extras%%'

    from os import tempnam
    modelpath = tempnam()
    os.mkdir(modelpath)
    df = df.toPandas()

if backend != "default":
    os.environ["KERAS_BACKEND"] = backend

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2

def parseExtras(layer_args,extras):
    pairs = extras.split()
    for pair in pairs:
        pos = pair.index("=")
        if pos > 0:
            name = pair[:pos]
            value = pair[pos+1:]
            if name not in layer_args:
                layer_args[name] = eval(value)

class LayerFactory(object):

    def __init__(self,predictor_count):
        self.predictor_count = predictor_count
        self.first_layer = True

    def createLayer(self,layer_type,layer_parameter,layer_extras):
        layer_args = {}
        parseExtras(layer_args,layer_extras)
        if self.first_layer:
            layer_args['input_shape'] = (self.predictor_count,)
            self.first_layer = False
        if layer_type == 'dense':
            sz = int(layer_parameter)
            return Dense(sz,**layer_args)
        if layer_type == 'dropout':
            frac = float(layer_parameter)
            return Dropout(frac,**layer_args)
        raise Exception("Invalid layer type:"+layer_type)

# get the list of unique target values
target_values = list(df[target].unique())

# one hot encoding step

def encode(c,target_value):
    if c == target_value:
        return 1
    else:
        return 0

y = pd.DataFrame()

for target_value in target_values:
    y[target_value] = df.apply(lambda row:encode(row[target],target_value),axis=1).astype(int)

X = df.as_matrix(fields)

model = Sequential()
lf = LayerFactory(len(fields))

for layer_index in range(0,MAX_LAYERS):
    if layer_types[layer_index] != 'none':
        model.add(lf.createLayer(layer_types[layer_index],layer_parameters[layer_index],layer_extras[layer_index]))

model.add(Dense(len(target_values), activation="softmax"))

model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'],
              optimizer='adam')

# Build the model

model.fit(X, y.as_matrix(), verbose=1, batch_size=1, nb_epoch=500)

model.save(os.path.join(modelpath,"model"))

model_metadata = { "predictors": fields, "target":target, "target_values":target_values }

s_metadata = json.dumps(model_metadata)

if ascontext:
    ascontext.setModelContentFromPath("model",modelpath)
    ascontext.setModelContentFromString("model.metadata",s_metadata)
else:
    open(modelmetadata_path,"w").write(s_metadata)


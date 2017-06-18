# encoding=utf-8
script_details = ("keras_nn.py", 0.8)

import sys
import os
import pandas as pd
import json
import math

# open("/tmp/test.py","w").write(open(__file__,"r").read())

class RedirectStream(object):
    def __init__(self, fname):
        try:
            self.dupf = open(fname, "wb")
        except:
            self.dupf = None

    def write(self, x):
        if self.dupf:
            self.dupf.write(x)
            self.dupf.flush()

    def flush(self):
        pass


ascontext = None

MAX_LAYERS = 10

# item_1033 = '%%item_1033%%'
# item_1035 = '%%item_1035%%'

class Configuration(object):
    def __init__(self):
        if "%%fields%%" == "":
            self.fields = []
        else:
            self.fields = map(lambda x: x.strip(), "%%fields%%".split(","))
        self.target = '%%target%%'
        self.num_epochs = int('%%num_epochs%%')
        self.backend = '%%backend%%'
        self.batch_size = int('%%batch_size%%')
        self.input_type = '%%input_type%%'
        self.text_field = '%%text_field%%'
        self.order_field = '%%order_field%%'
        self.vocabulary_size = int('%%vocabulary_size%%')
        self.word_limit = int('%%word_limit%%')
        self.validation_split = float('%%validation_split%%')
        self.window_size = int('%%window_size%%')
        self.look_ahead = int('%%look_ahead%%')
        self.verbose = 0
        if '%%verbose%%' == 'Y':
            self.verbose = 1

        self.redirect_output = False
        if '%%redirect_output%%' == 'Y':
            self.redirect_output = True
        self.output_path = '%%output_path%%'

        self.record_stats = False
        if '%%record_stats%%' == 'Y':
            self.record_stats = True
        self.stats_output_path = '%%stats_output_path%%'

        self.loss_function = '%%loss_function%%'
        self.optimizer = '%%optimizer%%'
        self.objective = '%%objective%%'
        self.target_scaling = '%%target_scaling%%'

        self.image_width = int('%%image_width%%')
        self.image_height = int('%%image_height%%')
        self.image_depth = int('%%image_depth%%')

        self.override_output_layer = '%%override_output_layer1%%'
        self.num_unsupervised_outputs = int('%%num_unsupervised_outputs%%')

        self.target_values = []
        self.seqtargets = None
        self.model_metadata = {}
        self.input_shape = None

        if self.loss_function == "auto":
            if self.objective == "classification":
                self.loss_function = "categorical_crossentropy"
            else:
                self.loss_function = "mean_squared_error"

        self.layer_types = []
        self.layer_parameters = []
        for layer_index in range(0, MAX_LAYERS):
            self.layer_types.append("none")
            self.layer_parameters.append("")

    def manual(self):
        self.layer_types[0] = '%%layer_0_type%%'
        self.layer_parameters[0] = '%%layer_0_parameter%%'

        self.layer_types[1] = '%%layer_1_type%%'
        self.layer_parameters[1] = '%%layer_1_parameter%%'

        self.layer_types[2] = '%%layer_2_type%%'
        self.layer_parameters[2] = '%%layer_2_parameter%%'

        self.layer_types[3] = '%%layer_3_type%%'
        self.layer_parameters[3] = '%%layer_3_parameter%%'

        self.layer_types[4] = '%%layer_4_type%%'
        self.layer_parameters[4] = '%%layer_4_parameter%%'

        self.layer_types[5] = '%%layer_5_type%%'
        self.layer_parameters[5] = '%%layer_5_parameter%%'

        self.layer_types[6] = '%%layer_6_type%%'
        self.layer_parameters[6] = '%%layer_6_parameter%%'

        self.layer_types[7] = '%%layer_7_type%%'
        self.layer_parameters[7] = '%%layer_7_parameter%%'

        self.layer_types[8] = '%%layer_8_type%%'
        self.layer_parameters[8] = '%%layer_8_parameter%%'

        self.layer_types[9] = '%%layer_9_type%%'
        self.layer_parameters[9] = '%%layer_9_parameter%%'

    def simple_nn(self):
        self.layer_types[0] = 'dense'
        self.layer_parameters[0] = '64, activation="sigmoid"'

    def auto(self):
        # attempt to auto configure the layers - based on any information supplied

        if self.input_type == "text":

            self.layer_types[0] = 'embedding'
            self.layer_parameters[0] = '%d,128' % (self.vocabulary_size)

            self.layer_types[1] = 'lstm'
            self.layer_parameters[1] = '128, dropout=0.2, recurrent_dropout=0.2'

        elif self.objective == "time_series":

            self.layer_types[0] = 'lstm'
            self.layer_parameters[0] = '50, return_sequences=True'

            self.layer_types[1] = 'dropout'
            self.layer_parameters[1] = '0.2'

            self.layer_types[2] = 'lstm'
            self.layer_parameters[2] = '100'

            self.layer_types[3] = 'dropout'
            self.layer_parameters[3] = '0.2'

        elif self.objective == "unsupervised":

            # try to create an auto-encoder to narrow down to 1/8 features...

            num_inputs_by_2 = int(math.ceil(len(self.fields) / 2.0))
            num_inputs_by_4 = int(math.ceil(num_inputs_by_2 / 2.0))
            num_inputs_by_8 = int(math.ceil(num_inputs_by_4 / 2.0))

            self.layer_types[0] = 'dense'
            self.layer_parameters[0] = '%d, activation="relu"' % (num_inputs_by_2)

            self.layer_types[1] = 'dense'
            self.layer_parameters[1] = '%d, activation="relu"' % (num_inputs_by_4)

            self.layer_types[2] = 'dense'
            self.layer_parameters[2] = '%d, activation="relu"' % (num_inputs_by_8)

            self.layer_types[3] = 'dense'
            self.layer_parameters[3] = '%d, activation="relu"' % (num_inputs_by_4)

            self.layer_types[4] = 'dense'
            self.layer_parameters[4] = '%d, activation="relu"' % (num_inputs_by_2)

        else:

            # for classification and regression, create a NN with 2 hidden layers?
            # reduce number of neurons by a factor of 3 at leach hidden layer

            num_inputs_by_3 = int(math.ceil(len(self.fields)/3.0))
            num_inputs_by_9 = int(math.ceil(num_inputs_by_3/3.0))


            self.layer_types[0] = 'dense'
            self.layer_parameters[0] = '%d, activation="sigmoid"'%(num_inputs_by_3)

            if num_inputs_by_9 < num_inputs_by_3:

                self.layer_types[1] = 'dense'
                self.layer_parameters[1] = '%d, activation="sigmoid"' % (num_inputs_by_9)

        self.dumpLayers("auto layers")

    def dumpLayers(self,title):
        print(title)
        print("="*len(title))
        for i in range(0,MAX_LAYERS):
            if self.layer_types[i] != "none":
                print("  Layer %d"%(i))
                print("    %s(%s)"%(self.layer_types[i],self.layer_parameters[i]))
        print("")

    def image_classifier(self):

        self.layer_types[0] = 'reshape'
        self.layer_parameters[0] = '(%d,%d,%d)' % (self.image_width, self.image_height, self.image_depth)

        self.layer_types[1] = 'conv2d'
        self.layer_parameters[1] = '32,(5,5)'

        self.layer_types[2] = 'maxPooling2d'

        self.layer_types[3] = 'dropout'
        self.layer_parameters[3] = '0.2'

        self.layer_types[4] = 'flatten'
        self.layer_types[5] = 'dense'
        self.layer_parameters[5] = "128, activation='relu'"

    def getFeatureMatrix(self, df):
        if cfg.input_type == "text":
            from keras.preprocessing.text import one_hot
            from keras.preprocessing.sequence import pad_sequences
            X = pad_sequences(
                df.apply(lambda row: one_hot(row[self.text_field].encode("utf-8"), self.vocabulary_size), axis=1),
                self.word_limit)
            self.fields = [cfg.text_field]
            self.input_shape = (self.word_limit,)
        elif self.objective == "time_series":
            num_series = 1+len(self.fields)
            data = [df[self.target].tolist()]
            num_rows = len(data[0])

            for field in self.fields:
                data.append(df[field].tolist())

            instances = []
            target_instances = []

            for index in range(num_rows - (self.window_size+1)):
                windows = []
                for windex in range(self.window_size):
                    series = []
                    for sindex in range(num_series):
                        series.append(data[sindex][index+windex])
                    windows.append(series)
                target_window = []
                for sindex in range(num_series):
                    target_window.append(data[sindex][index + self.window_size])
                instances.append(windows)
                target_instances.append(target_window)

            X = np.array(instances)
            self.seqtargets = np.array(target_instances)

            X = np.reshape(X, (X.shape[0], self.window_size, num_series))
            print(X.shape)
            self.input_shape = (self.window_size, num_series)
        else:
            X = df.as_matrix(self.fields)
            self.input_shape = (len(self.fields),)

        self.model_metadata["predictors"] = self.fields

        return X

    def getTargetMatrix(self, df):

        if self.objective == "classification":

            y = pd.DataFrame()

            # get the list of unique target values
            self.target_values = list(df[self.target].unique())

            # one hot encoding step
            def encode(c, target_value):
                if c == target_value:
                    return 1
                else:
                    return 0

            for target_value in self.target_values:
                y[target_value] = df.apply(lambda row: encode(row[self.target], target_value), axis=1).astype(int)

            y = y.as_matrix()
            self.model_metadata["target_values"] = self.target_values

        elif self.objective == "regression":
            y = pd.DataFrame()
            if self.target_scaling == 'minmax':
                min = df[self.target].min()
                max = df[self.target].max()

                y["target"] = df.apply(lambda row: (row[self.target] - min) / (max - min), axis=1).astype(float)

                self.model_metadata["target_max"] = max
                self.model_metadata["target_min"] = min
            else:
                y["target"] = df.apply(lambda row: row[self.target], axis=1).astype(float)

            y = y.as_matrix()

        elif self.objective == "time_series":
            y = self.seqtargets

        elif self.objective == "unsupervised":
            y = X[:, :]

        else:
            raise Exception("Unknown objective:" + self.objective)

        self.model_metadata["target"] = self.target
        return y

    def getOuptutLayer(self):
        if cfg.objective == "classification":
            return "Dense(%d, activation='softmax')" % (len(self.target_values))
        elif cfg.objective == "unsupervised":
            return "Dense(%d, activation='linear')" % (len(self.fields))
        elif cfg.objective == "time_series":
            return "Dense("+str(len(self.fields)+1)+",activation='linear')"
        else:
            return "Dense(1)"

    def getModelMetadata(self):
        return self.model_metadata

    def dumpMetrics(self):
        if self.record_stats:
            keys = ["epoch"]
            keys += sorted(key for key in h.history if isinstance(h.history[key], list))
            sf = open(self.stats_output_path, "w")
            sf.write(",".join(keys) + chr(10))
            for e in range(0, self.num_epochs):
                vals = []
                for k in keys:
                    if k == "epoch":
                        vals.append(str(e + 1))
                    else:
                        vc = h.history[k]
                        if e < len(vc):
                            vals.append(str(vc[e]))
                        else:
                            vals.append("?")
                sf.write(",".join(vals) + chr(10))


if len(sys.argv) > 1 and sys.argv[1] == "-test":
    import os

    wd = os.getcwd()

    datafile = "%%datafile%%"
    modelpath = "%%modelpath%%"
    modelmetadata_path = "%%modelmetadata_path%%"
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
    from os import tempnam

    modelpath = tempnam()
    os.mkdir(modelpath)
    df = df.toPandas()

network_configuration = '%%network_configuration%%'
cfg = Configuration()

if network_configuration == "manual":
    cfg.manual()
elif network_configuration == "simple_nn":
    cfg.simple_nn()
elif network_configuration == "auto":
    cfg.auto()
elif network_configuration == "image_classification":
    cfg.image_classifier()

if cfg.order_field:
    df = df.sort([cfg.order_field], ascending=[1])

if cfg.redirect_output:
    r = RedirectStream(cfg.output_path)
    sys.stdout = r
    sys.stderr = r

if cfg.backend != "default":
    os.environ["KERAS_BACKEND"] = cfg.backend

from keras.models import Sequential
from keras.layers import *
from keras.regularizers import *

layerNames = {
    'dense': 'Dense',
    'conv1d': 'Conv1D',
    'conv2d': 'Conv2D',
    'conv3d': 'Conv3D',
    'convolution1d' : 'Convolution1D',
    'flatten': 'Flatten',
    'activation': 'Activation',
    'reshape': 'Reshape',
    'dropout': 'Dropout',
    'lambda': 'Lambda',
    'maxPooling1d': 'MaxPooling1D',
    'maxPooling2d': 'MaxPooling2D',
    'maxPooling3d': 'MaxPooling3D',
    'embedding': 'Embedding',
    'lstm': 'LSTM'
}

class LayerFactory(object):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.first_layer = True
        self.code = ""

    def createLayer(self, layer_type, layer_parameter):
        print(layer_type+":"+layer_parameter)
        if layer_type == "_custom_":
            self.first_layer = False
            return eval(layer_parameter)
        if self.first_layer:
            if layer_parameter:
                layer_parameter += ", "
            layer_parameter += 'input_shape=' + str(self.input_shape)
            self.first_layer = False
        if layer_type not in layerNames:
            raise Exception("Invalid layer type:" + layer_type)
        ctr = layerNames[layer_type] + "("
        if layer_parameter:
            ctr += layer_parameter
        ctr += ")"
        self.code += "model.add(" + ctr + ")" + chr(10)
        return eval(ctr)

    def getCode(self):
        return self.code


# Define the input numpy matrix

X = cfg.getFeatureMatrix(df)

# Define the target numpy matrix

y = cfg.getTargetMatrix(df)

# Construct the model and add requested layers

code = """
# python code to define the keras model

from keras.models import Sequential
from keras.layers import *
from keras.regularizers import *

model = Sequential()

"""

model = Sequential()
lf = LayerFactory(cfg.input_shape)

for layer_index in range(0, MAX_LAYERS):
    if cfg.layer_types[layer_index] != 'none':
        model.add(lf.createLayer(cfg.layer_types[layer_index], cfg.layer_parameters[layer_index].strip()))

code += lf.getCode()

print(code)

model_metadata = cfg.getModelMetadata()

# Add an implicit output layer

output_layer = cfg.getOuptutLayer()
model.add(eval(output_layer))
code += "model.add(%s)" % (output_layer)
code += chr(10)
code += chr(10)

# Specify the loss function and optimizer, get ready to build

model.compile(loss=cfg.loss_function,
              metrics=['accuracy'],
              optimizer=cfg.optimizer)

code += "model.compile(loss=%s,metrics=['accuracy'],optimizer=%s)" % (cfg.loss_function, cfg.optimizer)

print(code)

# Build the model

h = model.fit(X, y, verbose=cfg.verbose, batch_size=cfg.batch_size, epochs=cfg.num_epochs,
              validation_split=cfg.validation_split)

# Write metrics to file if requested
cfg.dumpMetrics()

# For manual configuration if requested convert to a copy of the model comprising
# layers up to the specified output layer

if network_configuration == "manual" and cfg.override_output_layer != "none":
    originalmodel = model
    model = Sequential()
    lf = LayerFactory(cfg.input_shape)

    for layer_index in range(0, int(cfg.override_output_layer)):
        if cfg.layer_types[layer_index] != 'none':
            layer = lf.createLayer(cfg.layer_types[layer_index], cfg.layer_parameters[layer_index].strip())
            original_layer = originalmodel.get_layer(index=layer_index)
            layer.set_weights(original_layer.get_weights())
            model.add(layer)

# Persist the model and model metadata

model.save(os.path.join(modelpath, "model"))

s_metadata = json.dumps(model_metadata)

if ascontext:
    ascontext.setModelContentFromPath("model", modelpath)
    ascontext.setModelContentFromString("model.metadata", s_metadata)
else:
    open(modelmetadata_path, "w").write(s_metadata)

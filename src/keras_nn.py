script_details = ("keras_nn.py",0.1)

import sys
import os
import pandas as pd
import json

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2

ascontext=None
if len(sys.argv) > 1 and sys.argv[1] == "-test":
    import os
    wd = os.getcwd()
    df = pd.read_csv("~/Datasets/iris.csv")
    # specify predictors and target
    fields = ["sepal_length","sepal_width","petal_length","petal_width"]
    target = "species"
    hidden_layers = [16,0]
    num_epochs = 1000
    learning_rate = 0.01
    modelpath = "/tmp/dnn.model"
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
    hidden_layers = json.loads('[%%hidden_layers%%]')
    num_epochs = int('%%num_epochs%%')
    learning_rate = float('%%learning_rate%%')
    from os import tempnam
    modelpath = tempnam()
    os.mkdir(modelpath)
    df = df.toPandas()

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
model.add(Dense(hidden_layers[0],
                input_shape=(len(fields),),
                activation="tanh",
                W_regularizer=l2(0.001)))
if hidden_layers[1]:
    model.add(Dense(hidden_layers[1],
                activation="tanh",
                W_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(3, activation="softmax"))

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


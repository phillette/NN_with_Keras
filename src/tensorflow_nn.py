script_details = ("tensorflow_nn.py",0.1)

import time
import sys
import os
import pandas as pd
import numpy
import json

ascontext=None
if len(sys.argv) > 1 and sys.argv[1] == "-test":
    import os
    wd = os.getcwd()
    df = pd.read_csv("/home/niallm/Datasets/iris.csv")
    # specify predictors and target
    fields = ["SEPALLENGTH","SEPALWIDTH","PETALLENGTH","PETALWIDTH"]
    target = "CLASS"
    hidden_layers = [10,10]
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

for target_value in target_values:
    df[target_value] = df.apply(lambda row:encode(row[target],target_value),axis=1).astype(int)


import tensorflow as tf
sess = tf.InteractiveSession()

# create TF variables
n_inputs = len(fields)
n_outputs = len(target_values)

x = tf.placeholder(tf.float32, shape=[None, n_inputs])
y_ = tf.placeholder(tf.float32, shape=[None, n_outputs])

in_layer = x

# add hidden layers
for ucount in hidden_layers:
    w = tf.Variable(tf.random_normal([n_inputs, ucount]))
    b =  tf.Variable(tf.random_normal([ucount]))
    layer = tf.tanh(tf.add(tf.matmul(in_layer, w), b))
    n_inputs = ucount
    in_layer = layer

# add output layer
W = tf.Variable(tf.zeros([n_inputs,n_outputs]))
b = tf.Variable(tf.zeros([n_outputs]))

saver = tf.train.Saver()

y = tf.matmul(in_layer,W) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y,y_))
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

sess.run(tf.initialize_all_variables())

# get target and predictor 2-d arrays from the pandas dataframe
targetarr = numpy.array(df[target_values],dtype=int)
dataarr=numpy.array(df[fields])

for i in range(num_epochs):
    sess.run(optimizer,feed_dict={x: dataarr, y_: targetarr})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# print(sess.run(y,feed_dict={x: dataarr}))
print(sess.run(accuracy,feed_dict={x: dataarr, y_: targetarr}))

saver.save(sess, os.path.join(modelpath,"model"))

model_metadata = { "predictors": fields, "target":target, "target_values":target_values }

s_metadata = json.dumps(model_metadata)

if ascontext:
    ascontext.setModelContentFromPath("model",modelpath)
    ascontext.setModelContentFromString("model.metadata",s_metadata)
else:
    open(modelmetadata_path,"w").write(s_metadata)


script_details = ("tensorflow_nn_score.py",0.1)

import json
import sys
import pandas as pd
import time
import os
import numpy

ascontext=None

if len(sys.argv) > 1 and sys.argv[1] == "-test":
    import os
    wd = os.getcwd()
    df = pd.read_csv("/home/niallm/Datasets/iris.csv")
    # specify predictors and target
    fields = ["SEPALLENGTH","SEPALWIDTH","PETALLENGTH","PETALWIDTH"]
    target = "CLASS"
    hidden_layers=[10,10]
    modelpath = "/tmp/dnn.model"
    modelmetadata_path = "/tmp/dnn.metadata"

else:
    import spss.pyspark.runtime
    ascontext = spss.pyspark.runtime.getContext()
    sc = ascontext.getSparkContext()
    sqlCtx = ascontext.getSparkSQLContext()
    df = ascontext.getSparkInputData().toPandas()
    target = '%%target%%'
    fields =  map(lambda x: x.strip(),"%%fields%%".split(","))
    hidden_layers = json.loads('[%%hidden_layers%%]')
    schema = ascontext.getSparkInputSchema()

prediction_field = "$R-" + target

if ascontext:
    from pyspark.sql.types import StructField, StructType, StringType
    output_schema = StructType(schema.fields + [StructField(prediction_field, StringType(), nullable=True)])

    ascontext.setSparkOutputSchema(output_schema)
    if ascontext.isComputeDataModelOnly():
        sys.exit(0)
    else:
        modelpath = ascontext.getModelContentToPath("model")
        model_metadata = json.loads(ascontext.getModelContentToString("model.metadata"))
else:
    model_metadata = json.loads(open(modelmetadata_path,"r").read())

target_values = model_metadata["target_values"]

dataarr=numpy.array(df[fields])

import tensorflow as tf
sess = tf.InteractiveSession()

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

sess.run(tf.initialize_all_variables())

saver = tf.train.Saver()
saver.restore(sess,os.path.join(modelpath,"model"))

# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

result = sess.run(tf.argmax(y,1),feed_dict={x: dataarr})

# bring predictions into dataframe
df[prediction_field] = result

# recode to original class names
df[prediction_field] = df.apply(lambda row:target_values[row[prediction_field]],axis=1).astype(str)

if ascontext:
    outdf = sqlCtx.createDataFrame(df)
    ascontext.setSparkOutputData(outdf)
else:
    print(str(df))





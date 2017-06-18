# encoding=utf-8
script_details = ("keras_nn_score.py",0.8)

import json
import sys
import pandas as pd
import os
import numpy as np

from keras.models import load_model

ascontext=None

if len(sys.argv) > 1 and sys.argv[1] == "-test":
    datafile = "%%datafile%%"
    modelpath = "%%modelpath%%"
    modelmetadata_path = "%%modelmetadata_path%%"
    df = pd.read_csv(datafile)
else:
    import spss.pyspark.runtime
    ascontext = spss.pyspark.runtime.getContext()
    sc = ascontext.getSparkContext()
    sqlCtx = ascontext.getSparkSQLContext()
    df = ascontext.getSparkInputData().toPandas()

target = '%%target%%'
backend = '%%backend%%'
input_type = '%%input_type%%'
fields = []
if "%%fields%%" != "":
    fields =  map(lambda x: x.strip(),"%%fields%%".split(","))

text_field = '%%text_field%%'
vocabulary_size = int('%%vocabulary_size%%')
word_limit = int('%%word_limit%%')
window_size = int('%%window_size%%')
look_ahead = int('%%look_ahead%%')
objective = '%%objective%%'
order_field = '%%order_field%%'
override_output_layer = '%%override_output_layer%%'
num_unsupervised_outputs = int('%%num_unsupervised_outputs%%')

prefix = "$R"
prediction_field = prefix + "-" + target
probability_field = prefix + "P-" + target
step_field = prefix + "-STEP"

if order_field:
    df = df.sort([order_field],ascending=[1])

if ascontext:
    schema = ascontext.getSparkInputSchema()
    from pyspark.sql.types import StructField, StructType, StringType, FloatType
    added_fields = []
    if objective == 'classification':
        added_fields.append(StructField(prediction_field, StringType(), nullable=True))
        added_fields.append(StructField(probability_field, FloatType(), nullable=True))
        output_schema = StructType(schema.fields + added_fields)
    elif objective == 'regression':
        added_fields.append(StructField(prediction_field, FloatType(), nullable=True))
        output_schema = StructType(schema.fields + added_fields)
    elif objective == 'time_series':
        output_fields = [StructField(step_field, FloatType(), nullable=True),StructField(prediction_field, FloatType(), nullable=True)]
        for field in fields:
            output_fields.append(StructField("$R-"+field, FloatType(), nullable=True))
        output_schema = StructType(output_fields)
    elif objective == 'unsupervised':
        num_outputs = len(fields)
        if override_output_layer != "none":
            num_outputs = num_unsupervised_outputs
        output_schema = StructType(schema.fields + [StructField(prefix+"-"+str(i), FloatType(), nullable=True) for i in range(0,num_outputs)])

    ascontext.setSparkOutputSchema(output_schema)
    if ascontext.isComputeDataModelOnly():
        sys.exit(0)
    else:
        modelpath = ascontext.getModelContentToPath("model")
        model_metadata = json.loads(ascontext.getModelContentToString("model.metadata"))
else:
    model_metadata = json.loads(open(modelmetadata_path,"r").read())

if objective == "time_series":
    num_series = 1 + len(fields)
    data = [df[target].tolist()]
    num_rows = len(data[0])
    for field in fields:
        data.append(df[field].tolist())

    instances = []

    index = num_rows - (window_size + 1)
    windows = []
    for windex in range(window_size):
        series = []
        for sindex in range(num_series):
            series.append(data[sindex][index + windex])
        windows.append(series)

    instances.append(windows)

    dataarr = np.array(instances)

    dataarr = np.reshape(dataarr, (dataarr.shape[0], window_size, num_series))

elif input_type == "predictor_fields":
    dataarr=np.array(df[fields])
else:
    from keras.preprocessing.text import one_hot
    from keras.preprocessing.sequence import pad_sequences
    dataarr = pad_sequences(df.apply(lambda row: one_hot(row[text_field].encode("utf-8"), vocabulary_size), axis=1), word_limit)

score_model = load_model(os.path.join(modelpath,"model"))

if objective == "time_series":
    result = [[] for i in range(len(fields)+1)]
    steps = []
    for x in range(0,look_ahead):
        prediction = score_model.predict(dataarr)

        for i in range(len(fields)+1):
            result[i].append(prediction[0][i])
        dataarr = np.roll(dataarr,-1,axis=1)
        dataarr[0][window_size-1] = prediction[0]
        # result.append(prediction[0][0])
        steps.append(float(x+1))
else:
    result = score_model.predict(dataarr)

# bring predictions into dataframe
if objective == "classification":
    df[prediction_field] = np.argmax(result,axis=1)

    # recode predictions to original class names
    target_values = model_metadata["target_values"]
    df[prediction_field] = df.apply(lambda row:target_values[row[prediction_field]],axis=1).astype(str)
    df[probability_field] = np.max(result, axis=1).astype(float)

elif objective == "regression":
    df[prediction_field] = np.max(result, axis=1)

    if "target_max" in model_metadata and "target_min" in model_metadata:
        min = model_metadata["target_min"]
        max = model_metadata["target_max"]
        df[prediction_field] = df.apply(lambda row: min+(row[prediction_field] * (max-min)), axis=1).astype(float)

elif objective == "time_series":
    df = pd.DataFrame()
    df[step_field] = steps
    df[prediction_field] = result[0]
    i = 0
    for field in fields:
        i += 1
        df["$R-"+field] = result[i]

elif objective == "unsupervised":
    num_outputs = len(fields)
    if override_output_layer != "none":
        num_outputs = num_unsupervised_outputs
    nr_columns = result.shape[1]
    for i in range(0,num_outputs):
        if i < nr_columns:
            df[prefix+"-"+str(i)] = result[:,i]
        else:
            df[prefix+"-"+str(i)] = 0.0

if ascontext:
    outdf = sqlCtx.createDataFrame(df)
    ascontext.setSparkOutputData(outdf)
else:
    print(str(df))





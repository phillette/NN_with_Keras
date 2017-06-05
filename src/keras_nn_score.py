# encoding=utf-8
script_details = ("keras_nn_score.py",0.7)

import json
import sys
import pandas as pd
import os
import numpy as np

from keras.models import load_model

ascontext=None

if len(sys.argv) > 1 and sys.argv[1] == "-test":
    import os
    wd = os.getcwd()
    order_field = ''
    backend = 'theano'
    if len(sys.argv) > 2 and sys.argv[2] == "regression":
        input_type = 'predictor_fields'
        fields = ["sepal_length", "sepal_width", "petal_length"]
        target = "petal_width"
        modelpath = "/tmp/dnn.model.reg"
        modelmetadata_path = "/tmp/dnn.metadata.reg"
        objective = "regression"
        datafile = "~/Datasets/iris.csv"
    elif len(sys.argv) > 2 and sys.argv[2] == "text_classification":
        input_type = 'text'
        text_field = 'text'
        datafile = "~/Datasets/movie-pang02.csv"
        target = "class"
        modelpath = "/tmp/dnn.model.txtclass"
        modelmetadata_path = "/tmp/dnn.metadata.txtclass"
        vocabulary_size = 20000
        word_limit = 200
        objective = "classification"
    elif len(sys.argv) > 2 and sys.argv[2] == "time_series":
        input_type = 'predictor_fields'
        target = 'value'
        datafile = "~/Datasets/sinwave.csv"
        modelpath = "/tmp/dnn.model.timeseries"
        modelmetadata_path = "/tmp/dnn.metadata.timeseries"
        objective = "time_series"
        window_size = 50
        look_ahead = 10
    elif len(sys.argv) > 2 and sys.argv[2] == "unsupervised":
        input_type = 'predictor_fields'
        fields = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        target = ''
        datafile = "~/Datasets/iris.csv"
        modelpath = "/tmp/dnn.model.unsupervised"
        modelmetadata_path = "/tmp/dnn.metadata.unsupervised"
        objective = "unsupervised"
        num_unsupervised_outputs = 3
        override_output_layer = "1"
    else:
        input_type = 'predictor_fields'
        fields = ["sepal_length","sepal_width","petal_length","petal_width"]
        target = "species"
        modelpath = "/tmp/dnn.model.class"
        modelmetadata_path = "/tmp/dnn.metadata.class"
        objective = "classification"
        datafile = "~/Datasets/iris.csv"
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
    fields =  map(lambda x: x.strip(),"%%fields%%".split(","))
    text_field = '%%text_field%%'
    vocabulary_size = int('%%vocabulary_size%%')
    word_limit = int('%%word_limit%%')
    window_size = int('%%window_size%%')
    look_ahead = int('%%look_ahead%%')
    schema = ascontext.getSparkInputSchema()
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
        output_schema = StructType([StructField(step_field, FloatType(), nullable=True),
                                    StructField(prediction_field, FloatType(), nullable=True)])
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
    data = df[target].tolist()
    sequences = []
    index = len(data) - window_size
    sequences.append(data[index: index+window_size])
    seqarr = np.array(sequences)
    dataarr = np.reshape(seqarr, (1, window_size, 1))
elif input_type == "predictor_fields":
    dataarr=np.array(df[fields])
else:
    from keras.preprocessing.text import one_hot
    from keras.preprocessing.sequence import pad_sequences
    dataarr = pad_sequences(df.apply(lambda row: one_hot(row[text_field].encode("utf-8"), vocabulary_size), axis=1), word_limit)

score_model = load_model(os.path.join(modelpath,"model"))

if objective == "time_series":
    result = []
    steps = []
    for x in range(0,look_ahead):
        prediction = score_model.predict(dataarr)
        dataarr = np.roll(dataarr,-1,axis=1)
        dataarr[0][window_size-1] = [prediction]
        result.append(prediction[0][0])
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
    df[prediction_field] = result

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





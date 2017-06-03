# encoding=utf-8
script_details = ("keras_nn_score.py",0.6)

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
    schema = ascontext.getSparkInputSchema()
    objective = '%%objective%%'

prediction_field = "$R-" + target
probability_field = "$RP-" + target

if ascontext:
    from pyspark.sql.types import StructField, StructType, StringType, FloatType
    added_fields = []
    if objective == 'classification':
        added_fields.append(StructField(prediction_field, StringType(), nullable=True))
        added_fields.append(StructField(probability_field, FloatType(), nullable=True))
    if objective == 'regression':
        added_fields.append(StructField(prediction_field, FloatType(), nullable=True))

    output_schema = StructType(schema.fields + added_fields)

    ascontext.setSparkOutputSchema(output_schema)
    if ascontext.isComputeDataModelOnly():
        sys.exit(0)
    else:
        modelpath = ascontext.getModelContentToPath("model")
        model_metadata = json.loads(ascontext.getModelContentToString("model.metadata"))
else:
    model_metadata = json.loads(open(modelmetadata_path,"r").read())

if input_type == "predictor_fields":
    dataarr=np.array(df[fields])
else:
    from keras.preprocessing.text import one_hot
    from keras.preprocessing.sequence import pad_sequences
    dataarr = pad_sequences(df.apply(lambda row: one_hot(row[text_field].encode("utf-8"), vocabulary_size), axis=1), word_limit)

score_model = load_model(os.path.join(modelpath,"model"))

result = score_model.predict(dataarr)

# bring predictions into dataframe
if objective == "classification":
    df[prediction_field] = np.argmax(result,axis=1)
    df[probability_field] = np.max(result,axis=1)

    # recode predictions to original class names
    target_values = model_metadata["target_values"]
    df[prediction_field] = df.apply(lambda row:target_values[row[prediction_field]],axis=1).astype(str)

if objective == "regression":
    df[prediction_field] = np.max(result, axis=1)

    if "target_max" in model_metadata and "target_min" in model_metadata:
        min = model_metadata["target_min"]
        max = model_metadata["target_max"]
        df[prediction_field] = df.apply(lambda row: min+(row[prediction_field] * (max-min)), axis=1).astype(float)


if ascontext:
    outdf = sqlCtx.createDataFrame(df)
    ascontext.setSparkOutputData(outdf)
else:
    print(str(df))





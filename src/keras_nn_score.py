script_details = ("keras_nn_score.py",0.1)

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
    df = pd.read_csv("~/Datasets/iris.csv")
    # specify predictors and target
    fields = ["sepal_length","sepal_width","petal_length","petal_width"]
    target = "species"
    backend = 'theano'
    modelpath = "/tmp/dnn.model"
    modelmetadata_path = "/tmp/dnn.metadata"

else:
    import spss.pyspark.runtime
    ascontext = spss.pyspark.runtime.getContext()
    sc = ascontext.getSparkContext()
    sqlCtx = ascontext.getSparkSQLContext()
    df = ascontext.getSparkInputData().toPandas()
    target = '%%target%%'
    backend = '%%backend%%'
    fields =  map(lambda x: x.strip(),"%%fields%%".split(","))
    schema = ascontext.getSparkInputSchema()

prediction_field = "$R-" + target
probability_field = "$RP-" + target

if ascontext:
    from pyspark.sql.types import StructField, StructType, StringType, FloatType
    added_fields = []
    added_fields.append(StructField(prediction_field, StringType(), nullable=True))
    added_fields.append(StructField(probability_field, FloatType(), nullable=True))
    output_schema = StructType(schema.fields + added_fields)

    ascontext.setSparkOutputSchema(output_schema)
    if ascontext.isComputeDataModelOnly():
        sys.exit(0)
    else:
        modelpath = ascontext.getModelContentToPath("model")
        model_metadata = json.loads(ascontext.getModelContentToString("model.metadata"))
else:
    model_metadata = json.loads(open(modelmetadata_path,"r").read())

target_values = model_metadata["target_values"]

dataarr=np.array(df[fields])

score_model = load_model(os.path.join(modelpath,"model"))

result = score_model.predict(dataarr)

# bring predictions into dataframe
df[prediction_field] = np.argmax(result,axis=1)
df[probability_field] = np.max(result,axis=1)
# for i in range(0,len(target_values)):
#    df["$RP-"+target_values[i]] = result[:,i]

# recode predictions to original class names
df[prediction_field] = df.apply(lambda row:target_values[row[prediction_field]],axis=1).astype(str)

if ascontext:
    outdf = sqlCtx.createDataFrame(df)
    ascontext.setSparkOutputData(outdf)
else:
    print(str(df))





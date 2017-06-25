# NN_with_Keras

An extension node for experimenting with Deep Neural Network models in IBM SPSS Modeler using Keras

Learn more about Keras [from the Keras Documentation][4]

![Stream](https://raw.githubusercontent.com/IBMPredictiveAnalytics/NN_with_Keras/master/screenshots/stream.png)

---
Requirements
----
-	SPSS Modeler v18.0 or later
-   [Anaconda distribution of Python 2.7 (recommended)](https://www.continuum.io/downloads) or [Python 2.7](https://www.python.org/downloads)
-   [Keras for Python 2.7 (+Theano or +Tensorflow)](https://keras.io/)

More information here: [IBM Predictive Extensions][2]

---
Installation Instructions
----

#### Initial one-time set-up for PySpark Extensions

If using v18.0 of SPSS Modeler, navigate to the options.cfg file (Windows default path: C:\Program Files\IBM\SPSS\Modeler\18.0\config).  Open this file in a text editor and edit the setting eas_pyspark_python_path near the bottom of the document to read:

  eas_pyspark_python_path, "*C:/Users/IBM_ADMIN/Python_27/python.exe*"

  -   The italicized path should be replaced with the path to your python.exe from your Python installation.

#### Installing Keras

Your Python 2.7 installation must [have the Keras library and its dependencies, including Theano or Tensorflow installed](http://keras.io) for this extension to work.  Please follow the instructions provided by the Keras documentation.

#### Modeler Extension Installation
  1.	[Save the .mpe file][3] to your computer
  2.	In Modeler, click the Extensions menu, then click Install Local Extension Bundle
  3.	Navigate to where the .mpe was saved and click open
  4.	The extension will install and a pop-up will show what palette it was installed

---
Image Classification Example
----

This example is based on the mnist dataset (image classification) but using a small sample of the data to keep the repo and download size manageable.  The small sample will limit the performance that the classifier will achieve.

[Download the example stream][5]

[Download the example data (train)][6]

[Download the example data (test)][7]

---
License
----

[Apache 2.0][1]

---
Contributors
----
- Niall McCarroll - ([www.mccarroll.net](http://www.mccarroll.net/))


[1]:http://www.apache.org/licenses/LICENSE-2.0.html
[2]:https://developer.ibm.com/predictiveanalytics/downloads
[3]:https://raw.githubusercontent.com/IBMPredictiveAnalytics/NN_with_Keras/master/NN_with_Keras.mpe
[4]:http://keras.io
[5]:https://raw.githubusercontent.com/IBMPredictiveAnalytics/NN_with_Keras/master/examples/mnist.str
[6]:https://raw.githubusercontent.com/IBMPredictiveAnalytics/NN_with_Keras/master/examples/mnist_smallsample_train.csv
[7]:https://raw.githubusercontent.com/IBMPredictiveAnalytics/NN_with_Keras/master/examples/mnist_smallsample_test.csv

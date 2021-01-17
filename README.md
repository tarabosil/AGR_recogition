### AGR_recogition

In this project we use convolutional neural network to predict age, gender and race form face images. We use UTKFace database which consists of more than 20k face images.

To run the project you must firstly run the following command to install all the required requirements.

```buildoutcfg
pip install -r requirements.txt
```

In the **config.py** you can set the parameters for trainig and what you want to predict. The default value of parameter **do_training**
is set to False, which means that you will not train the model, but used alreday trained one. If you want to train the model then you set this parameter to True.
Second parameter is **type**. With this parameter you will choose which prediction you want to do. YOu can choose between age, gender and race.

To get the results (accuracy score, classification matrix, confusion matrix and ROC_AUC curve) you must run ``python main.py``.


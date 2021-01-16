### AGR_recogition

In this project we use convolutional neural network to predict age, gender and race form face images. 

To run the project you must firsly run the following command to install all the required requirements.

```buildoutcfg
pip install -r requirements.txt
```


In the main file (main.py) with parameter **type** you can choose which model you want to load it. You can choose between three different types: age, gender or race. If you want to load gender model, you will write **type = 'gender'**, the same holds for age and race. 
There is also **do_training** parameter. If it is set to False, then you will not train the model, but if you will set this parameter to True then you will train the model.

To get the results (accuracy score, ROC curve and AUC curve) you sholud run `` python main.py ``.

from itertools import cycle

import keras
import numpy as np
import cv2
import os
import seaborn as sns
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils.vis_utils import plot_model
from keras.utils.np_utils import to_categorical
from sklearn import metrics
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, Flatten
from evaluation import Evaluator
import config

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

evaluator = Evaluator()


def create_model(type):
    """ Create the model """
    model = Sequential()

    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(config.n, config.n, 3)))
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(config.num_classes[type], activation='softmax'))
    # model.summary()

    return model


def train_model(type):
    """ Train the model """
    model = create_model(type)

    checkpoint = ModelCheckpoint(f"models/{type}_model.h5",
                                 monitor="val_loss",
                                 mode="min",
                                 save_best_only=True,
                                 verbose=1)

    earlystop = EarlyStopping(monitor="val_loss",
                              min_delta=0,
                              patience=4,
                              verbose=1,
                              restore_best_weights=True)

    callbacks = [earlystop, checkpoint]

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        callbacks=callbacks,
                        epochs=10,
                        batch_size=64)
    # model.save(f"models/{type}_model.h5")

    sns.set()
    fig = plt.figure(0, (12, 4))

    ax = plt.subplot(1, 2, 1)
    sns.lineplot(history.epoch, history.history['accuracy'], label='train')
    sns.lineplot(history.epoch, history.history['val_accuracy'], label='validation')
    plt.title('Accuracy')
    plt.tight_layout()

    ax = plt.subplot(1, 2, 2)
    sns.lineplot(history.epoch, history.history['loss'], label='train')
    sns.lineplot(history.epoch, history.history['val_loss'], label='validation')
    plt.title('Loss')
    plt.tight_layout()

    plt.savefig(f'images/{type}_epoch_history1.png')
    # plt.show()


def preprocess_data(type):
    """ Preprocess the data """
    filenames = os.listdir(config.path)

    genders, ages, races, X_data = [], [], [], []

    for file in filenames:
        genders.append(file.split('_')[1])
        ages.append(file.split('_')[0])
        races.append(file.split('_')[2])
        image = cv2.imread(os.path.join(config.path, file))
        X_data.append(cv2.resize(image, (config.n, config.n)))

    print(f"Number of men: {genders.count('0')}, number of women: {genders.count('1')}")
    evaluator.plot_gender_barchart(genders)

    print(f"White: {races.count('0')}, black: {races.count('1')}, asian: {races.count('2')}, indian: {races.count('3')}, others: {races.count('4')}")
    evaluator.plot_race_barchart(races)

    y_data = []
    if type == 'gender':
        for i in genders:
            y_data.append(int(i))
    elif type == 'race':
        for i in races:
            y_data.append(int(i))
    elif type == 'age':
        for i in ages:
            if int(i) < 15:
                y_data.append(0)
            elif 15 <= int(i) < 30:
                y_data.append(1)
            elif 30 <= int(i) < 40:
                y_data.append(2)
            elif 40 <= int(i) < 60:
                y_data.append(3)
            elif 60 <= int(i) < 80:
                y_data.append(4)
            elif 80 <= int(i):
                y_data.append(5)
        print(f"Children: {y_data.count(0)}, Youth: {y_data.count(1)}, Adults: {y_data.count(2)}, Middle age: {y_data.count(3)},"
              f" Old: {y_data.count(4)}, Very old: {y_data.count(5)}")
        evaluator.plot_age_barchart(y_data)
    y_data = np.array(y_data)
    y_data = np.expand_dims(y_data, axis=-1)
    print(y_data.shape)

    X_data = np.array(X_data)
    print(X_data.shape)

    y_data = to_categorical(y_data)

    return X_data, y_data


def split_data(X_data, y_data):
    """ Split the data into train, val and test set """
    X_temp, X_test, y_temp, y_test = train_test_split(X_data, y_data,
                                                    test_size=0.20,
                                                    shuffle=True,
                                                    random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp,
                                                        test_size=0.20,
                                                        random_state=1)

    return X_train, y_train, X_test, y_test, X_val, y_val


def evaluate(model, type):
    """ Evaluate the model """
    preds = model.predict_classes(X_test)
    y_true = np.argmax(y_test, axis=1)
    test_accuracy = accuracy_score(y_true, preds)
    print(f"Accuracy for {type} prediction is: {test_accuracy}")

    matrix = metrics.confusion_matrix(y_true, preds)
    print(matrix)
    print(metrics.classification_report(y_true, preds))

    if type == 'gender':
        preds_multi = model.predict_proba(X_test)

        # print(f"F1-score: {metrics.f1_score(y_true, preds)}")
        # print(f"Precision: {metrics.precision_score(y_true, preds)}")
        # print(f"Recall: {metrics.recall_score(y_true, preds)}")

        evaluator.multi_roc_curve(y_test, preds_multi, type, config.num_classes, config.gender_classes)
    elif type == 'race':
        preds_multi = model.predict_proba(X_test)

        evaluator.multi_roc_curve(y_test, preds_multi, type, config.num_classes, config.race_classes)
    elif type == 'age':
        preds_multi = model.predict_proba(X_test)
        evaluator.multi_roc_curve(y_test, preds_multi, type, config.num_classes, config.age_classes)

    return y_true, preds


if __name__ == '__main__':

    X_data, y_data = preprocess_data(config.type)
    X_train, y_train, X_test, y_test, X_val, y_val = split_data(X_data, y_data)

    # train model
    if config.do_training:
        train_model(config.type)

    # load model
    model = create_model(config.type)
    model.load_weights(f'models/{config.type}_model.h5')
    y_true, preds = evaluate(model, config.type)
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

import os


#read data from csv, then change into [2,3] index
data1 = pd.read_csv("dataset.csv")
data = data1.sample(frac=0.2)
# test_data = pd.read_csv("testgasdone")
# predict_data = pd.read_csv("0.159r")
# a_predict_data = pd.read_csv("0.1510r")

features, labels = data.iloc[:,0:40], data.iloc[:,40]
# t_features, t_labels = test_data.iloc[:, 0:30], test_data.iloc[:, 30]
# p_features = predict_data.iloc[:, 0:21]
# a_p_features = a_predict_data.iloc[:, 0:21]


F, ls = tf.constant(features.values), tf.constant(labels.values)
# t_F, t_ls = tf.constant(t_features.values), tf.constant(t_labels.values)
# p_F = tf.constant(p_features.values)
# a_p_F = tf.constant(a_p_features.values)

F1 = tf.reshape(F, (-1, 10, 4, 1))
# t_F1 = tf.reshape(t_F, (-1, 7, 3, 1))
# p_F1 = tf.reshape(p_F, (-1, 7, 3, 1))
# a_p_F1 = tf.reshape(a_p_F, (-1, 7, 3, 1))

#print(data)
#print(F1)
#print(ls)

# sequential model

model = keras.Sequential(
    [   layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same'),
        layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same'),
        layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same'),
        layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same'),
        layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same'),
        # layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same'),
        # layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPool2D(pool_size=(2, 2), padding='same'),
        layers.Dropout(rate=0.25),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation='relu'),
        layers.Dense(10, activation='relu'),
        layers.Dense(2, activation='softmax'),
    ]
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=5e-05, beta_1=0.9, beta_2=0.999, epsilon=1e-10),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

#train
model.fit(F1, ls, batch_size=2048, validation_split=0.2, shuffle=True, epochs=10)

# model.evaluate(t_F1, t_ls, verbose=2)

model.save('saved_model/mix.ver1.0.try')

# prediction = model.predict(p_F1)
# y_test_pred = np.argmax(prediction, axis=1)
# y_test_output = y_test_pred.reshape(-1, 1)
# pd.DataFrame(y_test_output).to_csv('test_result1')
#
# a_prediction = model.predict(a_p_F1)
# a_y_test_pred = np.argmax(a_prediction, axis=1)
# a_y_test_output = a_y_test_pred.reshape(-1, 1)
# pd.DataFrame(a_y_test_output).to_csv('test_result2')

# print(y_test_pred)
# print(prediction)
# with open('testresult.csv', 'w') as f:
#     f.write(y_test_pred)

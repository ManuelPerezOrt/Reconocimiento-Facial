import tensorflow as tf
from tensorflow.keras import layers, regularizers
import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.models import load_model
from keras import models, layers, optimizers

import wandb
from wandb.keras import WandbCallback
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

wandb.login()
wandb.init(project="ModeloEntrenado")
wandb.config.learning_rate = 0.001
wandb.config.epochs = 3
wandb.config.batch_size = 100
wandb.config.loss = 'binary_crossentropy'
wandb.config.optimizer = 'Adam'

np.set_printoptions(precision=4)

# Eliminar el dobel espacio entre algunos datos de la tabla
with open('list_attr_celeba.txt', 'r') as f:
    print("skipping : " + f.readline())
    print("skipping headers : " + f.readline())
    with open('attr_celeba_prepared.txt' , 'w') as newf:
        for line in f:
            new_line = ' '.join(line.split())
            new_line = new_line.replace('-1', '0')
            newf.write(new_line)
            newf.write('\n')

df = pd.read_csv('attr_celeba_prepared.txt', sep=' ', header = None)

files = tf.data.Dataset.from_tensor_slices(df[0])
attributes = tf.data.Dataset.from_tensor_slices(df.iloc[:, 
                                                        [i for i in range(1, df.shape[1])
                                                         if i in [3,19,20,21,22,32,37]]].to_numpy())

data = tf.data.Dataset.zip((files, attributes))
print(data)
                 
path_to_images = 'RecFas/img_align_celeba/'
def process_file(file_name, attributes):
    image = tf.io.read_file(path_to_images + file_name)
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.image.resize(image, [60, 60])  # Reducir la dimensionalidad
    image /= 255.0  
    return image, attributes

labeled_images = data.map(process_file)

print(labeled_images)
inputs = tf.keras.Input(shape=(60, 60, 1))

def block(x, base_filter=33, pooling=True):
    x = layers.Conv2D(base_filter, 3, activation="relu", padding="same",
                       kernel_regularizer=regularizers.L1L2(l1=1e-4, l2=1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(2*base_filter, 3, activation="relu", padding="same",
                       kernel_regularizer=regularizers.L1L2(l1=1e-4, l2=1e-4))(x)
    x = layers.BatchNormalization()(x)
    if pooling:
        x = layers.MaxPooling2D(3, padding="same")(x)
    return x

x = block(inputs, base_filter=33)
x = block(x, base_filter=63)
x = block(x, base_filter=129)

x = layers.Flatten()(x)
x = layers.Dense(256, activation="relu",
                  kernel_regularizer=regularizers.L1L2(l1=1e-4, l2=1e-4))(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(128, activation="relu",
                  kernel_regularizer=regularizers.L1L2(l1=1e-4, l2=1e-4))(x)
x = layers.BatchNormalization()(x)
outputs = layers.Dense(7, activation="sigmoid")(x)

model = tf.keras.Model(inputs, outputs)
# Crear el modelo
RecFasModel = tf.keras.Model(inputs=inputs, outputs=outputs, name="RecFasModel")
# Imprimir el resumen del modelo
RecFasModel.summary()

RecFasModel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                     loss='binary_crossentropy', metrics=['binary_accuracy'])
labeled_images = labeled_images.batch(100)
RecFasModel.fit(labeled_images, epochs=3,callbacks=[WandbMetricsLogger(log_freq=5),
                                                     WandbModelCheckpoint("models")])

# Guardar el modelo en disco
RecFasModel.save('RecFasModel.h5')






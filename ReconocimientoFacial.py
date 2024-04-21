# Importando las bibliotecas necesarias
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras import models, layers, optimizers, regularizers
import os
import time
import shutil
from tensorflow.keras.models import load_model

import wandb
from wandb.keras import WandbCallback
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

wandb.login()
wandb.init(project="ModeloRec.Facial")
wandb.config.learning_rate = 2e-5
wandb.config.epochs = 45
wandb.config.batch_size = 100
wandb.config.loss = 'binary_crossentropy'
wandb.config.optimizer = 'RMSprop'

# Cargar el modelo previamente entrenado
RecFasModel = load_model('RecFasModel.h5')
print("Número total de capas en el modelo: ", len(RecFasModel.layers))

# Directorio donde están todas las imágenes
dir_imagenes = 'RecFas/img_align_celeba'

# Barrer todas las imágenes en el directorio original
#for nombre_imagen in os.listdir(dir_imagenes):    
    # Copiar la imagen al directorio correspondiente
#    if nombre_imagen < '162771':
#        shutil.copy(os.path.join(dir_imagenes, nombre_imagen), 'RecFas/train/no_yo')
#    else:
#        shutil.copy(os.path.join(dir_imagenes, nombre_imagen), 'RecFas/test/no_yo')

# Definir los generadores de datos para entrenamiento y validación
train_datagen = ImageDataGenerator(rescale=1./255,zoom_range=0.2, rotation_range = 5, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

# Asumiendo que las imágenes están en carpetas 'entrenamiento' y 'validacion'
train_generator = train_datagen.flow_from_directory(
        'RecFas/train',  # Directorio con las imágenes de entrenamiento
        target_size=(60, 60),  # Todas las imágenes se redimensionarán a 60x60 para coincidir con el modelo anterior
        batch_size=20,
        class_mode='binary')  # Como estamos usando binary_crossentropy loss, necesitamos etiquetas binarias

validation_generator = val_datagen.flow_from_directory(
        'RecFas/test',
        target_size=(60, 60),
        batch_size=20,
        class_mode='binary')

# 2. Construir un modelo con las capas convolucionales del modelo entrenado pero quitándole el clasificador
sufijo_único = str(time.time())

x = RecFasModel.layers[21].output
x = layers.Flatten(name="flatten_" + sufijo_único)(x)
x = layers.Dense(300, activation='relu',kernel_regularizer=regularizers.L1L2(l1=1e-3, l2=1e-3), name="dense_1_" + sufijo_único)(x)
x = layers.BatchNormalization(name="batch_normalization_1_" + sufijo_único)(x)
x = layers.Dropout(0.5, name="dropout_1_" + sufijo_único)(x)

x = layers.Dense(200, activation='relu',kernel_regularizer=regularizers.L1L2(l1=1e-3, l2=1e-3), name="dense_2_" + sufijo_único)(x)
x = layers.BatchNormalization(name="batch_normalization_2_" + sufijo_único)(x)
x = layers.Dropout(0.5, name="dropout_2_" + sufijo_único)(x)

x = layers.Dense(100, activation='relu',kernel_regularizer=regularizers.L1L2(l1=1e-3, l2=1e-3), name="dense_3_" + sufijo_único)(x)
x = layers.BatchNormalization(name="batch_normalization_3_" + sufijo_único)(x)
x = layers.Dropout(0.5, name="dropout_3_" + sufijo_único)(x)

x = layers.Dense(20, activation='relu',kernel_regularizer=regularizers.L1L2(l1=1e-3, l2=1e-3), name="dense_4_" + sufijo_único)(x)
x = layers.BatchNormalization(name="batch_normalization_4_" + sufijo_único)(x)
x = layers.Dropout(0.1, name="dropout_4_" + sufijo_único)(x)

outputs = layers.Dense(1, activation="sigmoid", kernel_regularizer=regularizers.L1L2(l1=1e-3, l2=1e-3), name="dense_output_" + sufijo_único)(x)

model = tf.keras.Model(inputs=RecFasModel.input, outputs=outputs, name="Model")

for layer in model.layers[:21]:
# 4. Congelar los pesos de la parte pre-entrenada
 layer.trainable = False
model.summary()

# 5. Entrenar este modelo
# Aquí se asume que ya tienes tus datos de entrenamiento y validación en `train_data`, `train_labels`, `val_data`, `val_labels`
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(learning_rate=2e-5), metrics=['acc', 'mse'])
history = model.fit(train_generator, steps_per_epoch=100, epochs=45, validation_data=validation_generator, validation_steps=50,callbacks=[WandbMetricsLogger(log_freq=5), WandbModelCheckpoint("models")])

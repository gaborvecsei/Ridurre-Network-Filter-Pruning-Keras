import shutil
from pathlib import Path

import numpy as np
from keras import datasets, utils, callbacks, optimizers, losses
from keras.preprocessing.image import ImageDataGenerator

from cifar_10_resnet import resnet
from filter_pruning import kmeans_pruning
from model_complexity import graph_complexity

TRAIN_LOGS_FOLDER_PATH = Path("./train_logs")
if TRAIN_LOGS_FOLDER_PATH.is_dir():
    shutil.rmtree(str(TRAIN_LOGS_FOLDER_PATH))
TRAIN_LOGS_FOLDER_PATH.mkdir()

# Creating ResNet50 model
model = resnet.resnet_v1((32, 32, 3), 20, 10)


def compile_model(my_model):
    my_model.compile(optimizer=optimizers.Adam(lr=0.001), loss=losses.categorical_crossentropy, metrics=["accuracy"])


compile_model(model)

# Loading data
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# Data Transform
x_train = x_train.astype(np.float32) / 255.0
y_train = utils.to_categorical(y_train)
x_train_mean = np.mean(x_train, axis=0)
x_train -= x_train_mean

x_test = x_test.astype(np.float32) / 255.0
y_test = utils.to_categorical(y_test)
x_test -= x_train_mean

x_train = x_train[:100]
y_train = y_train[:100]

print("Train shape: X {0}, y: {1}".format(x_train.shape, y_train.shape))
print("Test shape: X {0}, y: {1}".format(x_test.shape, y_test.shape))

# Data Augmentation with Data Generator
data_generator = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=20)

# Create callbacks
tensorboard_callback = callbacks.TensorBoard(log_dir=str(TRAIN_LOGS_FOLDER_PATH))
model_complexity_param = graph_complexity.ModelParametersCallback(TRAIN_LOGS_FOLDER_PATH, verbose=1)
model_checkpoint_callback = callbacks.ModelCheckpoint(str(TRAIN_LOGS_FOLDER_PATH) + "/model_{epoch:02d}.h5",
                                                      save_best_only=False,
                                                      save_weights_only=False,
                                                      verbose=1)
callbacks = [tensorboard_callback, model_complexity_param, model_checkpoint_callback]

# Train the model
EPOCHS = 3
BATCH_SIZE = 64
STEPS_PER_EPOCH = len(x_train) // BATCH_SIZE

model.fit_generator(data_generator.flow(x_train, y_train, BATCH_SIZE), epochs=EPOCHS, validation_data=(x_test, y_test),
                    callbacks=callbacks, steps_per_epoch=STEPS_PER_EPOCH)


# Prune the model
def finetune_model(my_model, initial_epoch, finetune_epochs):
    my_model.fit_generator(data_generator.flow(x_train, y_train, BATCH_SIZE), epochs=finetune_epochs,
                           validation_data=(x_test, y_test), callbacks=callbacks, initial_epoch=initial_epoch,
                           verbose=1, steps_per_epoch=STEPS_PER_EPOCH)


pruning = kmeans_pruning.KMeansFilterPruning(0.9, compile_model, finetune_model, 1, EPOCHS)
pruning.run_pruning(model)

# Train again for a reasonable number of epochs
model.fit_generator(data_generator.flow(x_train, y_train, BATCH_SIZE), epochs=10, validation_data=(x_test, y_test),
                    callbacks=callbacks, steps_per_epoch=STEPS_PER_EPOCH)

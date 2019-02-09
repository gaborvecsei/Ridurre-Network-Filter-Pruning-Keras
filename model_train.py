from pathlib import Path
import shutil
from keras import backend as K
from cifar_10_resnet import resnet
from keras import datasets, utils, callbacks, optimizers, losses
import numpy as np
from filter_pruning import graph_complexity
from filter_pruning import kmeans_pruning

TRAIN_LOGS_FOLDER_PATH = Path("./train_logs")
if TRAIN_LOGS_FOLDER_PATH.is_dir():
    shutil.rmtree(str(TRAIN_LOGS_FOLDER_PATH))
TRAIN_LOGS_FOLDER_PATH.mkdir()

# Creating ResNet50 model
model = resnet.resnet_v1((32, 32, 3), 20, 10)


def compile_model(model):
    model.compile(optimizer=optimizers.Adam(lr=0.001), loss=losses.categorical_crossentropy, metrics=["accuracy"])


compile_model(model)

# Loading data
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# TODO: this is only for testing
nb_data_points = 100
x_train = x_train[:nb_data_points]
y_train = y_train[:nb_data_points]
x_test = x_test[:nb_data_points]
y_test = y_test[:nb_data_points]

# Data Transform
x_train = x_train.astype(np.float32) / 255.0
y_train = utils.to_categorical(y_train)
x_train_mean = np.mean(x_train, axis=0)
x_train -= x_train_mean

x_test = x_test.astype(np.float32) / 255.0
y_test = utils.to_categorical(y_test)
x_test -= x_train_mean

# Create callbacks
tensorboard_callback = callbacks.TensorBoard(log_dir=str(TRAIN_LOGS_FOLDER_PATH))
model_complexity_param = graph_complexity.ModelComplexityCallback(TRAIN_LOGS_FOLDER_PATH, K.get_session())
model_checkpoint_callback = callbacks.ModelCheckpoint(str(TRAIN_LOGS_FOLDER_PATH) + "/model_{epoch}.h5",
                                                      save_best_only=False,
                                                      save_weights_only=False,
                                                      verbose=1)
callbacks = [tensorboard_callback, model_complexity_param, model_checkpoint_callback]

# Train the model
model.fit(x_train, y_train, 32, epochs=3, validation_data=(x_test, y_test), callbacks=callbacks)


# Prune the model
def finetune_model(model, initial_epoch):
    model.fit(x_train, y_train, 32, epochs=1, validation_data=(x_test, y_test), callbacks=callbacks,
              initial_epoch=initial_epoch)


pruning = kmeans_pruning.KMeansFilterPruning(0.9, compile_model, finetune_model, 1, 3)
pruning.run_pruning(model)

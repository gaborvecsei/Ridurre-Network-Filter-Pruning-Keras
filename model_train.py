from keras import backend as K
from keras.applications import resnet50
from keras import datasets, utils, callbacks, optimizers, losses
import numpy as np
from filter_pruning import kmeans_pruning

from filter_pruning import model_complexity_calculation

# Creating ResNet50 model
model = resnet50.ResNet50(input_shape=(32, 32, 3), classes=10, include_top=False, weights="imagenet")
model.compile(optimizer=optimizers.Adam(lr=0.001), loss=losses.categorical_crossentropy)

# Loading data
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# Data Transform
x_train = x_train.astype(np.float32) / 255.0
y_train = utils.to_categorical(y_train)

x_test = x_test.astype(np.float32) / 255.0
y_test = utils.to_categorical(y_test)

# Define callbacks for the training
TRAIN_LOGS_FOLDER_PATH = "./train_logs"

kmeans_pruning_callback = kmeans_pruning.KMeansFilterPruning(0, 1, 0.9)
tensorboard_callback = callbacks.TensorBoard(log_dir=TRAIN_LOGS_FOLDER_PATH)
model_complexity_param = model_complexity_calculation.ModelComplexityCallback(TRAIN_LOGS_FOLDER_PATH, K.get_session())

callbacks = [kmeans_pruning_callback, tensorboard_callback, model_complexity_param]

# Train the model
model.fit(x_train, y_train, 32, epochs=100, validation_data=(x_test, y_test), callbacks=callbacks)

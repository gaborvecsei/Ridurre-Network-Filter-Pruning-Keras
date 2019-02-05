from keras import backend as K
from keras.applications import resnet50
from keras import datasets, utils, callbacks, optimizers, losses
import numpy as np
from filter_pruning import kmeans_pruning

from filter_pruning.model_complexity_calculation import calculate_flops_and_parameters

model = resnet50.ResNet50(include_top=False, weights=None)
model.compile(optimizer=optimizers.Adam(lr=0.001), loss=losses.categorical_crossentropy)

flops, nb_params = calculate_flops_and_parameters(K.get_session())
print("Initial FLOPS: {0:,}, Parameters: {1}".format(flops, nb_params))

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

x_train = x_train.astype(np.float32) / 255.0
y_train = utils.to_categorical(y_train)

x_test = x_test.astype(np.float32) / 255.0
y_test = utils.to_categorical(y_test)

callbacks = [kmeans_pruning.KMeansFilterPruning(10, 5, 0.9), callbacks.TensorBoard(log_dir="train_logs")]

model.fit(x_train, y_train, 32, epochs=100, validation_data=(x_test, y_test), callbacks=callbacks)

flops, nb_params = calculate_flops_and_parameters(K.get_session())
print("Finished FLOPS: {0:,}, Parameters: {1}".format(flops, nb_params))

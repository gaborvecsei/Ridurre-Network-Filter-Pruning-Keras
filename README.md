# Filter Pruning in Deep Convolutional Networks

TODO...

## Usage

Check out the [`model_pruning_example.py` file](/model_pruning_example.py) for a simple but extensive tutorial

You need to define 2 callbacks for the pruning

- **Model compile function**
    - 1 argument:
        - `model` which is a `keras.models.Model`
    - This should define how to compile the model
    ```python
    def compile_model(my_model):
    my_model.compile(optimizer=optimizers.Adam(lr=0.001),
                     loss=losses.categorical_crossentropy,
                     metrics=["accuracy"])
    ```
- **Finetune function**
    - 3 arguments:
        - `model` which is a `keras.models.Model`
        - `initial_epoch` which is an `int`: This defines the initial epoch state for the model fitting.
        For example it is 12 if we trained the model for 12 epochs before this function was called
        - `finetune_epochs` which is an `int`: Defines how much should we train after a pruning.
    - This should define how to finetune out model
    ```python
    def finetune_model(my_model, initial_epoch, finetune_epochs):
    my_model.fit(x_train,
                 y_train,
                 32,
                 epochs=finetune_epochs,
                 validation_data=(x_test, y_test),
                 callbacks=callbacks,
                 initial_epoch=initial_epoch,
                 verbose=1)
    ```

## Papers

- [Demystifying Neural Network Filter Pruning](https://openreview.net/pdf?id=rJffBWBtoX)
	- [Review of the paper](https://openreview.net/forum?id=rJffBWBtoX)
- [Filter Level Pruning Based on Similar Feature Extraction for
Convolutional Neural Networks](https://www.jstage.jst.go.jp/article/transinf/E101.D/4/E101.D_2017EDL8248/_pdf)

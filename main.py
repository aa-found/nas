import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import Dense, Flatten

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from tensorflow.keras import backend as k
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard


from tensorflow.keras.optimizers import Adam

import datetime
import random


#__________Tempo var ____________  go back to layer_count()
num_layers = 5

f_kernel_size = random.choice([(3, 3), (5, 5)])
f_pool_size  = random.choice([(2, 2), (3, 3)])
f_strides = random.choice([(1, 1), (2, 2)])


# Create a new log directory (to avoid mixing logs from different runs)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)


available_layer_types = ['Dense', 'Dropout', 'Conv2D', 'MaxPooling2D']

#global vars
num_architectures = 10  # Modify this number as needed
epo = 5
bs = 32
min_layers = 2
max_layers = 7
current_layers = min_layers


#layer_types = ['Dense', 'Conv2D', 'MaxPooling2D']
#layer_probabilities = [0.3, 0.3, 0.4]

input_shape = (28, 28, 1)  # Example input shape, adjust according to your data



# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Normalize and reshape the input data
x_train = x_train / 255.0
x_train = np.expand_dims(x_train, -1)  # Add channel dimension
input_shape = x_train.shape[1:]  # Shape of MNIST images

# Assuming y_train and y_test are your original labels
y_train_encoded = to_categorical(y_train, num_classes=10)
y_test_encoded = to_categorical(y_test, num_classes=10)


model_checkpoint_callback = ModelCheckpoint(
    filepath='save/model_{epoch}.keras',  # Change file extension to .keras
    save_best_only=True,
    monitor='val_loss',
    mode='min',
    save_freq='epoch'
)

initial_learning_rate = 0.01  # Start with this learning rate
decay_steps = 1000            # After how many steps to apply decay
decay_rate = 0.96             # The rate of decay

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    staircase=True)  # Set to False for continuous decay

# Use this learning rate schedule in the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)



def generate_layer_count():
    global current_layers
    if current_layers < max_layers:
        current_layers += 1  # Increment the layer count
    return current_layers


# Randomly select a layer type based on predefined probabilities
def select_layer_type():
    return np.random.choice(layer_types, p=layer_probabilities)

def generate_layer_parameters(layer_type):
    if layer_type == 'Dense':
        units = np.random.randint(32, 512)  # Random number of neurons
        activation = np.random.choice(['relu', 'tanh', 'sigmoid'])
        return {'units': units, 'activation': activation}

    elif layer_type == 'Dropout':
        rate = np.random.uniform(0.2, 0.5)
        return {'rate': rate}

    elif layer_type == 'Conv2D':
        filters = np.random.randint(16, 128)  # Random number of filters
        kernel_size = f_kernel_size
        strides = f_strides
        activation = np.random.choice(['relu', 'tanh'])
        return {'filters': filters, 'kernel_size': kernel_size, 'strides': strides, 'activation': activation}

    elif layer_type == 'MaxPooling2D':
        pool_size = f_pool_size
        return {'pool_size': pool_size}

    else:
        raise ValueError(f"Unsupported layer type: {layer_type}")



def generate_layers(num_layers, available_layer_types):
    generated_layers = []
    for _ in range(num_layers):
        layer_type = np.random.choice(available_layer_types)
        parameters = generate_layer_parameters(layer_type)
        generated_layers.append((layer_type, parameters))
    return generated_layers


def build_model(input_shape, generated_layers, output_units=10):
    model = Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    
    for layer_type, params in generated_layers:
        try:
            # Create a tentative layer to compute its output shape
            if layer_type in ['Conv2D', 'MaxPooling2D']:
                params['padding'] = 'same'  # Use 'same' padding
                layer = getattr(tf.keras.layers, layer_type)(**params, input_shape=model.output_shape[1:])
            else:
                layer = getattr(tf.keras.layers, layer_type)(**params)

            # Compute the tentative output shape
            tentative_output_shape = layer.compute_output_shape(model.output_shape)

            # Check if the output shape dimensions are valid
            if all(dim is None or dim > 0 for dim in tentative_output_shape):
                model.add(layer)
        except ValueError:
            # Skip this layer if it leads to invalid output shape
            continue

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=output_units, activation='softmax'))

    loss = 'categorical_crossentropy'

    return model



def train_model(model, x_train, y_train, x_test, y_test, epochs, batch_size, optimizer, loss, tensorboard_callback):
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=epo, batch_size=batch_size, validation_data=(x_test, y_test), callbacks=[tensorboard_callback])
    return history


def display_architectures(num_architectures):
    for i in range(num_architectures):
        print(f"\nArchitecture {i + 1}:\n{'=' * 65}")
        generated_layers = generate_layers(num_layers, available_layer_types)
        model = build_model(input_shape, generated_layers)
        model.summary()

        # Create a unique log directory for each architecture
        log_dir = f"logs/architecture_{i+1}/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False)

        # Train the model with its respective TensorBoard callback
        history = train_model(model, x_train, y_train_encoded, x_test, y_test_encoded, epo, bs, optimizer, 'categorical_crossentropy', tensorboard_callback)

        
       

#generated_layers = generate_layers()
#model = build_model(input_shape, generated_layers

display_architectures(num_architectures)


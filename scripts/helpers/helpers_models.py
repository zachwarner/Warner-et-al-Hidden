#!/usr/bin/python3
# helpers_models.py
# Defines models for fitting to image data
# Zach Warner
# 4 September 2020

##### SETUP #####

### import modules
from keras.models import Sequential, Model
from keras.layers.advanced_activations import ELU
from keras.applications.inception_v3 import InceptionV3
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, ZeroPadding2D, GlobalAveragePooling2D

##### MODELS #####

### Inception V3 transfer learning
def model_inception(variable_name, img_height, img_width, n_channel = 3, n_rlu = 1024, dropout_prop = .5, activation = 'elu', unfreeze = 2):
    # initialize the trained inception V3 model
    base_model = InceptionV3(weights = 'imagenet', include_top = False, input_shape = (img_height, img_width, n_channel))
    base_model.save('/data/base_inception_model_' + variable_name + '.h5')
    # add a global spatial average pooling layer
    last_layer = base_model.output
    last_layer = GlobalAveragePooling2D()(last_layer)
    # normalize the batch
    BatchNormalization()
    # add a fully-connected layer
    last_layer = Dense(n_rlu, activation = activation)(last_layer)
    # add a dropout layer
    last_layer = Dropout(dropout_prop)(last_layer)
    # get the predictions through the sigmoid layer
    out = Dense(1, activation = 'sigmoid', name = 'output_layer')(last_layer)
    # put it together into one model
    network = Model(inputs = base_model.input, outputs = out)
    # freeze all but the final two layers
    freeze = 251 - unfreeze
    for layer in network.layers[:freeze]:
        layer.trainable = True
    for layer in network.layers[freeze:]:
        layer.trainable = True
    # return the model
    return(network)

### Cantu (2019) model
def model_cantu_original(img_height, img_width, n_channel = 3, n_border = 1, n_filter = 32, n_pixel = 3, n_pool = 2, n_rlu = 4096, dropout_prop = .2, activation = 'elu'):
    # initiate a sequential backpropagation network
    network = Sequential()
    # add border to avoid edges being washed away
    network.add(ZeroPadding2D(padding = (n_border, n_border), input_shape = (img_height, img_width, n_channel), data_format = "channels_last"))
    # pass the image through the filters
    network.add(Conv2D(n_filter, (n_pixel, n_pixel)))
    # normalize the batch
    BatchNormalization()
    # add Exponential Linear Unit
    network.add(ELU())
    # add pooling layer
    network.add(MaxPooling2D(pool_size = (n_pool, n_pool)))
    # loop through these steps 5 times, increasing the filter size
    # repeat 1: filter remains the same
    network.add(ZeroPadding2D(padding = (n_border, n_border)))
    network.add(Conv2D(n_filter, (n_pixel, n_pixel)))
    BatchNormalization()
    network.add(ELU())
    network.add(MaxPooling2D(pool_size = (n_pool, n_pool)))
    # repeat 2: filter doubled
    n_filter = n_filter*2
    network.add(ZeroPadding2D(padding = (n_border, n_border)))
    network.add(Conv2D(n_filter, (n_pixel, n_pixel)))
    BatchNormalization()
    network.add(ELU())
    network.add(MaxPooling2D(pool_size = (n_pool, n_pool)))
    # repeat 3: filter still doubled
    network.add(ZeroPadding2D(padding = (n_border, n_border)))
    network.add(Conv2D(n_filter, (n_pixel, n_pixel)))
    BatchNormalization()
    network.add(ELU())
    network.add(MaxPooling2D(pool_size = (n_pool, n_pool)))
    # repeat 4: filter quadrupled
    n_filter = n_filter*2
    network.add(ZeroPadding2D(padding = (n_border, n_border)))
    network.add(Conv2D(n_filter, (n_pixel, n_pixel)))
    BatchNormalization()
    network.add(ELU())
    network.add(MaxPooling2D(pool_size = (n_pool, n_pool)))
    # repeat 5: filter octupled
    n_filter = n_filter*2
    network.add(ZeroPadding2D(padding = (n_border, n_border)))
    network.add(Conv2D(n_filter, (n_pixel, n_pixel)))
    BatchNormalization()
    network.add(ELU())
    network.add(MaxPooling2D(pool_size = (n_pool, n_pool)))
    # add a dropout step
    network.add(Dropout(dropout_prop))
    # flatten the output from a square matrix to a vector
    network.add(Flatten())
    # reshape into RLUs, batch-normalize, and activate
    if n_rlu < 16:
        n_rlu = 16
    network.add(Dense(n_rlu))
    BatchNormalization()
    network.add(Activation(activation))
    # increase the dropout portion then repeat these steps, decrasing the RLUs
    dropout_prop = round(dropout_prop + .1, 2)
    n_rlu = int(n_rlu/8)
    network.add(Dropout(dropout_prop))
    network.add(Dense(n_rlu))
    network.add(Activation(activation))
    # increase the dropout proportion again and repeat
    dropout_prop = round(dropout_prop + .2, 2)
    network.add(Dropout(dropout_prop))
    network.add(Dense(1))
    BatchNormalization()
    # generate the final prediction
    network.add(Activation('sigmoid'))
    # return
    return(network)

### Model assuming easy data difficulty and complexity
def model_easy(img_height, img_width, n_channel = 3, n_border = 1, n_filter = 16, n_pixel = 3, n_pool = 2, n_rlu = 512, dropout_prop = .1, activation = 'elu', batchnorm = True):
    # initiate a sequential backpropagation network
    network = Sequential()
    # Convolution Block 1
    network.add(ZeroPadding2D(padding = (n_border, n_border), input_shape = (img_height, img_width, n_channel), data_format = "channels_last"))
    network.add(Conv2D(n_filter, (n_pixel, n_pixel)))
    if batchnorm:
        BatchNormalization()
    network.add(ELU())
    network.add(MaxPooling2D(pool_size = (n_pool, n_pool)))
    # Convolution Block 2
    network.add(ZeroPadding2D(padding = (n_border, n_border)))
    network.add(Conv2D(n_filter*2, (n_pixel, n_pixel)))
    if batchnorm:
        BatchNormalization()
    network.add(ELU())
    network.add(MaxPooling2D(pool_size = (n_pool, n_pool)))
    # Convolution Block 3
    network.add(ZeroPadding2D(padding = (n_border, n_border)))
    network.add(Conv2D(n_filter*4, (n_pixel, n_pixel)))
    if batchnorm:
        BatchNormalization()
    network.add(ELU())
    network.add(MaxPooling2D(pool_size = (n_pool, n_pool)))
    # Fully Connected 1
    network.add(Dropout(dropout_prop))
    network.add(Flatten())
    network.add(Dense(n_rlu))
    BatchNormalization()
    network.add(Activation(activation))
    # Fully Connected 2
    network.add(Dropout(dropout_prop))
    network.add(Dense(int(n_rlu/4)))
    BatchNormalization()
    network.add(Activation(activation))
    # Final layer
    network.add(Dropout(dropout_prop))
    network.add(Dense(1))
    BatchNormalization()
    # generate the final prediction
    network.add(Activation('sigmoid'))
    # return
    return(network)

### Model assuming an average data complexity and difficulty
def model_average(img_height, img_width, n_channel = 3, n_border = 1, n_filter = 32, n_pixel = 3, n_pool = 2, n_rlu = 1024, dropout_prop = .1, activation = 'elu', batchnorm = True):
    # initiate a sequential backpropagation network
    network = Sequential()
    # Convolution Block 1
    network.add(ZeroPadding2D(padding = (n_border, n_border), input_shape = (img_height, img_width, n_channel), data_format = "channels_last"))
    network.add(Conv2D(n_filter, (n_pixel, n_pixel)))
    if batchnorm:
        BatchNormalization()
    network.add(ELU())
    network.add(MaxPooling2D(pool_size = (n_pool, n_pool)))
    # Convolution Block 2
    network.add(ZeroPadding2D(padding = (n_border, n_border)))
    network.add(Conv2D(n_filter*2, (n_pixel, n_pixel)))
    if batchnorm:
        BatchNormalization()
    network.add(ELU())
    network.add(MaxPooling2D(pool_size = (n_pool, n_pool)))
    # Convolution Block 3
    network.add(ZeroPadding2D(padding = (n_border, n_border)))
    network.add(Conv2D(n_filter*4, (n_pixel, n_pixel)))
    if batchnorm:
        BatchNormalization()
    network.add(ELU())
    network.add(MaxPooling2D(pool_size = (n_pool, n_pool)))
    # Fully Connected 1
    network.add(Dropout(dropout_prop))
    network.add(Flatten())
    network.add(Dense(n_rlu))
    BatchNormalization()
    network.add(Activation(activation))
    # Fully Connected 2
    network.add(Dropout(dropout_prop))
    network.add(Dense(int(n_rlu/2)))
    BatchNormalization()
    network.add(Activation(activation))
    # Final layer
    network.add(Dropout(dropout_prop))
    network.add(Dense(1))
    BatchNormalization()
    # generate the final prediction
    network.add(Activation('sigmoid'))
    # return
    return(network)


### Model assuming 'wide' data
def model_wide(img_height, img_width, n_channel = 3, n_border = 1, n_filter = 32, n_pixel = 3, n_pool = 2, n_rlu = 4096, dropout_prop = .1, activation = 'elu', batchnorm = True):
    # initiate a sequential backpropagation network
    network = Sequential()
    # Convolution Block 1
    network.add(ZeroPadding2D(padding = (n_border, n_border), input_shape = (img_height, img_width, n_channel), data_format = "channels_last"))
    network.add(Conv2D(n_filter, (n_pixel, n_pixel)))
    if batchnorm:
        BatchNormalization()
    network.add(ELU())
    network.add(MaxPooling2D(pool_size = (n_pool, n_pool)))
    # Convolution Block 2
    network.add(ZeroPadding2D(padding = (n_border, n_border)))
    network.add(Conv2D(n_filter*2, (n_pixel, n_pixel)))
    if batchnorm:
        BatchNormalization()
    network.add(ELU())
    network.add(MaxPooling2D(pool_size = (n_pool, n_pool)))
    # Convolution Block 3
    network.add(ZeroPadding2D(padding = (n_border, n_border)))
    network.add(Conv2D(n_filter*4, (n_pixel, n_pixel)))
    if batchnorm:
        BatchNormalization()
    network.add(ELU())
    network.add(MaxPooling2D(pool_size = (n_pool, n_pool)))
    # Convolution Block 4
    network.add(ZeroPadding2D(padding = (n_border, n_border)))
    network.add(Conv2D(n_filter*8, (n_pixel, n_pixel)))
    if batchnorm:
        BatchNormalization()
    network.add(ELU())
    network.add(MaxPooling2D(pool_size = (n_pool, n_pool)))
    # Fully Connected 1
    network.add(Dropout(dropout_prop))
    network.add(Flatten())
    network.add(Dense(n_rlu))
    BatchNormalization()
    network.add(Activation(activation))
    # Fully Connected 2
    network.add(Dropout(dropout_prop))
    network.add(Dense(int(n_rlu/8)))
    BatchNormalization()
    network.add(Activation(activation))
    # Final layer
    network.add(Dropout(dropout_prop))
    network.add(Dense(1))
    BatchNormalization()
    # generate the final prediction
    network.add(Activation('sigmoid'))
    # return
    return(network)

### Model assuming 'deep' data
def model_deep(img_height, img_width, n_channel = 3, n_border = 1, n_filter = 16, n_pixel = 3, n_pool = 2, n_rlu = 2048, dropout_prop = .1, activation = 'elu', batchnorm = True):
    # initiate a sequential backpropagation network
    network = Sequential()
    # Convolution Block 1
    network.add(ZeroPadding2D(padding = (n_border, n_border), input_shape = (img_height, img_width, n_channel), data_format = "channels_last"))
    network.add(Conv2D(n_filter, (n_pixel, n_pixel)))
    if batchnorm:
        BatchNormalization()
    network.add(ELU())
    network.add(MaxPooling2D(pool_size = (n_pool, n_pool)))
    # Convolution Block 2
    network.add(ZeroPadding2D(padding = (n_border, n_border)))
    network.add(Conv2D(n_filter, (n_pixel, n_pixel)))
    if batchnorm:
        BatchNormalization()
    network.add(ELU())
    network.add(MaxPooling2D(pool_size = (n_pool, n_pool)))
    # Convolution Block 3
    network.add(ZeroPadding2D(padding = (n_border, n_border)))
    network.add(Conv2D(n_filter, (n_pixel, n_pixel)))
    if batchnorm:
        BatchNormalization()
    network.add(ELU())
    network.add(MaxPooling2D(pool_size = (n_pool, n_pool)))
    # Convolution Block 4
    network.add(ZeroPadding2D(padding = (n_border, n_border)))
    network.add(Conv2D(n_filter, (n_pixel, n_pixel)))
    if batchnorm:
        BatchNormalization()
    network.add(ELU())
    network.add(MaxPooling2D(pool_size = (n_pool, n_pool)))
    # Convolution Block 5
    network.add(ZeroPadding2D(padding = (n_border, n_border)))
    network.add(Conv2D(n_filter*2, (n_pixel, n_pixel)))
    if batchnorm:
        BatchNormalization()
    network.add(ELU())
    network.add(MaxPooling2D(pool_size = (n_pool, n_pool)))
    # Convolution Block 6
    network.add(ZeroPadding2D(padding = (n_border, n_border)))
    network.add(Conv2D(n_filter*2, (n_pixel, n_pixel)))
    if batchnorm:
        BatchNormalization()
    network.add(ELU())
    network.add(MaxPooling2D(pool_size = (n_pool, n_pool)))
    # Convolution Block 7
    network.add(ZeroPadding2D(padding = (n_border, n_border)))
    network.add(Conv2D(n_filter*4, (n_pixel, n_pixel)))
    if batchnorm:
        BatchNormalization()
    network.add(ELU())
    network.add(MaxPooling2D(pool_size = (n_pool, n_pool)))
    # Fully Connected 1
    network.add(Dropout(dropout_prop))
    network.add(Flatten())
    network.add(Dense(n_rlu))
    BatchNormalization()
    network.add(Activation(activation))
    # Fully Connected 2
    network.add(Dropout(dropout_prop))
    network.add(Dense(int(n_rlu/2)))
    BatchNormalization()
    network.add(Activation(activation))
    # Fully Connected 3
    network.add(Dropout(dropout_prop))
    network.add(Dense(int(n_rlu/8)))
    BatchNormalization()
    network.add(Activation(activation))
    # Final layer
    network.add(Dropout(dropout_prop))
    network.add(Dense(1))
    BatchNormalization()
    # generate the final prediction
    network.add(Activation('sigmoid'))
    # return
    return(network)

### Model assuming difficult data structure
def model_hard(img_height, img_width, n_channel = 3, n_border = 1, n_filter = 32, n_pixel = 3, n_pool = 2, n_rlu = 4096, dropout_prop = .1, activation = 'elu', batchnorm = True):
    # initiate a sequential backpropagation network
    network = Sequential()
    # Convolution Block 1
    network.add(ZeroPadding2D(padding = (n_border, n_border), input_shape = (img_height, img_width, n_channel), data_format = "channels_last"))
    network.add(Conv2D(n_filter, (n_pixel, n_pixel)))
    if batchnorm:
        BatchNormalization()
    network.add(ELU())
    network.add(MaxPooling2D(pool_size = (n_pool, n_pool)))
    # Convolution Block 2
    network.add(ZeroPadding2D(padding = (n_border, n_border)))
    network.add(Conv2D(n_filter*2, (n_pixel, n_pixel)))
    if batchnorm:
        BatchNormalization()
    network.add(ELU())
    network.add(MaxPooling2D(pool_size = (n_pool, n_pool)))
    # Convolution Block 3
    network.add(ZeroPadding2D(padding = (n_border, n_border)))
    network.add(Conv2D(n_filter*2, (n_pixel, n_pixel)))
    if batchnorm:
        BatchNormalization()
    network.add(ELU())
    network.add(MaxPooling2D(pool_size = (n_pool, n_pool)))
    # Convolution Block 4
    network.add(ZeroPadding2D(padding = (n_border, n_border)))
    network.add(Conv2D(n_filter*4, (n_pixel, n_pixel)))
    if batchnorm:
        BatchNormalization()
    network.add(ELU())
    network.add(MaxPooling2D(pool_size = (n_pool, n_pool)))
    # Convolution Block 5
    network.add(ZeroPadding2D(padding = (n_border, n_border)))
    network.add(Conv2D(n_filter*8, (n_pixel, n_pixel)))
    if batchnorm:
        BatchNormalization()
    network.add(ELU())
    network.add(MaxPooling2D(pool_size = (n_pool, n_pool)))
    # Fully Connected 1
    network.add(Dropout(dropout_prop))
    network.add(Flatten())
    network.add(Dense(n_rlu))
    BatchNormalization()
    network.add(Activation(activation))
    # Fully Connected 2
    network.add(Dropout(dropout_prop))
    network.add(Dense(int(n_rlu/2)))
    BatchNormalization()
    network.add(Activation(activation))
    # Fully Connected 3
    network.add(Dropout(dropout_prop))
    network.add(Dense(int(n_rlu/8)))
    BatchNormalization()
    network.add(Activation(activation))
    # Final layer
    network.add(Dropout(dropout_prop))
    network.add(Dense(1))
    BatchNormalization()
    # generate the final prediction
    network.add(Activation('sigmoid'))
    # return
    return(network)

#!/usr/bin/python3
# helpers_sequence_generator.py
# Defines a custom generator from keras's Sequence class
# Zach Warner
# 4 September 2020

##### SETUP #####

### load modules
import os
import keras
import gzip
import numpy as np
from keras import callbacks as cbks
from keras.utils.generic_utils import Progbar, to_list
from keras.utils.data_utils import OrderedEnqueuer
from tensorflow.python.keras.utils.data_utils import iter_sequence_infinite
from scripts.helpers.helpers_data_generator import ImageDataGenerator, crop_image, resize_image, denoise_image_median

class OurGenerator(keras.utils.Sequence):

    def __init__(self, directory, partition, variable_name, augmentation, img_height, img_width, batch_size=32, shuffle=False, labels=None, seed=None, zipped='no', crop_forms = 'yes', denoise = 'yes', keep_names = 'no'):
        self.directory = directory
        self.X = partition
        self.y = labels
        self.nb_sample = len(self.X)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.zipped = zipped
        self.img_height = img_height
        self.img_width = img_width
        self.crop_forms = crop_forms
        self.denoise = denoise
        self.variable_name = variable_name
        self.keep_names = keep_names
        self.augmentation = augmentation
        # set up the imagedatagenerator from the parameters passed
        self.imgaug = ImageDataGenerator(**self.augmentation)
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        
        # Generate indexes of the batch
        if ((index+1)*self.batch_size <= self.nb_sample):
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        else:
            indexes = self.indexes[index*self.batch_size:]
        IDs = [self.X[k] for k in indexes]
        # get seeds for image augmentation
        np.random.seed(self.seed + index)
        seeds = np.random.randint(0, 4294967295, len(IDs))
        # Generate data
        if self.keep_names == 'yes':            
            if self.y is None:
                X, z = self.__data_generation(IDs, seeds)
                return X, z
            else:
                X, y, z = self.__data_generation(IDs, seeds)
                return X, y, z
        else:
            if self.y is None:
                X = self.__data_generation(IDs, seeds)
                return X
            else:
                X, y = self.__data_generation(IDs, seeds)
                return X, y      

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.nb_sample)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, IDs, seeds):
        'Generates data containing batch_size samples'
        for i, ID in enumerate(IDs):
            if self.zipped =='yes':
                x = np.load(gzip.GzipFile(os.path.join(self.directory, ID + '.npy.gz'), "r"))
            else:
                x = np.load(os.path.join(self.directory, ID + '.npy'))
            # ensure just three channels -- remove transparency if present
            x = x[...,:3]
            # crop the images to the areas of interest
            if self.crop_forms == 'yes':
                x = crop_image(x, self.variable_name)
            # resize image according to parameters
            x = resize_image(x, self.img_height, self.img_width)
            # denoise the image if requested
            if self.denoise == 'yes':
                x = denoise_image_median(x)
            # get random transformation parameters
            x_seed = seeds[i]
            params = self.imgaug.get_random_transform(x.shape, x_seed)
            # apply the transformation
            x = self.imgaug.apply_transform(x, params)
            # standardize the images
            x = self.imgaug.standardize(x)
            # return the batch(es)
            if i==0:
                batch_x = np.empty((len(IDs),)+x.shape)
            batch_x[i] = x
        if self.y is None and self.keep_names == 'yes':
            return batch_x, IDs
        elif self.y is None and self.keep_names != 'yes':
            return batch_x, IDs
        else: 
            batch_y = [self.y[x] for x in IDs]
            batch_y = np.asarray(batch_y)
            if self.keep_names == 'yes':
                return batch_x, batch_y, IDs
            else:
                return batch_x, batch_y

### function to predict out of the model, preserving file names
def OurPredictor(model, generator, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0):
    
    model._make_predict_function()
    steps_done = 0
    all_outs = []
    labels = []
    files = []
    if steps is None:
        steps = len(generator)
    enqueuer = None
    y, z = None, None

    # Check if callbacks have not been already configured
    if not isinstance(callbacks, cbks.CallbackList):
        callbacks = cbks.CallbackList(callbacks)
        callback_model = model._get_callback_model()
        callbacks.set_model(callback_model)
        callback_params = {
            'steps': steps,
            'verbose': verbose,
        }
        callbacks.set_params(callback_params)

    callbacks.model.stop_training = False
    callbacks._call_begin_hook('predict')

    try:
        if workers > 0:
            enqueuer = OrderedEnqueuer(
                generator,
                use_multiprocessing=use_multiprocessing)
            enqueuer.start(workers=workers, max_queue_size=max_queue_size)
            output_generator = enqueuer.get()
        else:
            output_generator = iter_sequence_infinite(generator)

        if verbose == 1:
            progbar = Progbar(target=steps)

        while steps_done < steps:
            generator_output = next(output_generator)
            if isinstance(generator_output, tuple):
                # if length is 2, we're predicting out of sample and want values and names
                if len(generator_output) == 2:
                    x, z = generator_output
                # if length is 3, we're predicting on the test sample and want values, true values, and names
                elif len(generator_output) == 3:
                    x, y, z = generator_output
                else:
                    raise ValueError('Output of generator should be '
                                     'a tuple `(x, y, sample_weight)` '
                                     'or `(x, y)`. Found: ' +
                                     str(generator_output))
            else:
                # Assumes a generator that only
                # yields inputs (not targets and sample weights).
                x = generator_output

            if x is None or len(x) == 0:
                # Handle data tensors support when no input given
                # step-size = 1 for data tensors
                batch_size = 1
            elif isinstance(x, list):
                batch_size = x[0].shape[0]
            elif isinstance(x, dict):
                batch_size = list(x.values())[0].shape[0]
            else:
                batch_size = x.shape[0]
            if batch_size == 0:
                raise ValueError('Received an empty batch. '
                                 'Batches should contain '
                                 'at least one item.')

            # generate predictions
            batch_logs = {'batch': steps_done, 'size': batch_size}
            callbacks._call_batch_hook('predict', 'begin', steps_done, batch_logs)
            outs = model.predict_on_batch(x)
            outs = to_list(outs)

            # save the labels
            if y is not None:
                y = y.tolist()
                labels.extend(y)
            # save the names
            if z is not None:
                files.extend(z)
            # save the predictions
            if not all_outs:
                for out in outs:
                    all_outs.append([])
            for i, out in enumerate(outs):
                all_outs[i].append(out)
            batch_logs['outputs'] = outs
            callbacks._call_batch_hook('predict', 'end', steps_done, batch_logs)
            steps_done += 1
            if verbose == 1:
                progbar.update(steps_done)

        callbacks._call_end_hook('predict')
    finally:
        if enqueuer is not None:
            enqueuer.stop()

    # clean up the predictions
    if len(all_outs) == 1:
        if steps_done == 1:
            to_return = all_outs[0][0]
        else:
            to_return = np.concatenate(all_outs[0])
    if steps_done == 1:
        to_return = [out[0] for out in all_outs]
    else:
        to_return = [np.concatenate(out) for out in all_outs]
        to_return = to_return[0].tolist()
        to_return = [item for sublist in to_return for item in sublist]

    # return results
    if y is not None:
        if z is not None:
            return to_return, labels, files
        else:
            return to_return, labels
    else:
        if z is not None:
            return to_return, files
        else:
            return to_return

### function to compute class weights
def calc_class_weights(partition, labels):
    # trim to just train labels
    labels_trim = [labels[item] for item in labels if item in partition['train']]
    # compute proportions
    sample_size = len(labels_trim)
    n1 = sum(labels_trim)/sample_size
    n0 = 1 - n1
    # normalize and return
    weights = np.max((n0, n1))/(n0, n1)
    weights = {0: weights[0], 1: weights[1]}
    return weights
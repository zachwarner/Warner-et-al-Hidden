#!/usr/bin/python3
# estimate_models.py
# Estimates the models
# Zach Warner
# 4 September 2020

##### SETUP #####

### import modules
import os # os needs some options to get reproducibility
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import History, ModelCheckpoint, ReduceLROnPlateau

### import our own functions
from scripts.helpers.helpers_data_generator import ImageDataGenerator
from scripts.helpers.helpers_sequence_generator import OurGenerator, OurPredictor
from scripts.helpers.helpers_performance import plot_diagnostics, crosstab, classification_metrics
from scripts.helpers.helpers_reading_data import parse_script_call, create_dicts, get_oos_dict
from scripts.helpers.helpers_models import model_average, model_wide, model_deep, model_hard, model_inception

### set random seeds -- all are required for precise replication
seed = 8675309 # hey jenny
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

### get arguments from script call
parameters = parse_script_call()
variable_name = parameters['variable_name']
img_height = np.int(parameters['img_size'])
img_width = np.int(parameters['img_size'])
batch_size = np.int(parameters['batch_size'])
n_epoch = np.int(parameters['n_epoch'])

### set up directory for saving things
save_dir = os.path.join(os.getcwd(), '/results/fit/')

##### VALIDATE FUNCTION CALL AND DATA #####

### create dictionaries
partition, labels = create_dicts(variable_name = variable_name)

### get data flow parameters
n_train = len(partition['train'])
n_validation = len(partition['validation'])
n_test = len(partition['test'])

# set the steps per epoch
n_step = n_train//batch_size
n_val_step = n_validation//batch_size

##### SET UP DATA FLOW #####

### set up the augmentation parameters
if parameters['aug'] == 'none':
	aug = {'rescale': 1./255}
elif parameters['aug'] == 'little':
	aug = {'rescale': 1./255, 'samplewise_center': True, 'samplewise_std_normalization': True, 'brightness_range': [.95, 1.05], 'channel_shift_range': 5, 'rotation_range': .05, 'width_shift_range': .05, 'height_shift_range': .05, 'zoom_range': .05, 'shear_range': .05, 'horizontal_flip': True, 'vertical_flip': True}

### let the data flow
train_generator = OurGenerator(directory = '/data/tiffs2013/train/', partition = partition['train'], labels = labels, variable_name = variable_name, augmentation = aug, batch_size = batch_size, img_height = img_height, img_width = img_width, seed = seed, keep_names = 'no')
validation_generator = OurGenerator(directory = '/data/tiffs2013/train/', partition = partition['validation'], labels = labels, variable_name = variable_name, augmentation = aug, batch_size = batch_size, img_height = img_height, img_width = img_width, seed = seed, keep_names = 'no')
test_generator = OurGenerator(directory = '/data/tiffs2013/train/', partition = partition['test'], labels = labels, variable_name = variable_name, augmentation = aug, batch_size = 1, img_height = img_height, img_width = img_width, seed = seed, keep_names = 'yes')

##### SET UP THE MODEL #####

### choose the model
if parameters['model'] == 'model_average':
	model = model_average(img_height = img_height, img_width = img_width)
elif parameters['model'] == 'model_wide':
	model = model_wide(img_height = img_height, img_width = img_width)
elif parameters['model'] == 'model_deep':
	model = model_deep(img_height = img_height, img_width = img_width)
elif parameters['model'] == 'model_hard':
	model = model_hard(img_height = img_height, img_width = img_width)
elif parameters['model'] == 'model_inception':
	model = model_inception(variable_name = variable_name, img_height = img_height, img_width = img_width)

### compile the model
model.compile(loss = 'binary_crossentropy', optimizer = 'adadelta', metrics = ['accuracy', 'mse'])

### set up the model log
history = History()

### set up the model to save
checkpoint = ModelCheckpoint(filepath = '/results/models/' + variable_name + ".h5", monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'auto')

### set up the dynamic adjustments to learning rate
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 5, cooldown = 1)

### set up callbacks list
callbacks_list = [checkpoint, history, reduce_lr]

##### TRAIN THE MODEL #####

### fit the model
model.fit_generator(generator = train_generator, steps_per_epoch = n_step, epochs = n_epoch, callbacks = callbacks_list, validation_data = validation_generator, validation_steps = n_val_step, verbose = 1)

### predict out on the test sample and reshape
predictions, true_values, file_names = OurPredictor(model, test_generator, steps = n_test, verbose = 1)
pred_data = pd.DataFrame({'prediction': predictions, 'predicted_class':[round(x) for x in predictions], 'true_value': true_values, 'file_name': file_names})

### compute classification accuracy metrics
report = classification_metrics(y_true = pred_data['true_value'], y_pred = pred_data['predicted_class'])
report = pd.DataFrame.from_dict(report, orient = 'index', columns = ['value'])
report.to_csv(save_dir + 'classification_report_' + variable_name + '.csv', na_rep = 'NA', index = True)

### save a confusion matrix
true_positives, true_negatives, false_positives, false_negatives = crosstab(y_true = pred_data['true_value'], y_pred = pred_data['predicted_class'])
cf = pd.DataFrame(data = {'Real 1': [true_positives, false_negatives], 'Real 0': [false_positives, true_negatives]}, index = ['Predicted 1', 'Predicted 0'])
cf.to_csv(save_dir + 'confusion_matrix_' + variable_name + '.csv', index = True)

### save training history
plot_diagnostics(name = '/results/fit/diagnostics_plot_' + variable_name + '.pdf', history = history, variable_name = variable_name, n_epoch = n_epoch)

##### PREDICT OUT OF SAMPLE #####

### get the out of sample data
oos_partition = get_oos_dict(variable_name = variable_name)
n_oos = len(oos_partition['oos'])

### set up the out of sample data flow
aug_oos = {'rescale': 1./255}
oos_generator = OurGenerator(directory = '/data/tiffs2013/test/', partition = oos_partition['oos'], variable_name = variable_name, augmentation = aug_oos, batch_size = 1, img_height = img_height, img_width = img_width, seed = seed, keep_names = 'yes')

### predict out of the model and save
predictions, file_names = OurPredictor(model, oos_generator, steps = n_oos, verbose = 1)
oos_data = pd.DataFrame({'prediction_' + variable_name: predictions, 'prediction_class_' + variable_name:[round(x) for x in predictions], 'filename': file_names})
oos_data.to_csv(save_dir + 'predictions_' + variable_name + '.csv', na_rep = "NA", index = False)

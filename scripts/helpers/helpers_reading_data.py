#!/usr/bin/python3
# helpers_reading_data.py
# Defines helper functions for reading and processing image data
# Zach Warner
# 4 September 2020

##### SET UP #####

import os
import sys
import csv
import random
import re
import glob
import numpy as np
from pathlib import Path

### function to print a progress bar for long loops not running in parallel
def progressBar(value, endvalue, bar_length = 30):
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write("\rPercent: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()

### function to parse arguments passed to the estimate_model script
def parse_script_call():
    # create empty dictionary
    parameter_dict = {}
    # iterate over inputs (skip sys.argv[0] since it's the function call)
    for user_input in sys.argv[1:]:
    # skip if we forgot to assign both name and value
        if "=" not in user_input:
            continue
        # get the name
        var_name = user_input.split("=")[0]
        # get the value
        var_value = user_input.split("=")[1]
        parameter_dict[var_name] = var_value
    # return the result
    return(parameter_dict)

### switch a variable name for its appropriate column index
def select_variable(variable_name):
    # define the universe of variables
    vars = ['filename', 'constituency_number', 'polling_station_number', 'stream_number', 'serial_number', 'qr_code', 'sheet_filename_match', 'editing_results', 'po_signature', 'first_page_stamped', 'dpo_signature', 'different_signature', 'agents', 'signed', 'all_agents_signed', 'different_sign', 'refusals', 'second_page_stamped', 'comments', 'missing_page_2', 'good_scan', 'edited_tallies', 'edited_results']
    # match the variable to its index, with a try/except for error handling
    try:
        ind = vars.index(variable_name)
    except ValueError:
        print("Selected invalid variable name. Chose one of the variable column names in labeled-training-data.csv and define it as string.")
    # defensive programming: cannot use ID variables
    if ind < 4:
        raise Exception('You cannot pass "filename," "constituency_number," "polling_station_number," or "stream_number" to the function call.')
    # return the result
    return(ind)

### function to get training and validation data as dictionaries
def create_dicts(variable_name, validation_prop = .1667, test_prop = .1667):

    ### translate named variable to index in labels
    variable_selected = select_variable(variable_name = variable_name)

    ### get the labels
    # set the path for the labeled training data
    labels_loc = os.getcwd() + "data/labeled-training-data.csv"
    # read in the training data
    labels = []
    with open(labels_loc, newline = '') as csvfile:
        reader = csv.reader(csvfile, delimiter = ',')
        for row in reader:
            labels.append(row)
    # make it an array
    labels = np.asarray(labels)
    # remove elements we don't need
    labels = np.delete(labels, [0, 1], 0)
    labels = np.delete(labels, [0, 5, 12], 1)
    # remove exception files -- missing PDFs
    labels = np.delete(labels, np.argwhere(labels[:,0] == '016-082-0409-139-1.pdf'), 0)
    labels = np.delete(labels, np.argwhere(labels[:,0] == '016-082-0409-153-1.pdf'), 0)
    # replace missing values with NAs
    labels[labels == 'N/A'] = 'NA'
    labels[labels == ''] = 'NA'
    labels[labels == ' '] = 'NA'
    labels[labels == 'nan'] = 'NA'
    # clean names variable
    labels[:,0] = np.char.replace(labels[:,0], '.pdf', '')
    # change pages (of the PDF) according to variable selected
    if variable_selected > 10 and variable_selected != 20 and variable_selected != 21 and variable_selected != 22:
        labels[:,0] = [s + ' 2' for s in labels[:,0]]
    labels = {x[0]:x[variable_selected] for x in labels}
    # remove NAs
    labels = {x:labels[x] for x in labels if not labels[x] == 'NA'}
    # convert to integers
    labels = {x:int(y) for x, y in labels.items()}

    ### get the train/validation/test partitions
    # create sample sizes
    sample_size = len(labels)
    n_validation = 500
    n_test = 500
    n_train = sample_size - n_validation - n_test
    # draw a random train sample from the dictionary
    train = random.sample(labels.keys(), n_train)
    # remove that train sample from the dictionary
    labels_trim = {item: labels[item] for item in labels if not item in train}
    # draw a random validation sample from the dictionary
    validation = random.sample(labels_trim.keys(), n_validation)
    # remove that validation sample from the dictionary
    labels_trim = {item: labels_trim[item] for item in labels_trim if not item in validation}
    if len(labels_trim) != n_test:
        ValueError('Cannot divide the training data into train, validation, and test splits in these proportions.')
    # assign the remaining sample to test
    test = list(labels_trim.keys())
    partition = {'train': train, 'validation': validation, 'test': test}

    ### return the objects
    return(partition, labels)

### create a dictionary with all of the file names for the test data
def get_oos_dict(variable_name, zipped = 'no'):

    ### translate named variable to index in labels
    variable_selected = select_variable(variable_name = variable_name)

    ### get all oos files
    loc = os.getcwd() + "data/tiffs2013/test"

    ### clean them
    if zipped == 'yes':
        all_files = Path(loc).rglob('*.npy.gz')
    else:
        all_files = Path(loc).rglob('*.npy')
    all_files = [str(x) for x in all_files if x.is_file()]

    ### clean
    all_files = [re.sub('^(.*)/', '', x) for x in all_files]
    if zipped == 'yes':
        if variable_selected > 10 and variable_selected != 20 and variable_selected != 21 and variable_selected != 22:
            all_files = [x for x in all_files if ' 2.npy.gz' in x]
        else:
            all_files = [x for x in all_files if '.npy.gz' in x]
            all_files = [x for x in all_files if not ' 2.npy.gz' in x]
        all_files = [re.sub('.npy.gz', '', x) for x in all_files]
    else:
        if variable_selected > 10 and variable_selected != 20 and variable_selected != 21 and variable_selected != 22:
            all_files = [x for x in all_files if ' 2.npy' in x]
        else:
            all_files = [x for x in all_files if '.npy' in x]
            all_files = [x for x in all_files if not ' 2.npy' in x]
        all_files = [re.sub('.npy', '', x) for x in all_files]

    ### return it as a dictionary
    partition = {'oos': all_files}
    return(partition)

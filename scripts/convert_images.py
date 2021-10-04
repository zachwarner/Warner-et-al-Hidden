#!/usr/bin/python3
# convert_images.py
# Converts images from PDF to npy arrays
# Zach Warner
# 4 September 2020

### import dependencies
import csv
import os
import sys
import random
import gzip
import re
import tqdm
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from pdf2image import convert_from_path
from keras.preprocessing.image import load_img, img_to_array
from helpers_reading_data import progressBar, parse_script_call

### set random seed
random.seed(8675309) # hey jenny

### get argument from script call
parameters = parse_script_call()
pixels = np.int(parameters['dpi'])
if 'zipped' in parameters:
	zipped = str(parameters['zipped'])
else:
	zipped = 'no'

### get file names and locations
# set the path for the input
in_path = "data/forms2013"
# get all image names
all_files = Path(in_path).rglob('*.pdf')
# get all image names with path
all_files = [str(x) for x in all_files if x.is_file()]
all_files = ['/' + x for x in all_files]
# trim to only include individual polling stations/streams
regex = re.compile('split')
all_files = [x for x in all_files if regex.search(x)]
# get all image names without path
all_fn = [re.sub('^(.*)/', '', x) for x in all_files]
all_fn = [re.sub('.pdf', '', x) for x in all_fn]

### get the training data csv
# set the path for the labeled training data
labels_loc = os.getcwd() + "data/labeled-training-data.csv"
# read in the training data csv
labels = []
with open(labels_loc, newline = '') as csvfile:
    reader = csv.reader(csvfile, delimiter = ',')
    for row in reader:
        labels.append(row)
# make it an array
labels = np.asarray(labels)
# remove elements we don't need -- we only need column with names
labels = np.delete(labels, [0, 1], 0)
labels = labels[:,1]
# clean to make same type as all_fn
labels = np.char.replace(labels, '.pdf', '')
# set standard image dimensions
img_width = round(pixels/100*827)
img_height = round(pixels/100*1169)

### fix one broken PDF
broken = all_files.index('/data/forms2013/074-KITUI SOUTH/074-KITUI SOUTH - split files/074 KITUI SOUTH -3.pdf')
del all_files[broken], all_fn[broken]

### define the function
def pdf_to_npy_gz(i, zipped = zipped):
# for i in range(len(all_files)):
	names = []
	# get the pages for each PDF
	pages = convert_from_path(all_files[i], pixels)
	# check whether it's in the training data
	in_train = all_fn[i] in labels
	# save the names for the first page
	names = names + [all_fn[i]]
	# save names for any additional pages
	if len(pages) > 1:
		y = [all_fn[i] + ' ' + str(x+1) for x in range(1,len(pages)) if x >= 1]
		names = names + y
	# convert each page to a numpy array via a temporary tiff
	for j in range(len(pages)):
		# create the temporary file
		pages[j].save(str(names[j]) + '.tiff', 'TIFF')
		# load the photo with the correct dimensions
		photo = load_img(str(names[j]) + '.tiff', target_size=(img_height, img_width))
		# delete the temproary file on disk
		os.remove(str(names[j]) + '.tiff')
		# convert the image to an array
		photo = img_to_array(photo, dtype='float32')
		# save to the array on disk
		if in_train:
			out_path = '/data/tiffs2013/train/'
		else:
			out_path = '/data/tiffs2013/test/'
		if zipped == 'yes':
			with gzip.GzipFile(out_path + names[j] + '.npy.gz' ,'w') as f:
				np.save(f, photo)
		else:
			np.save(out_path + names[j] + '.npy', photo)

### run the function in parallel
if __name__ == '__main__':
	pool = Pool()
	for _ in tqdm.tqdm(pool.imap_unordered(pdf_to_npy_gz, range(len(all_files))), total = len(all_files)):
		pass

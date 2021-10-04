#!/bin/bash
# voting-fraud-replication-master.sh
# A script to run all of the code and analysis for voting-fraud-replication
# Zach Warner
# 4 September 2020

# Set the working directory
cd /path/to/the/folder/this/file/is/in

### Grab the docker image from my repo
docker pull zachwarner/voting-irregularities:latest
docker run --rm -it -v $(pwd)/data:/data -v $(pwd)/logs:/logs -v $(pwd)/results:/results -v $(pwd)/scripts:/scripts zachwarner/voting-irregularities:latest

### Create the train and test samples by sampling polling streams
R CMD Rscript /scripts/sample_pdfs.R &> "/logs/sample_pdfs.log"

### Convert PDFs to numpy arrays
# This will take a few hours, depending on what you set for dpi, and how many
# cores are available to your docker container. Zipping adds time, too.
PYTHONHASHSEED=8675309 python3 /scripts/convert_images.py dpi=100 zipped=no &> "/logs/convert_images.log"

### Tune the models
# Given the very high computational costs of doing this (540 models, each
# taking approx 2 days to run), we do this on our university cluster. Scripts
# to replicate this tuning exercise can be obtained by emailing us.

### Estimate the models
# We need to keep resetting the hash seed to make sure it replicates precisely

# agents
PYTHONHASHSEED=8675309 python3 /scripts/estimate_model.py variable_name=agents n_epoch=20 model=model_deep aug=none img_size=256 batch_size=16 &> "/logs/estimate_model_agents.log"
# all agents signed
PYTHONHASHSEED=8675309 python3 /scripts/estimate_model.py variable_name=all_agents_signed n_epoch=40 model=model_inception aug=none img_size=256 batch_size=32 &> "/logs/estimate_model_all_agents_signed.log"
# different sign
PYTHONHASHSEED=8675309 python3 /scripts/estimate_model.py variable_name=different_sign n_epoch=20 model=model_deep aug=none img_size=512 batch_size=32 &> "/logs/estimate_model_different_sign.log"
# edited results
PYTHONHASHSEED=8675309 python3 /scripts/estimate_model.py variable_name=edited_results n_epoch=30 model=model_inception aug=none img_size=512 batch_size=16 &> "/logs/estimate_model_edited_results.log"
# first page stamped
PYTHONHASHSEED=8675309 python3 /scripts/estimate_model.py variable_name=first_page_stamped n_epoch=30 model=model_inception aug=little img_size=256 batch_size=32 &> "/logs/estimate_model_first_page_stamped.log"
# good scan
PYTHONHASHSEED=8675309 python3 /scripts/estimate_model.py variable_name=good_scan n_epoch=20 model=model_deep aug=none img_size=512 batch_size=16 &> "/logs/estimate_model_good_scan.log"
# po signature
PYTHONHASHSEED=8675309 python3 /scripts/estimate_model.py variable_name=po_signature n_epoch=30 model=model_deep aug=none img_size=512 batch_size=16 &> "/logs/estimate_model_po_signature.log"
# qr code
PYTHONHASHSEED=8675309 python3 /scripts/estimate_model.py variable_name=qr_code n_epoch=20 model=model_deep aug=none img_size=256 batch_size=16 &> "/logs/estimate_model_qr_code.log"
# signed
PYTHONHASHSEED=8675309 python3 /scripts/estimate_model.py variable_name=signed n_epoch=20 model=model_deep aug=none img_size=512 batch_size=32 &> "/logs/estimate_model_signed.log"

# R script 3: clean the output data
R CMD Rscript /scripts/clean_data.R &> "/logs/clean_data.log"

# R script 4: produce analysis
R CMD Rscript /scripts/analyze_results.R &> "/logs/analyze_results.log"

# Fix permissions
chmod -R a+rwX /data
chmod -R a+rwX /logs
chmod -R a+rwX /results
chmod -R a+rwX /scripts

# Exit
exit

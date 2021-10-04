### Replication archive for: "Hidden in Plain Sight?"

Welcome to the replication archive for "Hidden in Plain Sight? Irregularities on Statutory Forms and Electoral Fraud," by [Zach Warner](www.zachwarner.net), [J. Andrew Harris](http://www.jandrewharris.org/), Michelle Brown, and [Christian Arnold](http://christianarnold.org/), forthcoming in *Electoral Studies*.

The authors are listed in reverse-alphabetical order and all authors are jointly responsible for this replication archive.

### Replication instructions

Please read the computational considerations section below before attempting a complete replication run.

First, you should clone this repo. You'll also need to install Docker and ensure that it is running. See the instructions on Docker's website [here](https://docs.docker.com/get-docker/).

Second, you may need to upgrade the computing resources Docker is able to access on your machine. At time of writing, these options are listed under `Preferences -> Resources -> Advanced` and include CPUs, Memory, Swap, and Disk image size.

Third, open `Warner-et-al-Hidden.sh` and change line 8, which sets the working directory. Replace the placeholder working directory command by changing it to wherever the `.sh` script is located (and, if not on a UNIX system, the correct path command).

Finally, open a command-line environment (e.g., on a Mac, Terminal). Run each line of code in `Warner-et-al-Hidden.sh`. Note that you will need an internet connection at the very beginning, to pull the Docker image `zachwarner/voting-irregularities`.

### Computational considerations

Fitting each model in `python` consumes approximately 60-70Gb of memory, and so is not generally recommended. Further, on my Mac Pro with 192Gb of RAM, running the entire script line-by-line from start to finish takes approximately 5 days.

If you are interested in extending or replicating the statistical analysis only, then you can just open the `analyze_results.R` script and set the working directory as the replication archive. Note that you'll have to change some of the paths in the script from, e.g., `"/data"` to `"./data"` etc., unless you work within the Docker container.

### Manifest

This replication archive includes a number of files, so we provide the complete directory structure below.

Briefly, the `data` folder provides all of the data used in the final analysis. However, we do not include the original data files used to generate our final merged data for a few reasons. First, many of them are proprietary and we do not have the rights to redistribute them (e.g., the original scanned forms from Kenya's 2013 election or the Election Observation Group's data on polling station observers). Second, they are prohibitively large to distribute via GitHub; for example, the scanned forms alone are approximately 600Gb of data. Thus, the `covariates`, `forms2013`, and `tiffs2013` folders in `data` do not contain data.

More specifically,
- `filenames2013.csv` provides a list of all of the polling stations and the associated file name for the PDF of the scanned Form 34A
- `training2013.csv` is the stratified random sample of Forms 34A that were assigned to be manually coded for irregularities.
- `labeled-training-data.csv` is the labeled training data used to fit our deep neural networks.
- `final-for-analysis-trimmed.csv` is the final merged data used to produce the analyses presented in the paper.
- `edits/edited_results_recheck.xlsx` is the manual re-check of forms that were coded as having irregularities
- `final-audit-edited-tallies.csv` is the manual re-check, cleaned for analysis

The `logs` folder provides the log output for each of the scripts called in the `.sh` master file. It also includes the Dockerfile used to create the Docker image. (You do not actually need to build the Docker image.)

All results are stored in `results`, including diagnostics stored during the final fits of our models.

With the exception of the master `.sh` file, all scripts are in the `scripts` folder. The `helpers` folder defines `python` functions which are loaded as a local module in the `estimate_model.py` script.

```bash
├── data/
│   ├── covariates/
│   ├── edits/
│   │   └── edited_results_recheck.xlsx
│   ├── filenames2013.csv
│   ├── final-audit-edited-tallies.csv
│   ├── final-for-analysis-trimmed.csv
│   ├── forms2013/
│   ├── labeled-training-data.csv
│   ├── tiffs2013/
│   │   ├── test/
│   │   └── train/
│   └── training2013.csv
├── logs/
│   ├── analyze_results.log
│   ├── clean_data.log
│   ├── convert_images.log
│   ├── Dockerfile
│   ├── estimate_model_agents.log
│   ├── estimate_model_all_agents_signed.log
│   ├── estimate_model_different_sign.log
│   ├── estimate_model_edited_results.log
│   ├── estimate_model_first_page_stamped.log
│   ├── estimate_model_good_scan.log
│   ├── estimate_model_po_signature.log
│   ├── estimate_model_qr_code.log
│   ├── estimate_model_signed.log
│   └── sample_pdfs.log
├── README.md
├── results/
│   ├── figure-3.pdf
│   ├── figure-a1.pdf
│   ├── figure-a2.pdf
│   ├── figure-a3.pdf
│   ├── figure-a4.pdf
│   ├── figure-a5.pdf
│   ├── fit/
│   │   ├── classification_report_agents.csv
│   │   ├── classification_report_all_agents_signed.csv
│   │   ├── classification_report_different_sign.csv
│   │   ├── classification_report_edited_results.csv
│   │   ├── classification_report_first_page_stamped.csv
│   │   ├── classification_report_good_scan.csv
│   │   ├── classification_report_po_signature.csv
│   │   ├── classification_report_qr_code.csv
│   │   ├── classification_report_signed.csv
│   │   ├── confusion_matrix_agents.csv
│   │   ├── confusion_matrix_all_agents_signed.csv
│   │   ├── confusion_matrix_different_sign.csv
│   │   ├── confusion_matrix_edited_results.csv
│   │   ├── confusion_matrix_first_page_stamped.csv
│   │   ├── confusion_matrix_good_scan.csv
│   │   ├── confusion_matrix_po_signature.csv
│   │   ├── confusion_matrix_qr_code.csv
│   │   ├── confusion_matrix_signed.csv
│   │   ├── diagnostics_plot_agents.pdf
│   │   ├── diagnostics_plot_all_agents_signed.pdf
│   │   ├── diagnostics_plot_different_sign.pdf
│   │   ├── diagnostics_plot_edited_results.pdf
│   │   ├── diagnostics_plot_first_page_stamped.pdf
│   │   ├── diagnostics_plot_good_scan.pdf
│   │   ├── diagnostics_plot_po_signature.pdf
│   │   ├── diagnostics_plot_qr_code.pdf
│   │   ├── diagnostics_plot_signed.pdf
│   │   ├── predictions_agents.csv
│   │   ├── predictions_all_agents_signed.csv
│   │   ├── predictions_different_sign.csv
│   │   ├── predictions_edited_results.csv
│   │   ├── predictions_first_page_stamped.csv
│   │   ├── predictions_good_scan.csv
│   │   ├── predictions_po_signature.csv
│   │   ├── predictions_qr_code.csv
│   │   └── predictions_signed.csv
│   ├── table-1.csv
│   ├── table-2.csv
│   ├── table-3.csv
│   ├── table-a11.csv
│   ├── table-a12.csv
│   ├── table-a13.csv
│   ├── table-a14.csv
│   └── table-a15.csv
├── scripts/
│   ├── analyze_results.R
│   ├── clean_data.R
│   ├── convert_images.py
│   ├── estimate_model.py
│   ├── helpers/
│   │   ├── helpers_data_generator.py
│   │   ├── helpers_models.py
│   │   ├── helpers_performance.py
│   │   ├── helpers_reading_data.py
│   │   └── helpers_sequence_generator.py
│   └── sample_pdfs.R
├── README.md
└── Warner-et-al-Hidden.sh
```

### Hyperparameter tuning

This archive does not include the code used to run the entire hyperparameter tuning exercise. This is because we used `slurm` arrays on Cardiff University's high-performance computing cluster, and the code for these scripts is very specific to the requirements of that cluster.

To replicate the hyperparameter tuning, we recommend that you move the data, scripts, and folder structure to your university's cluster. You can then batch-create slurm scripts that simply call the `estimate_model.py` script, setting the tuning parameters as required (see the `Warner-et-al-hidden.sh` script for an example of how those parameters are specified). As discussed in the online appendix, we tuned over a hyperparameter grid of:

- `batch_size`: {16, 32}
- `img_size` = {256, 512}
- `aug` = {none, little, lot}
- `model`: {model_inception, model_average, model_wide, model_deep, model_hard}

You will also want to increase `n_epoch` to whatever limit you feel is reasonable. In our tuning code, we interrupted model fitting every 10 epochs to compute F<sub>1</sub> scores; model fitting stopped when we reached the maximum runtime on our cluster of 48 hours, when we reached 300 epochs, or when the F<sub>1</sub> was greater than 0.99.

### Contact

If you have questions or comments, then feel free to get in touch with Zach at [zachwarner@purdue.edu](mailto:zachwarner@purdue.edu) or Chris at [arnoldc6@cardiff.ac.uk](mailto:arnoldc6@cardiff.ac.uk).

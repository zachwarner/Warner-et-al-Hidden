#!/usr/bin/env bash

# Set the base environment
FROM ubuntu:18.04
ENV DEBIAN_FRONTEND=noninteractive

# Add the CRAN and poppler repositories for newer versions of R and poppler
RUN apt-get update \
	&& apt-get install -y software-properties-common=0.96.24.32.14 \
	&& apt-get update \
	&& apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9 \
	&& add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu bionic-cran40/' \
	&& add-apt-repository -y ppa:cran/poppler

# Install Ubuntu libraries
RUN apt-get update \
	&& apt-get install -y libpoppler-cpp-dev=0.74.0-bionic0 \
	&& apt-get install -y poppler-utils=0.74.0-bionic0 \
	&& apt-get install -y r-base-core=4.0.2-1.1804.0 \
	&& apt-get install -y r-base-dev=4.0.2-1.1804.0 \
	&& apt-get install -y python3.6=3.6.9-1~18.04ubuntu1.4 \
	&& apt-get install -y python3-pip=9.0.1-2.3~ubuntu1.18.04.5 \
	&& apt-get install -y python3-setuptools=39.0.1-2 \
	&& apt-get install -y python3-dev=3.6.7-1~18.04 \
	&& apt-get install -y libjpeg-dev=8c-2ubuntu8 \
	&& apt-get install -y libcurl4-openssl-dev=7.58.0-2ubuntu3.14 \
	&& apt-get install -y zlib1g-dev=1:1.2.11.dfsg-0ubuntu2 \
	&& apt-get install -y libssl-dev=1.1.1-1ubuntu2.1~18.04.13 \
	&& apt-get install -y libgdal-dev=2.2.3+dfsg-2 \
	&& apt-get install -y libproj-dev=4.9.3-2 \
	&& apt-get install -y wget=1.19.4-1ubuntu2.2

# Get R packages
RUN wget http://cran.r-project.org/src/contrib/Archive/remotes/remotes_2.1.0.tar.gz \
	&& R CMD INSTALL remotes_2.1.0.tar.gz \
	&& Rscript -e 'remotes::install_version("foreign", "0.8-81")' \
	&& Rscript -e 'remotes::install_version("ggplot2", "3.3.5")' \
	&& Rscript -e 'remotes::install_version("MASS", "7.3-54")' \
	&& Rscript -e 'remotes::install_version("mice", "3.13.0")' \
	&& Rscript -e 'remotes::install_version("miceadds", "3.11-6")' \
	&& Rscript -e 'remotes::install_version("patchwork", "1.1.1")' \
	&& Rscript -e 'remotes::install_version("raster", "3.4-13")' \
	&& Rscript -e 'remotes::install_version("rdrobust", "1.0.5")' \
	&& Rscript -e 'remotes::install_version("readxl", "1.3.1")' \
	&& Rscript -e 'remotes::install_version("rgdal", "1.5-23")' \
	&& Rscript -e 'remotes::install_version("sandwich", "3.0-1")' \
	&& Rscript -e 'remotes::install_version("sp", "1.4-5")'

# Get Python modules - h5py downgrade last because it breaks in 3.0.0
RUN pip3 install keras==2.3.1 \
	&& pip3 install matplotlib==3.1.2 \
	&& pip3 install numpy==1.18.1 \
	&& pip3 install pandas==0.25.3 \
	&& pip3 install pdf2image==1.12.1 \
	&& pip3 install scikit-image==0.16.2 \
	&& pip3 install scikit-learn==0.22.1 \
	&& pip3 install tensorflow==2.0.0b1 \
	&& pip3 install tqdm==4.42.1 \
	&& pip3 install h5py==2.10.0

# Set Python path for local scripts
ENV PYTHONPATH '/:/scripts/helpers'

#!/usr/bin/env Rscript
# sample_pdfs.R
# Draws a stratified random sample of polling streams
# Zach Warner
# 4 September 2020

##### SETUP #####

### set the RNGversion to R3.5, which is what was used when the training sample was drawn
RNGversion("3.5")

### set seed
set.seed(8675309) # hey jenny

### choose overall training size
n_train <- 3000

##### GET THE NUMBER OF SAMPLES PER COUNTY #####

### read in data with PDF names
df <- read.csv("/data/filenames2013.csv", stringsAsFactors = F)

### aggregate population by county
co <- aggregate(Qty.of.voters.in.the.stream ~ COUNTY_NAME, df, sum)
names(co) <- c("county", "pop")

### get the number of samples to draw from each county
co$pop_prop <- co$pop/sum(co$pop)
co$samples <- co$pop_prop*n_train
co$n_samples <- round(co$samples, 0)

### check that the number of samples rounds to 3000
stopifnot(sum(co$n_samples) == 3000)

##### CHECKING THE FILES THAT EXIST #####

### export the list of files 
ff <- list.files(path = "/data/forms2013", full.names = T, pattern = "*.pdf", recursive = T)

### match PDF names to their full paths
f <- data.frame(path = ff, name = ff)
f$name <- gsub("(.*)([0-9]{3}-[0-9]{3}-[0-9]{4}-[0-9]{3}-[0-9]{1,2})", "\\2", 
               f$name)

### trim df to account for missingness in the files
df <- df[which(df$filename %in% f$name),]

#### DRAWING A STRATIFIED SAMPLE #####

### split the files by county
dat <- split(df, df$COUNTY_NAME)
dat <- lapply(dat, function(x) {
  x[sample(nrow(x), co$n_samples[which(co$county == unique(x$COUNTY_NAME))]),]
})
dat <- do.call(rbind, dat)

### confirm samples all there
stopifnot(nrow(dat) == n_train)

### check that the samples are balanced
co$n_sampled <- NA
for(i in 1:nrow(co)){
  co$n_sampled[i] <- nrow(dat[which(dat$COUNTY_NAME == co$county[i]),])
}
stopifnot(all.equal(co$n_samples, co$n_sampled))

##### SAVE THE SAMPLED FILE PATHS FOR BASH SCRIPT #####

### trim to the samples
f <- f[which(f$name %in% dat$filename),]

### paste the full path
f$path <- gsub("^\\.", "", f$path)
f$path <- paste(getwd(), f$path, sep = "")

##### CREATE THE CODING SPREADSHEET #####

### create a holding dataframe
out <- data.frame(matrix(ncol = 0, nrow = n_train))

### coder is final
out$coder <- rep("Final", n_train)

### filename
out$filename <- dat$filename

### constituency number
out$constituency_num <- dat$CONSTITUENCY_CODE

### polling station number
out$polling_stn_num <- dat$POLLING_STATION_CODE

### stream number
out$stream_num <- dat$Stream

### write it
write.csv(out, "/data/training2013.csv", row.names = F)

### version control
sessionInfo()
# R version 4.1.1 (2021-08-10)
# Platform: x86_64-pc-linux-gnu (64-bit)
# Running under: Ubuntu 18.04.5 LTS
# 
# Matrix products: default
# BLAS:   /usr/lib/x86_64-linux-gnu/blas/libblas.so.3.7.1
# LAPACK: /usr/lib/x86_64-linux-gnu/lapack/liblapack.so.3.7.1
# 
# Random number generation:
# RNG:     Mersenne-Twister 
# Normal:  Inversion 
# Sample:  Rounding 
# 
# locale:
# [1] C
# 
# attached base packages:
# [1] stats     graphics  grDevices utils     datasets  methods   base     
# 
# loaded via a namespace (and not attached):
# [1] compiler_4.1.1
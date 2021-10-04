#!/usr/bin/env Rscript
# clean_data.R
# Cleans data in preparation for final modeling and analysis
# Zach Warner
# 30 August 2021

##### SET UP #####

### load required libraries
library(foreign); library(readxl); library(sp)

### set the seed
set.seed(8675309) # hey jenny

##### IMPORT MASTER DATA #####

### read in filenames data
df <- read.csv("/data/filenames2013.csv", stringsAsFactors = F)

### clean
df$X <- df$cawid <- df$constid <- df$countyid <- NULL
names(df) <- tolower(names(df))
names(df)[which(names(df) == "qty.of.voters.in.the.stream")] <- 
  "registered_voters_filenames"

### get the list of pdfs availabile to us
ff <- list.files(path = "/data/forms2013", full.names = T, pattern = "*.pdf", recursive = T)

### match PDF names to their full paths
f <- data.frame(path = ff, name = ff)
f$name <- gsub("(.*)([0-9]{3}-[0-9]{3}-[0-9]{4}-[0-9]{3}-[0-9]{1,2})", "\\2", 
               f$name)

### save variable for whether missing or not
df$data_source <- ifelse(df$filename %in% f$name, "present in all data", 
                         "present in master sample but not forms")

### clean up
rm(f, ff)

##### CLEAN UP MASTER DATAFRAME #####

### fix ID variables to character and pad length as appropriate
# county
df$county_code <- as.character(df$county_code)
df$county_code <- ifelse(nchar(df$county_code) == 1, 
                         paste("00", df$county_code, sep = ""),
                         df$county_code)
df$county_code <- ifelse(nchar(df$county_code) == 2, 
                         paste("0", df$county_code, sep = ""),
                         df$county_code)
# constituency
df$constituency_code <- as.character(df$constituency_code)
df$constituency_code <- ifelse(nchar(df$constituency_code) == 1, 
                               paste("00", df$constituency_code, sep = ""),
                               df$constituency_code)
df$constituency_code <- ifelse(nchar(df$constituency_code) == 2, 
                               paste("0", df$constituency_code, sep = ""),
                               df$constituency_code)
# ward
df$caw_code <- as.character(df$caw_code)
df$caw_code <- ifelse(nchar(df$caw_code) == 1, 
                      paste("000", df$caw_code, sep = ""),
                      df$caw_code)
df$caw_code <- ifelse(nchar(df$caw_code) == 2, 
                      paste("00", df$caw_code, sep = ""),
                      df$caw_code)
df$caw_code <- ifelse(nchar(df$caw_code) == 3, 
                      paste("0", df$caw_code, sep = ""),
                      df$caw_code)
names(df)[which(names(df) == "caw_code")] <- "ward_code"
names(df)[which(names(df) == "caw_name")] <- "ward_name"
# polling station
df$polling_station_code <- as.character(df$polling_station_code)
df$polling_station_code <- ifelse(nchar(df$polling_station_code) == 1, 
                                  paste("00", df$polling_station_code, sep = ""),
                                  df$polling_station_code)
df$polling_station_code <- ifelse(nchar(df$polling_station_code) == 2, 
                                  paste("0", df$polling_station_code, sep = ""),
                                  df$polling_station_code)
# stream
df$stream <- as.character(df$stream)

### fix psid variable
df$psid <- paste(df$constituency_code, df$polling_station_code, sep = '/')

##### READ IN OUTCOME DATA FROM MANUAL CODING OF TRAINING SAMPLE #####

### import the training sample data and trim it
train <- read.csv("/data/labeled-training-data.csv", stringsAsFactors = F)
train <- train[-1,]
train$X <- train$FIRST.PAGE <- train$SECOND.PAGE <- NULL

### fix names
names(train) <- c("filename", "constituency_code", "polling_station_code", 
                  "stream", "serial_number", "qr_code", 
                  "sheet_filename_match", "editing_results", "po_signature", 
                  "first_page_stamped", "dpo_signature", "different_signature",
                  "agents", "signed", "all_agents_signed", "different_sign", 
                  "refusals", "second_page_stamped", "comments", 
                  "missing_page_2", "good_scan", "edited_tallies")

### create edited_results variable
train$editing_results <- as.integer(train$editing_results)
train$edited_results <- ifelse(train$editing_results == 1 | 
                                 train$edited_tallies == 1, 1, 0)

### drop variables we aren't using
train <- train[,c("filename", "constituency_code", "polling_station_code", 
                  "stream", "qr_code", "edited_results", "po_signature", 
                  "first_page_stamped", "agents", "signed", 
                  "all_agents_signed", "different_sign", "good_scan")]

### add empty predictions columns for merging in prediction data
for(i in 5:ncol(train)){
  eval(parse(text = paste("train$prediction_", names(train)[i], 
                          " <- rep(NA, nrow(train))", sep = "")))
}

### add in a variable for training sample status
train$train <- 1

### minor cleaning
for(i in 1:ncol(train)){
  train[,i] <- as.character(train[,i])
  train[which(train[,i] == ""),i] <- NA
  train[which(train[,i] == "N/A"),i] <- NA
}
rm(i)

### fix ID variables to be the right length
# constituency
train$constituency_code <- ifelse(nchar(train$constituency_code) == 1, 
                               paste("00", train$constituency_code, sep = ""),
                               train$constituency_code)
train$constituency_code <- ifelse(nchar(train$constituency_code) == 2, 
                               paste("0", train$constituency_code, sep = ""),
                               train$constituency_code)
# polling station
train$polling_station_code <- ifelse(nchar(train$polling_station_code) == 1, 
                                  paste("00", train$polling_station_code, sep = ""),
                                  train$polling_station_code)
train$polling_station_code <- ifelse(nchar(train$polling_station_code) == 2, 
                                  paste("0", train$polling_station_code, sep = ""),
                                  train$polling_station_code)

##### READ IN OUTCOME DATA FROM CLASSIFICATION TASKS AND MERGE #####

### read in data
p <- list.files(path = "/results/fit", full.names = T, pattern = "predictions")
p <- lapply(p, FUN = function(x) read.csv(x, stringsAsFactors = F))

### remove the page 2 denomination
for(i in 1:length(p)){
  p[[i]]$filename <- gsub(" 2", "", p[[i]]$filename, fixed = T)
}

### merge
p <- Reduce(function(...) merge(..., all = T), p)

### clean filename to match the style in the train dataframe
p$filename <- paste(p$filename, ".pdf", sep = "")

### add variable for train status
p$train <- "0"

### extract ID variables to match train dataframe
# constituency code
p$constituency_code <- gsub("^[0-9]{3}-", "", p$filename)
p$constituency_code <- gsub("-.*", "", p$constituency_code)
# polling station code
p$polling_station_code <- gsub("^.*[0-9]{4}-", "", p$filename)
p$polling_station_code <- gsub("-.*", "", p$polling_station_code)
# stream
p$stream <- gsub("^.*-", "", p$filename)
p$stream <- gsub(".pdf", "", p$stream, fixed = T)

### drop cases that can't be matched on geographic IDs - these are typically
### merge files that snuck into the final data
p <- p[grepl("^\\d+$", p$constituency_code),]

### rename columns to match then reorder to match train data
names(p)[which(grepl("prediction_class", names(p), fixed = T))] <- 
  gsub("prediction_class_", "", 
       names(p)[which(grepl("prediction_class", names(p), fixed = T))])
p <- p[names(train)]

### merge train and predicted data
p <- merge(p, train, all = T)

### delete training sample
rm(train)

### create PSID
p$psid <- paste(p$constituency_code, p$polling_station_code, sep = "/")

##### MERGE OUTCOME DATA WITH MASTER DATAFRAME #####

### merge with the main dataframe
df <- merge(df, p, 
            by = c("psid", "filename", "constituency_code", 
                   "polling_station_code", "stream"), 
            all = T)
rm(p)

### create a unique polling station - stream identifier
df$psstrid <- paste(df$psid, df$stream, sep = "/")

### keep track of where the extra files came from
df$data_source[which(is.na(df$data_source))] <- 
  "present in forms but not master sample"

##### IMPORT DATA WITH ELECTION RESULTS #####

### read in data with election results
res <- read.csv("/data/covariates/polling-stream-results.csv", 
  stringsAsFactors = F)

### remove a few duplicate observations
res <- data.frame(res[-which(duplicated(res$psstrid)),])

### remove unnecessary columns
res$fullid <- res$const <- res$constid <- res$ps <- res$psnum <- res$str <- 
  res$error <- NULL

### fix column names
names(res) <- c("registered_voters", "ballots_spoiled", "ballots_cast",
  "ballots_rejected", "ballots_disputed", "ballots_objected", "ballots_valid",
  "votes_kiyiapi", "votes_karua", "votes_dida", "votes_mudavadi", 
  "votes_muite", "votes_kenneth", "votes_odinga", "votes_kenyatta", 
  "turnout_count", "turnout", "vote_share_odinga", "vote_share_kenyatta",
  "vote_margin", "vote_share_other", "psstrid", "psid")

### create absolute vote margin
res$vote_margin_abs <- abs(res$vote_margin)

### check whether there are errors in the tallies
vals <- which(grepl("votes", names(res)))
res$error_valid <- as.integer(rowSums(res[,vals]) != res$ballots_valid)
res$error_cast <- as.integer(rowSums(res[,c(names(res)[vals], "ballots_rejected")]) != res$ballots_cast)
res$problem <- ifelse(res$error_valid + res$error_cast > 0, 1, 0)

### merge
df <- merge(df, res, by = c("psstrid", "psid"), all = T)

### update data_source info
df$data_source[which(is.na(df$data_source))] <-
  "present in results but not forms or master sample"

### clean up
rm(vals)

##### MERGE IN PVT DATA #####

### load PVT data
pvt <- read.csv("/data/covariates/polling-stream-pvt.csv", stringsAsFactors = F)

### harmonize some variable classes and names
pvt$ward_code <- as.character(as.numeric(pvt$wardid))
pvt$county_code <- as.character(as.numeric(pvt$countyid))
pvt$stream <- as.character(as.numeric(pvt$str))
pvt$polling_station_code <- as.character(as.numeric(pvt$psnum))
pvt$constituency_code <- as.character(as.numeric(pvt$constid))
pvt$pvt_observed <- as.numeric(pvt$obs)
pvt$pvt_obs_no_over <- as.numeric(pvt$snatl)
pvt$registered_voters_pvt <- as.numeric(pvt$rv)

### trim so we only have the variables we want
pvt <- pvt[,c("county_code", "constituency_code", "ward_code",
              "polling_station_code", "stream", "pvt_observed", 
              "pvt_obs_no_over", "registered_voters_pvt")]

### fix ID variables to character and pad length as appropriate
# county
pvt$county_code <- ifelse(nchar(pvt$county_code) == 1, 
                         paste("00", pvt$county_code, sep = ""),
                         pvt$county_code)
pvt$county_code <- ifelse(nchar(pvt$county_code) == 2, 
                         paste("0", pvt$county_code, sep = ""),
                         pvt$county_code)
# constituency
pvt$constituency_code <- ifelse(nchar(pvt$constituency_code) == 1, 
                               paste("00", pvt$constituency_code, sep = ""),
                               pvt$constituency_code)
pvt$constituency_code <- ifelse(nchar(pvt$constituency_code) == 2, 
                               paste("0", pvt$constituency_code, sep = ""),
                               pvt$constituency_code)
# ward
pvt$ward_code <- ifelse(nchar(pvt$ward_code) == 1, 
                      paste("000", pvt$ward_code, sep = ""),
                      pvt$ward_code)
pvt$ward_code <- ifelse(nchar(pvt$ward_code) == 2, 
                      paste("00", pvt$ward_code, sep = ""),
                      pvt$ward_code)
pvt$ward_code <- ifelse(nchar(pvt$ward_code) == 3, 
                      paste("0", pvt$ward_code, sep = ""),
                      pvt$ward_code)
# polling station
pvt$polling_station_code <- ifelse(nchar(pvt$polling_station_code) == 1, 
                                  paste("00", pvt$polling_station_code, sep = ""),
                                  pvt$polling_station_code)
pvt$polling_station_code <- ifelse(nchar(pvt$polling_station_code) == 2, 
                                  paste("0", pvt$polling_station_code, sep = ""),
                                  pvt$polling_station_code)

### create a count of the number of registered voters by polling center
# group streams by polling station
pvt$psid <- paste(pvt$constituency_code, pvt$polling_station_code, sep = "/")
# count registered voters by station
pvt <- split(pvt, f = pvt$psid)
pvt <- lapply(pvt, FUN = function(x){
  x$registered_voters_center <- sum(x$registered_voters_pvt, na.rm = T)
  x
})
pvt <- do.call(rbind, pvt)
# clean up
pvt$psid <- NULL

### merge
df <- merge(df, pvt, by = c("county_code", "constituency_code", "ward_code",
                            "polling_station_code", "stream"), all = T)

##### COMPUTE PVT WEIGHTS #####

### We want to compute the probability of a polling station being treated in
### the national sample, which is a stratified (by county) random sample. We
### will compute IPW as well as stabilized IPW.

### create a dataset with the stratification information
x <- aggregate(df$pvt_observed, by = list(county = df$county_name), FUN = sum)
y <- data.frame(table(df$county_name))
names(y) <- c("county", "n")
names(x) <- c("county", "n_obs")
ww <- merge(x, y, by = "county", all = T)
ww$prob <- ww$n_obs/ww$n
p_t <- sum(ww$n_obs, na.rm = T)/sum(ww$n)

### create weight variables and store observations without county name since 
### they'll get dropped below and we want to keep them for completeness
df$pvt_ipw <- df$pvt_sipw <- rep(NA, nrow(df))
dia <- df[is.na(df$county_name),]

### create weights for treated units within each county
dd <- split(df, f = df$county_name)
for(i in 1:length(dd)){
  dd[[i]]$pvt_ipw <- ifelse(dd[[i]]$pvt_observed == 1, 
                        1/ww$prob[which(ww$county == unique(dd[[i]]$county_name))],
                        1/(1 - ww$prob[which(ww$county == unique(dd[[i]]$county_name))]))
  dd[[i]]$pvt_sipw <- ifelse(dd[[i]]$pvt_observed == 1, 
                            p_t/ww$prob[which(ww$county == unique(dd[[i]]$county_name))],
                            (1-p_t)/(1 - ww$prob[which(ww$county == unique(dd[[i]]$county_name))]))
}
df <- do.call(rbind, dd)
df <- rbind(df, dia)

### clean up
row.names(df) <- NULL
df <- df[order(df$psid, df$psstrid),]
rm(x, y, ww, dd, i, p_t, dia)

##### IMPORT COVARIATE DATA #####

### import covariate data
cov <- read.csv("/data/covariates/polling-station-covariates.csv", stringsAsFactors = F)

### subset and change names
cov <- cov[,c("psid", "pd", "pov", "lit", "deprat", "sba", "dem", "slope", 
  "terrain", "lights")]
names(cov) <- c("psid", "pop_density", "poverty_rate", "literacy_rate", 
                "dependency_ratio", "skilled_birth_att", "terrain_elevation", 
                "terrain_slope", "terrain_ruggedness", "night_lights")

### merge
df <- merge(df, cov, by = "psid", all = T)
rm(cov)

##### MERGE IN ETHNICITY DATA #####

### import ethnicity data
eth <- read.csv("/data/covariates/polling-stream-ethnicity.csv", stringsAsFactors = F)

### change names
names(eth)[1:12] <- paste("prop_", tolower(names(eth)[1:12]), sep = "")

### subset to columns we need
eth$rv <- eth$str <- eth$psid <- NULL

### merge with the main dataframe
df <- merge(df, eth, by = "psstrid", all = T)
rm(eth)

### update data source variable
df$data_source[which(is.na(df$data_source))] <- 
  "present in ethnicity but not forms or master sample"

##### IMPORT GEOLOCATION DATA #####

### import geolocation data
loc <- readRDS("/data/covariates/polling-station-location.rds")

### reproject the data to be in lat-long
geo_proj <- "+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0"
loc <- spTransform(loc, geo_proj)
rm(geo_proj)

### transform into simple data frame
loc <- data.frame(loc)

### subset
loc <- loc[,c("psid13", "type13", "subtype13", "mp_code", "coords.x1", 
              "coords.x2")]
# fix names
names(loc) <- c("psid", "polling_station_type", "polling_station_subtype", 
                "mp_code", "longitude", "latitude")

### merge
df <- merge(df, loc, by = "psid", all = T)
rm(loc)

### update data source variable
df$data_source[which(is.na(df$data_source))] <- 
  "present in location but not forms or master sample"

##### CREATE MEASURES OF ISOLATION #####

### generate distance matrix
dd_reg <- df[, c("psstrid", "registered_voters_filenames", "longitude", 
                 "latitude")]
dd_reg <- na.omit(dd_reg)
distmat_reg <- dist(dd_reg[,c("longitude", "latitude")], diag = T, upper = T)
distmat_reg <- as.matrix(distmat_reg)

### get a naive isolation value, nearest non-zero neighbor
iso <- unname(apply(distmat_reg, 2, function(x) {min(x[which(x > 0)])}))

### multiply through by population, as proxied by registered voters, decayed
distmat_reg <- .5^distmat_reg
distmat_reg <- dd_reg$registered_voters_filenames*distmat_reg

### take the maximum value, log it, reverse it, and shift it, for each
distmat_reg <- unname(apply(distmat_reg, 2, max))
distmat_reg <- log(distmat_reg)*-1
distmat_reg <- distmat_reg + abs(min(distmat_reg))

### add it back to the dataframe
dd_reg$isolation_simple <- iso
dd_reg$isolation_doogan_rv <- distmat_reg
df <- merge(df, dd_reg, by = c("psstrid", "registered_voters_filenames", 
                               "longitude", "latitude"), 
            all = T)
rm(distmat_reg, dd_reg, iso)

### now do the same steps for population density
dd_den <- df[, c("psstrid", "pop_density", "longitude", "latitude")]
dd_den <- na.omit(dd_den)
distmat_den <- dist(dd_den[,c("longitude", "latitude")], diag = T, upper = T)
distmat_den <- as.matrix(distmat_den)
distmat_den <- .5^distmat_den
distmat_den <- dd_den$pop_density*distmat_den
distmat_den <- unname(apply(distmat_den, 2, max))
distmat_den <- log(distmat_den)*-1
distmat_den <- distmat_den + abs(min(distmat_den))
dd_den$isolation_doogan_popden <- distmat_den
df <- merge(df, dd_den, by = c("psstrid", "pop_density", "longitude", 
                               "latitude"), 
             all = T)
rm(distmat_den, dd_den)

### but we need to remove duplicate rows induced by the all = T. Note only
### necessary because pop_density is not observed at stream level, unlike rv was
df <- df[!duplicated(df),]

##### FIX VARIABLE CLASSES BEFORE SOME RECODING #####

### make numerics
nums <- which(!(names(df) %in% c("psid", "psstrid", "filename", "mp_code", 
                                 "constituency_name", "constituency_code", 
                                 "county_name", "county_code", "stream", 
                                 "ward_name", "ward_code", "data_source", 
                                 "polling_station_name", "polling_station_code", 
                                 "polling_station_type", "polling_station_subtype")))
for(i in nums){
  df[,i] <- as.numeric(df[,i])
}
rm(i, nums)

##### RESCALE ETHNICITY VARIABLES #####

### fix a few proportions which are below zero or over one, though none are >1
props <- which(grepl("prop_", names(df)))
for(i in props){
  df[which(df[,i] < 0),i] <- 0
  df[which(df[,i] > 1),i] <- 1
}

### now we need to rescale so that proportions sum to 1
prop_total <- rowSums(df[,which(grepl("prop_", names(df), fixed = T))], 
                      na.rm = T)
for(i in props){
  df[,i] <- (1/prop_total)*df[,i]
}
rm(props, i, prop_total)

### generate measure of ethnic fractionalization
df$ethnic_frac <- (1 - (df$prop_kalenjin^2 + df$prop_kamba^2 + 
  df$prop_kikuyu^2 + df$prop_kisii^2 + df$prop_luhya^2 + df$prop_luo^2 + 
  df$prop_masai^2 + df$prop_meru^2 + df$prop_mijikenda^2 + df$prop_pokot^2 + 
  df$prop_somali^2 + df$prop_turkana^2))

##### CREATE STRONGHOLD VARIABLES #####

### variable for whether it was an oversampled county
df$county_oversampled <- ifelse(df$county_name %in% c("NAKURU", "MOMBASA", 
                                                      "NAIROBI CITY"),
                                1, 0)

### compute county-wide ethnic breakdowns
ww <- data.frame(county_name = df$county_name,
                 rv = df$registered_voters_filenames,
                 kikuyu = df$prop_kikuyu*df$registered_voters_filenames,
                 kalenjin = df$prop_kalenjin*df$registered_voters_filenames,
                 luo = df$prop_luo*df$registered_voters_filenames,
                 kamba = df$prop_kamba*df$registered_voters_filenames)
ww <- aggregate(.~ county_name, ww, FUN=sum)
ww$prop_kikuyu <- ww$kikuyu/ww$rv
ww$prop_kalenjin <- ww$kalenjin/ww$rv
ww$prop_luo <- ww$luo/ww$rv
ww$prop_kamba <- ww$kamba/ww$rv
ww$prop_gov <- ww$prop_kalenjin + ww$prop_kikuyu
ww$prop_opp <- ww$prop_luo + ww$prop_kamba

### create lists of counties in each stronghold variously defined
gov_sh <- ww$county_name[which(ww$prop_gov >= .9)]
opp_sh <- ww$county_name[which(ww$prop_opp >= .9)]
gov_sh_80 <- ww$county_name[which(ww$prop_gov >= .8)]
opp_sh_80 <- ww$county_name[which(ww$prop_opp >= .8)]
gov_sh_core <- ww$county_name[which(ww$prop_kikuyu >= .9)]
opp_sh_core <- ww$county_name[which(ww$prop_luo >= .9)]
gov_sh_core_80 <- ww$county_name[which(ww$prop_kikuyu >= .8)]
opp_sh_core_80 <- ww$county_name[which(ww$prop_luo >= .8)]

### code up the final variables
df$govt_stronghold <- ifelse(df$county_name %in% gov_sh, 1, 0)
df$opp_stronghold <- ifelse(df$county_name %in% opp_sh, 1, 0)
df$govt_stronghold_80 <- ifelse(df$county_name %in% gov_sh_80, 1, 0)
df$opp_stronghold_80 <- ifelse(df$county_name %in% opp_sh_80, 1, 0)
df$govt_stronghold_core <- ifelse(df$county_name %in% gov_sh_core, 1, 0)
df$opp_stronghold_core <- ifelse(df$county_name %in% opp_sh_core, 1, 0)
df$govt_stronghold_core_80 <- ifelse(df$county_name %in% gov_sh_core_80, 1, 0)
df$opp_stronghold_core_80 <- ifelse(df$county_name %in% opp_sh_core_80, 1, 0)

### clean up
rm(ww, gov_sh, gov_sh_80, opp_sh, opp_sh_80, gov_sh_core, gov_sh_core_80, 
   opp_sh_core, opp_sh_core_80)

### create manual codings based on our understanding of counties
df$govt_stronghold_manual <- ifelse(df$county_name %in% c("KIAMBU", "KIRINYAGA",
                                                          "MURANG'A", 
                                                          "NYANDARUA", "NYERI"), 
                                    1, 0)
df$opp_stronghold_manual <- ifelse(df$county_name %in% c("HOMA BAY", "KISUMU", 
                                                         "MIGORI", "SIAYA"), 
                                   1, 0)

### create codings based on county vote share: start by aggregating vs by county
dd <- split(df, f = df$county_name)
dd <- lapply(dd, FUN = function(x){
  k_vs_co <- sum(x$votes_kenyatta, na.rm = T)/sum(x$ballots_valid, na.rm = T)
  o_vs_co <- sum(x$votes_odinga, na.rm = T)/sum(x$ballots_valid, na.rm = T)
  return(data.frame(county_name = unique(x$county_name),
                    k_vs = k_vs_co,
                    o_vs = o_vs_co))
})
dd <- do.call(rbind, dd)
row.names(dd) <- NULL

### grab counties defined as strongholds at diff thresholds
gov75 <- dd$county_name[which(dd$k_vs > .75)]
gov80 <- dd$county_name[which(dd$k_vs > .80)]
gov85 <- dd$county_name[which(dd$k_vs > .85)]
gov90 <- dd$county_name[which(dd$k_vs > .90)]
gov95 <- dd$county_name[which(dd$k_vs > .95)]
opp75 <- dd$county_name[which(dd$o_vs > .75)]
opp80 <- dd$county_name[which(dd$o_vs > .80)]
opp85 <- dd$county_name[which(dd$o_vs > .85)]
opp90 <- dd$county_name[which(dd$o_vs > .90)]
opp95 <- dd$county_name[which(dd$o_vs > .95)]

### create vote share codings
df$govt_stronghold_vs_75 <- ifelse(df$county_name %in% gov75, 1, 0)
df$govt_stronghold_vs_80 <- ifelse(df$county_name %in% gov80, 1, 0)
df$govt_stronghold_vs_85 <- ifelse(df$county_name %in% gov85, 1, 0)
df$govt_stronghold_vs_90 <- ifelse(df$county_name %in% gov90, 1, 0)
df$govt_stronghold_vs_95 <- ifelse(df$county_name %in% gov95, 1, 0)
df$opp_stronghold_vs_75 <- ifelse(df$county_name %in% opp75, 1, 0)
df$opp_stronghold_vs_80 <- ifelse(df$county_name %in% opp80, 1, 0)
df$opp_stronghold_vs_85 <- ifelse(df$county_name %in% opp85, 1, 0)
df$opp_stronghold_vs_90 <- ifelse(df$county_name %in% opp90, 1, 0)
df$opp_stronghold_vs_95 <- ifelse(df$county_name %in% opp95, 1, 0)

### finally, create three-factor variables
df$sh_vs_75 <- ifelse(df$govt_stronghold_vs_75 == 1, "gov", "mixed")
df$sh_vs_75 <- ifelse(df$opp_stronghold_vs_75 == 1, "opp", df$sh_vs_75)
df$sh_vs_80 <- ifelse(df$govt_stronghold_vs_80 == 1, "gov", "mixed")
df$sh_vs_80 <- ifelse(df$opp_stronghold_vs_80 == 1, "opp", df$sh_vs_80)
df$sh_vs_85 <- ifelse(df$govt_stronghold_vs_85 == 1, "gov", "mixed")
df$sh_vs_85 <- ifelse(df$opp_stronghold_vs_85 == 1, "opp", df$sh_vs_85)
df$sh_vs_90 <- ifelse(df$govt_stronghold_vs_90 == 1, "gov", "mixed")
df$sh_vs_90 <- ifelse(df$opp_stronghold_vs_90 == 1, "opp", df$sh_vs_90)
df$sh_vs_95 <- ifelse(df$govt_stronghold_vs_95 == 1, "gov", "mixed")
df$sh_vs_95 <- ifelse(df$opp_stronghold_vs_95 == 1, "opp", df$sh_vs_95)

### and do the same for the foregoing definitions
df$sh <- ifelse(df$govt_stronghold == 1, "gov", "mixed")
df$sh <- ifelse(df$opp_stronghold == 1, "opp", df$sh)
df$sh_core <- ifelse(df$govt_stronghold_core == 1, "gov", "mixed")
df$sh_core <- ifelse(df$opp_stronghold_core == 1, "opp", df$sh_core)
df$sh_80 <- ifelse(df$govt_stronghold_80 == 1, "gov", "mixed")
df$sh_80 <- ifelse(df$opp_stronghold_80 == 1, "opp", df$sh_80)
df$sh_core_80 <- ifelse(df$govt_stronghold_core_80 == 1, "gov", "mixed")
df$sh_core_80 <- ifelse(df$opp_stronghold_core_80 == 1, "opp", df$sh_core_80)
df$sh_manual <- ifelse(df$govt_stronghold_manual == 1, "gov", "mixed")
df$sh_manual <- ifelse(df$opp_stronghold_manual == 1, "opp", df$sh_manual)

### now let's make them factors and relevel
df$sh_vs_75 <- factor(df$sh_vs_75, levels = c("mixed", "gov", "opp"))
df$sh_vs_80 <- factor(df$sh_vs_80, levels = c("mixed", "gov", "opp"))
df$sh_vs_85 <- factor(df$sh_vs_85, levels = c("mixed", "gov", "opp"))
df$sh_vs_90 <- factor(df$sh_vs_90, levels = c("mixed", "gov", "opp"))
df$sh_vs_95 <- factor(df$sh_vs_95, levels = c("mixed", "gov", "opp"))
df$sh <- factor(df$sh, levels = c("mixed", "gov", "opp"))
df$sh_core <- factor(df$sh_core, levels = c("mixed", "gov", "opp"))
df$sh_80 <- factor(df$sh_80, levels = c("mixed", "gov", "opp"))
df$sh_core_80 <- factor(df$sh_core_80, levels = c("mixed", "gov", "opp"))
df$sh_manual <- factor(df$sh_manual, levels = c("mixed", "gov", "opp"))

##### CREATE GROUPED VARIABLES #####

df$agent_problem <- ifelse(df$agents == 0 | 
                             df$all_agents_signed == 0 | 
                             df$signed == 0 |
                             df$different_sign == 0,
                           1, 0)
df$document_problem <- ifelse(df$good_scan == 0 | 
                                df$qr_code == 0, 
                              1, 0)
df$procedure_problem <- ifelse(df$first_page_stamped == 0 | 
                                 df$po_signature  == 0, 
                               1, 0)
df$any_problem <- ifelse(df$document_problem == 1 | 
                           df$agent_problem == 1 | 
                           df$procedure_problem == 1 | 
                           df$edited_results == 1, 
                         1, 0)

##### FINAL REFORMATTING #####

### clean up columns
df <- df[order(df$psid, df$psstrid), 
         c("filename", "psid", "psstrid", "county_name", "county_code",  
           "constituency_name",  "constituency_code", "ward_name", "ward_code",
           "polling_station_name", "polling_station_code", "mp_code", 
           "polling_station_type", "polling_station_subtype", "longitude", 
           "latitude", "stream", "data_source", "train", "agents", 
           "all_agents_signed", "different_sign",
           "edited_results", "first_page_stamped", "good_scan", 
           "po_signature", "qr_code", "signed", 
           "prediction_agents", "prediction_all_agents_signed", 
           "prediction_different_sign", 
           "prediction_edited_results", "prediction_first_page_stamped", 
           "prediction_good_scan", "prediction_po_signature", 
           "prediction_qr_code", "prediction_signed", "any_problem", "agent_problem", 
           "document_problem", "procedure_problem", "registered_voters", 
           "registered_voters_filenames", "registered_voters_pvt", 
           "registered_voters_center", "turnout", "turnout_count", 
           "votes_kenyatta",  "votes_odinga", "votes_dida", "votes_karua", 
           "votes_kenneth", "votes_kiyiapi", "votes_mudavadi", "votes_muite", 
           "vote_margin", "vote_margin_abs", "vote_share_kenyatta", 
           "vote_share_odinga", "vote_share_other", "ballots_cast", 
           "ballots_spoiled", "ballots_disputed", "ballots_rejected", 
           "ballots_objected", "ballots_valid", "error_valid", 
           "error_cast", "problem", "pvt_observed", "pvt_obs_no_over", 
           "pvt_ipw", "pvt_sipw", "pop_density", "poverty_rate", "literacy_rate", 
           "dependency_ratio", "skilled_birth_att", "terrain_elevation", 
           "terrain_slope", "terrain_ruggedness", "night_lights", 
           "isolation_simple", "isolation_doogan_rv", "isolation_doogan_popden", 
           "prop_kalenjin", "prop_kamba", "prop_kikuyu", "prop_kisii", 
           "prop_luhya", "prop_luo", "prop_masai", "prop_meru", 
           "prop_mijikenda", "prop_pokot", "prop_somali", "prop_turkana", 
           "ethnic_frac", "county_oversampled", 
           "govt_stronghold", "opp_stronghold", 
           "govt_stronghold_80", "opp_stronghold_80", 
           "govt_stronghold_core", "opp_stronghold_core",
           "govt_stronghold_core_80", "opp_stronghold_core_80",
           "govt_stronghold_manual", "opp_stronghold_manual",
           "govt_stronghold_vs_75", "govt_stronghold_vs_80",
           "govt_stronghold_vs_85", "govt_stronghold_vs_90",
           "govt_stronghold_vs_95", "opp_stronghold_vs_75", 
           "opp_stronghold_vs_80", "opp_stronghold_vs_85", 
           "opp_stronghold_vs_90", "opp_stronghold_vs_95",
           "sh_vs_75", "sh_vs_80", "sh_vs_85", "sh_vs_90", "sh_vs_95",
           "sh", "sh_core", "sh_80", "sh_core_80", "sh_manual")]

##### EXPORT #####

### export the data
write.csv(df, "/data/final-for-analysis.csv", row.names = F)

### trim to necessary variables
df <- df[,c("county_code", "county_name", "constituency_code", 
            "polling_station_code", "stream", "psstrid", "agents", 
            "all_agents_signed", "different_sign", "first_page_stamped", 
            "good_scan", "po_signature", "qr_code", "signed", 
            "document_problem", "procedure_problem", "agent_problem", 
            "edited_results", "any_problem", "turnout", "vote_share_kenyatta", 
            "vote_margin_abs", "sh_vs_80", "sh_vs_75", "sh_vs_85", "sh_80", 
            "pop_density", "terrain_ruggedness", "ethnic_frac",
            "isolation_simple", "poverty_rate", "literacy_rate", "night_lights",
            "pvt_observed", "pvt_ipw", "registered_voters_center")]

### export the replication version
write.csv(df, '/data/final-for-analysis-trimmed.csv', row.names = F)

### delete everything
rm(list = ls())

##### CLEAN THE DATA ON TYPES OF EDITS MADE #####

### read in data
ef1 <- read_excel("/data/edits/edited_results_recheck.xlsx", sheet = 1)
ef2 <- read_excel("/data/edits/edited_results_recheck.xlsx", sheet = 2)
ef3 <- read_excel("/data/edits/edited_results_recheck.xlsx", sheet = 3)
ef <- rbind(ef1, ef2, ef3)
ef <- as.data.frame(ef)
rm(ef1, ef2, ef3)

### clean the data
# change names
names(ef) <- c("file", "polling_station_name", "polling_station_code",
               "stream", "constituency_name", "constituency_code", "stage",
               "registered", "spoiled", "cast", "rejected", "disputed",
               "objected", "valid_votes", "comments")
# fix a few classes
ef$registered <- as.numeric(ef$registered) # ignore warning
ef$cast <- as.numeric(ef$cast) # ignore warning
# harmonize stage variable codings
ef$stage[which(ef$stage == "ORIGINAL" | ef$stage == "ORIGNAL")] <- "Original"
ef$stage[which(ef$stage == "FINAL")] <- "Final"

### reshape
orig <- ef[which(ef$stage == "Original"),]
orig <- orig[,c("file", "registered", "spoiled", "cast", "rejected", 
  "disputed", "objected", "valid_votes", "comments")]
names(orig) <- paste(names(orig),"orig", sep = "_")
fin <- ef[which(ef$stage == "Final"),]
fin <- fin[,c("file", "registered", "spoiled", "cast", "rejected", "disputed",
 "objected", "valid_votes", "comments")]
names(fin) <- paste(names(fin), "fin", sep = "_")
# remerge
ef <- merge(orig, fin, by.x = "file_orig", by.y = "file_fin", all = T)

### clean the original values to fix NAs (which mean no change)
ef$registered_orig <- ifelse(is.na(ef$registered_orig), ef$registered_fin, 
                             ef$registered_orig)
ef$spoiled_orig <- ifelse(is.na(ef$spoiled_orig), ef$spoiled_fin, 
                             ef$spoiled_orig)
ef$cast_orig <- ifelse(is.na(ef$cast_orig), ef$cast_fin, 
                             ef$cast_orig)
ef$rejected_orig <- ifelse(is.na(ef$rejected_orig), ef$rejected_fin, 
                             ef$rejected_orig)
ef$disputed_orig <- ifelse(is.na(ef$disputed_orig), ef$disputed_fin, 
                             ef$disputed_orig)
ef$objected_orig <- ifelse(is.na(ef$objected_orig), ef$objected_fin, 
                             ef$objected_orig)
ef$valid_votes_orig <- ifelse(is.na(ef$valid_votes_orig), ef$valid_votes_fin, 
                             ef$valid_votes_orig)
### set -9999 to NA now
ef[ef == -9999] <- NA

### now compute differences
ef$registered_diff <- ef$registered_fin - ef$registered_orig
ef$spoiled_diff <- ef$spoiled_fin - ef$spoiled_orig
ef$cast_diff <- ef$cast_fin - ef$cast_orig
ef$rejected_diff <- ef$rejected_fin - ef$rejected_orig
ef$disputed_diff <- ef$disputed_fin - ef$disputed_orig
ef$objected_diff <- ef$objected_fin - ef$objected_orig
ef$valid_votes_diff <- ef$valid_votes_fin - ef$valid_votes_orig

### export
write.csv(ef, "/data/final-audit-edited-tallies.csv", row.names = F)

### clean up
rm(list=ls())

##### VERSION CONTROL #####
sessionInfo()
# R version 4.1.1 (2021-08-10)
# Platform: x86_64-pc-linux-gnu (64-bit)
# Running under: Ubuntu 18.04.5 LTS
# 
# Matrix products: default
# BLAS:   /usr/lib/x86_64-linux-gnu/blas/libblas.so.3.7.1
# LAPACK: /usr/lib/x86_64-linux-gnu/lapack/liblapack.so.3.7.1
# 
# locale:
# [1] C
# 
# attached base packages:
# [1] stats     graphics  grDevices utils     datasets  methods   base     
# 
# other attached packages:
# [1] sp_1.4-5        sandwich_3.0-1  rdrobust_1.0.5  patchwork_1.1.1
# [5] miceadds_3.11-6 mice_3.13.0     MASS_7.3-54     ggplot2_3.3.5  
# 
# loaded via a namespace (and not attached):
# [1] Rcpp_1.0.7       pillar_1.6.2     compiler_4.1.1   tools_4.1.1     
# [5] digest_0.6.27    lifecycle_1.0.0  tibble_3.1.4     gtable_0.3.0    
# [9] lattice_0.20-44  pkgconfig_2.0.3  rlang_0.4.11     DBI_1.1.1       
# [13] withr_2.4.2      dplyr_1.0.7      generics_0.1.0   vctrs_0.3.8     
# [17] mitools_2.4      grid_4.1.1       tidyselect_1.1.1 glue_1.4.2      
# [21] R6_2.5.1         fansi_0.5.0      tidyr_1.1.3      purrr_0.3.4     
# [25] farver_2.1.0     magrittr_2.0.1   backports_1.2.1  scales_1.1.1    
# [29] ellipsis_0.3.2   colorspace_2.0-2 labeling_0.4.2   utf8_1.2.2      
# [33] munsell_0.5.0    broom_0.7.9      crayon_1.4.1     zoo_1.8-9
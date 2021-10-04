#!/usr/bin/env Rscript
# analyze_results.R
# Conducts all statistical analysis and creates all figures and tables
# Zach Warner
# 30 August 2021

##### SET UP ######

### load packages
library(ggplot2); library(MASS); library(miceadds); library(patchwork) 
library(rdrobust); library(sandwich); library(sp)

### set seed
set.seed(8675309) # hey jenny

### read in data
df <- read.csv('/data/final-for-analysis-trimmed.csv', stringsAsFactors = F)

##### ELECTION OVERVIEW -- IN INTRODUCTION SECTION TEXT #####

### NB: The 50.51% is from the official, published results.

### print the global turnout and margins
median(df$turnout, na.rm = T)
median(df$vote_margin_abs, na.rm = T)

##### FIGURE 1 #####

### Figure 1 is just a pdf embedded in the TeX file.

##### FIGURE 2 #####

### Figure 2 is drawn directly in Tikz in the TeX file.

##### TABLE 1 #####

### load in the F1s and extract them
nms <- list.files(path = "/results/fit", pattern = "classification_report",
                  full.names = T)
f1s <- lapply(nms, FUN = function(x){
  x <- read.csv(x)
  x$value[which(x$X == "f1")]
})
f1s <- unlist(f1s)

### clean the names a little
nms <- gsub(".*/classification_report_", "", nms)
nms <- gsub(".csv", "", nms, fixed = T)
vec_switch <- Vectorize(vectorize.args = "x", FUN = function(x) {
  switch(x, "agents" = "No agents listed", 
         "all_agents_signed" = "Any agent did not sign",
         "different_sign" = "Agent signatures appear identical",
         "edited_results" = "Results edited",
         "first_page_stamped" = "Form not stamped",
         "good_scan" = "Poor scan quality",
         "po_signature" = "Presiding officer did not sign",
         "qr_code" = "QR code missing",
         "signed" = "No agents signed")})
nms <- unname(vec_switch(nms))

### Get the minority proportions
minority_prop <- c(table(df$agents)[1]/length(na.omit(df$agents)),
                   table(df$all_agents_signed)[1]/length(na.omit(df$all_agents_signed)),
                   table(df$different_sign)[1]/length(na.omit(df$different_sign)),
                   table(df$edited_results)[2]/length(na.omit(df$edited_results)),
                   table(df$first_page_stamped)[1]/length(na.omit(df$first_page_stamped)),
                   table(df$good_scan)[1]/length(na.omit(df$good_scan)),
                   table(df$po_signature)[1]/length(na.omit(df$po_signature)),
                   table(df$qr_code)[1]/length(na.omit(df$qr_code)),
                   table(df$signed)[1]/length(na.omit(df$signed)))

### set up a table
tab <- data.frame(name = nms, prop = round(minority_prop, 3), 
                  f1 = round(f1s, 3))

### reorder to match the discussion in text
tab$order <- c(5, 6, 8, 9, 3, 2, 4, 1, 7)
tab <- tab[order(tab$order),]
tab$order <- NULL

### save it
write.csv(tab, "/results/table-1.csv", row.names = F)

### clean up
rm(list=ls()[!(ls() %in% c("df"))])

##### CANTU F1 -- IN METHODS SECTION AND APPENDIX #####

### Cantu's approximate F1
# use reported values to back out approximate true/false positives/negatives
n_samp <- 20
n_img <- 150
tp <- .85*n_samp*n_img
fp <- .07*n_samp*n_img
tn <- .93*n_samp*n_img
fn <- .15*n_samp*n_img

### compute precision and recall
prec <- tp/(tp + fp)
rec <- tp/(tp + fn)

### compute Cantu approximate f1
f1 <- 2*(prec*rec)/(prec + rec)
f1

# now let's back out how many mispredicted cases this would produce
pc_misclass <- ((fp + fn)/6000)*100
pc_misclass

### compare to our edited results model
# read in the confusion matrix for edited results
us <- read.csv("/results/fit/confusion_matrix_edited_results.csv", 
               stringsAsFactors = F)

### extract the numbers from the matrix
fp_us <- us$Real.0[which(us$X == "Predicted 1")]
fn_us <- us$Real.1[which(us$X == "Predicted 0")]
pc_misclass_us <- ((fp_us + fn_us)/500)*100
pc_misclass_us

### percentage reduction over the Cantu numbers
(pc_misclass - pc_misclass_us)/pc_misclass

### clean up
rm(list=ls()[!(ls() %in% c("df"))])

##### FIGURE 3 #####

### make sure our stronghold variable is factored with "mixed" the baseline
df$sh_vs_80 <- factor(df$sh_vs_80, levels = c("mixed", "gov", "opp"))

### document problems
m_doc <- glm.cluster(df, document_problem ~ sh_vs_80 + pop_density + 
               terrain_ruggedness + ethnic_frac + isolation_simple + 
               poverty_rate + literacy_rate + night_lights +
               as.factor(constituency_code), cluster = "constituency_code")

### procedure problems
m_proc <- glm.cluster(df, procedure_problem ~ sh_vs_80 + pop_density + 
                terrain_ruggedness + ethnic_frac + isolation_simple + 
                poverty_rate + literacy_rate + night_lights +
                as.factor(constituency_code), cluster = "constituency_code")

### agent problems
m_ag <- glm.cluster(df, agent_problem ~ sh_vs_80 + pop_density + 
              terrain_ruggedness + ethnic_frac + isolation_simple + 
              poverty_rate + literacy_rate + night_lights +
              as.factor(constituency_code), cluster = "constituency_code")

### edited results
m_ed <- glm.cluster(df, edited_results ~ sh_vs_80 + pop_density + 
              terrain_ruggedness + ethnic_frac + isolation_simple + 
              poverty_rate + literacy_rate + night_lights +
              as.factor(constituency_code), cluster = "constituency_code")

### get them into shape for plotting
dd <- data.frame(mod = c("Document", "Document", "Procedure", "Procedure",
                         "Agent", "Agent", "Edit", "Edit"),
                 sh = rep(c("gov", "opp"), 4),
                 est = c(coef(m_doc)[2:3],
                         coef(m_proc)[2:3],
                         coef(m_ag)[2:3],
                         coef(m_ed)[2:3]),
                 se = c(sqrt(diag(vcov(m_doc))[2:3]),
                        sqrt(diag(vcov(m_proc))[2:3]),
                        sqrt(diag(vcov(m_ag))[2:3]),
                        sqrt(diag(vcov(m_ed))[2:3])))
# create hi-lo
dd$lo <- dd$est - qnorm(.975)*dd$se
dd$hi <- dd$est + qnorm(.975)*dd$se
# set x-axis locations
dd$loc <- sort(rep(c(1:4), 2))
dd$loc <- ifelse(dd$sh == "gov", dd$loc - .1, dd$loc + .1)

### plot the estimates
p3 <- ggplot(dd) +
  theme_minimal() +
  theme(panel.grid.minor = element_blank(),
        axis.text = element_text(color = "black")) +
  geom_hline(yintercept = 0, color = 'grey60', linetype = "longdash") +
  geom_linerange(aes(x = loc, ymin = lo, ymax =  hi),
                 show.legend = F) +
  geom_point(aes(x = loc, y = est, shape = factor(sh)),
             show.legend = F) +
  scale_shape_manual(values = c("gov" = 15, "opp" = 19)) +
  scale_x_continuous(breaks = c(1:4),
                     labels = c("Document problem", "Procedure problem",
                                "Agent problem", "Edited results")) +
  labs(y = "Predicted effect", x = NULL, title = NULL)

ggsave(p3, filename = '/results/figure-3.pdf', height = 4, width = 6)

##### EFFECT SIZE INTERPRTATION -- IN RESULTS SECTION TEXT #####

### effect size for procedure problems in opposition strongholds
# baseline, competitive areas
coef(m_proc)[1]
# coefficient
coef(m_proc)[3]
# percentage increase
(coef(m_proc)[3])/coef(m_proc)[1]

### effect size for edited results in opposition strongholds
# baseline, competitive areas
coef(m_ed)[1]
# coefficient
coef(m_ed)[3]
# percentage decrease
1 - (coef(m_ed)[1] + coef(m_ed)[3])/coef(m_ed)[1]

### clean up
rm(list=ls()[!(ls() %in% c("m_ag", "m_doc", "m_ed", "m_proc", "df"))])

##### TABLE 2 #####

### create subsets
dgov <- df[which(df$sh_vs_80 == "gov"),]
dopp <- df[which(df$sh_vs_80 == "opp"),]
dmix <- df[which(df$sh_vs_80 == "mixed"),]

### Turnout models, all polling stations
# procedure problem
m1 <- glm.cluster(df, turnout ~ procedure_problem + pop_density + 
                    terrain_ruggedness + ethnic_frac + isolation_simple + 
                    poverty_rate + night_lights + literacy_rate + 
                    as.factor(constituency_code), 
                  cluster = "constituency_code")
# results edited
m2 <- glm.cluster(df, turnout ~ edited_results + pop_density + 
                    terrain_ruggedness + ethnic_frac + isolation_simple + 
                    poverty_rate + night_lights + literacy_rate + 
                    as.factor(constituency_code), 
                  cluster = "constituency_code")

### Kenyatta vote share models, all polling stations
# procedure problem
m3 <- glm.cluster(df, vote_share_kenyatta ~ procedure_problem + pop_density + 
                    terrain_ruggedness + ethnic_frac + isolation_simple + 
                    poverty_rate + night_lights + literacy_rate + 
                    as.factor(constituency_code), 
                  cluster = "constituency_code")
# results edited
m4 <- glm.cluster(df, vote_share_kenyatta ~ edited_results + pop_density + 
                    terrain_ruggedness + ethnic_frac + isolation_simple + 
                    poverty_rate + night_lights + literacy_rate + 
                    as.factor(constituency_code), 
                  cluster = "constituency_code")

### Absolute margin models, all polling stations
# procedure problem
m5 <- glm.cluster(df, vote_margin_abs ~ procedure_problem + pop_density + 
                    terrain_ruggedness + ethnic_frac + isolation_simple + 
                    poverty_rate + night_lights + literacy_rate + 
                    as.factor(constituency_code), 
                  cluster = "constituency_code")
# results edited
m6 <- glm.cluster(df, vote_margin_abs ~ edited_results + pop_density + 
                    terrain_ruggedness + ethnic_frac + isolation_simple + 
                    poverty_rate + night_lights + literacy_rate + 
                    as.factor(constituency_code), 
                  cluster = "constituency_code")

### Turnout models, government strongholds
# procedure problem
m7 <- glm.cluster(dgov, turnout ~ procedure_problem + pop_density + 
                    terrain_ruggedness + ethnic_frac + isolation_simple + 
                    poverty_rate + night_lights + literacy_rate + 
                    as.factor(constituency_code), 
                  cluster = "constituency_code")
# results edited
m8 <- glm.cluster(dgov, turnout ~ edited_results + pop_density + 
                    terrain_ruggedness + ethnic_frac + isolation_simple + 
                    poverty_rate + night_lights + literacy_rate + 
                    as.factor(constituency_code), 
                  cluster = "constituency_code")

### Kenyatta vote share models, government strongholds
# procedure problem
m9 <- glm.cluster(dgov, vote_share_kenyatta ~ procedure_problem + pop_density + 
                    terrain_ruggedness + ethnic_frac + isolation_simple + 
                    poverty_rate + night_lights + literacy_rate + 
                    as.factor(constituency_code), 
                  cluster = "constituency_code")
# results edited
m10 <- glm.cluster(dgov, vote_share_kenyatta ~ edited_results + pop_density + 
                    terrain_ruggedness + ethnic_frac + isolation_simple + 
                    poverty_rate + night_lights + literacy_rate + 
                    as.factor(constituency_code), 
                  cluster = "constituency_code")

### Absolute margin models, government strongholds
# procedure problem
m11 <- glm.cluster(dgov, vote_margin_abs ~ procedure_problem + pop_density + 
              terrain_ruggedness + ethnic_frac + isolation_simple + 
              poverty_rate + night_lights + literacy_rate + 
              as.factor(constituency_code), 
            cluster = "constituency_code")
# results edited
m12 <- glm.cluster(dgov, vote_margin_abs ~ edited_results + pop_density + 
              terrain_ruggedness + ethnic_frac + isolation_simple + 
              poverty_rate + night_lights + literacy_rate + 
              as.factor(constituency_code), 
            cluster = "constituency_code")

### Turnout models, opposition strongholds
# procedure problem
m13 <- glm.cluster(dopp, turnout ~ procedure_problem + pop_density + 
              terrain_ruggedness + ethnic_frac + isolation_simple + 
              poverty_rate + night_lights + literacy_rate + 
              as.factor(constituency_code), 
            cluster = "constituency_code")
# results edited
m14 <- glm.cluster(dopp, turnout ~ edited_results + pop_density + 
              terrain_ruggedness + ethnic_frac + isolation_simple + 
              poverty_rate + night_lights + literacy_rate + 
              as.factor(constituency_code), 
            cluster = "constituency_code")

### Kenyatta vote share models, opposition strongholds
# procedure problem
m15 <- glm.cluster(dopp, vote_share_kenyatta ~ procedure_problem + pop_density + 
              terrain_ruggedness + ethnic_frac + isolation_simple + 
              poverty_rate + night_lights + literacy_rate + 
              as.factor(constituency_code), 
            cluster = "constituency_code")
# results edited
m16 <- glm.cluster(dopp, vote_share_kenyatta ~ edited_results + pop_density + 
              terrain_ruggedness + ethnic_frac + isolation_simple + 
              poverty_rate + night_lights + literacy_rate + 
              as.factor(constituency_code), 
            cluster = "constituency_code")

### Absolute margin models, opposition strongholds
# procedure problem
m17 <- glm.cluster(dopp, vote_margin_abs ~ procedure_problem + pop_density + 
              terrain_ruggedness + ethnic_frac + isolation_simple + 
              poverty_rate + night_lights + literacy_rate + 
              as.factor(constituency_code), 
            cluster = "constituency_code")
# results edited
m18 <- glm.cluster(dopp, vote_margin_abs ~ edited_results + pop_density + 
              terrain_ruggedness + ethnic_frac + isolation_simple + 
              poverty_rate + night_lights + literacy_rate + 
              as.factor(constituency_code), 
            cluster = "constituency_code")

### organize into the table
# store all the values
tab <- data.frame(matrix(nrow = 18, ncol = 5))
names(tab) <- c("irr", "dv", "sh", "est", "sig")
tab$irr <- rep(c("Procedure problem", "Edited results"), 9)
tab$dv <- rep(c(rep("Turnout", 2), rep("Kenyatta VS", 2), rep("Abs. margin", 2)), 3)
tab$sh <- c(rep("all", 6), rep("gov", 6), rep("opp", 6))
for(i in 1:nrow(tab)){
  eval(parse(text = paste("tab$est[", i, "] <- coef(m", i, ")[2]", sep = "")))
  eval(parse(text = paste("tab$sig[", i, "] <- summary(m", i ,
                          ")[2,4]", sep = "")))
}
# round, convert to characters, add text decoration, remove unnecessary vars
tab$est <- ifelse(tab$sig < .05, paste(sprintf("%.2f", tab$est), "*", sep = ""),
                  sprintf("%.2f", tab$est))
tab$sig <- NULL
# reshape and fix the column names
tab <- reshape(tab, idvar = c("irr", "sh"), timevar = "dv", direction = "wide")
names(tab) <- gsub("est.", "", names(tab))
# save
write.csv(tab, "/results/table-2.csv", row.names = F)

##### EFFECT SIZE INTERPRETATION -- IN RESULTS SECTION TEXT #####

### interpretation of effect size in opposition areas, edited result, abs margin
# coefficient
coef(m2)[2]
# baseline
coef(m2)[1]
# effect size
1 - (coef(m2)[1] + coef(m2)[2])/coef(m2)[1]

### clean up
rm(list=ls()[!(ls() %in% c("m_ag", "m_doc", "m_ed", "m_proc", "df", "dgov", 
                           "dopp", "dmix"))])

##### TABLE 3 #####

### document problems, gov sample
m1 <- glm.cluster(dgov, document_problem ~ as.factor(pvt_observed) + 
                    pop_density + terrain_ruggedness + ethnic_frac + 
                    isolation_simple + poverty_rate + night_lights + 
                    literacy_rate + as.factor(constituency_code), 
                  cluster = "constituency_code", weights = dgov$pvt_ipw)

### procedure problems, gov sample
m2 <- glm.cluster(dgov, procedure_problem ~ as.factor(pvt_observed) + 
                    pop_density + terrain_ruggedness + ethnic_frac + 
                    isolation_simple + poverty_rate + night_lights + 
                    literacy_rate + as.factor(constituency_code), 
                  cluster = "constituency_code", weights = dgov$pvt_ipw)

### agent problems, gov sample
m3 <- glm.cluster(dgov, agent_problem ~ as.factor(pvt_observed) + 
                    pop_density + terrain_ruggedness + ethnic_frac + 
                    isolation_simple + poverty_rate + night_lights + 
                    literacy_rate + as.factor(constituency_code), 
                  cluster = "constituency_code", weights = dgov$pvt_ipw)

### edited results, gov sample
m4 <- glm.cluster(dgov, edited_results ~ as.factor(pvt_observed) + 
                    pop_density + terrain_ruggedness + ethnic_frac + 
                    isolation_simple + poverty_rate + night_lights + 
                    literacy_rate + as.factor(constituency_code), 
                  cluster = "constituency_code", weights = dgov$pvt_ipw)

### any problem, gov sample
m5 <- glm.cluster(dgov, any_problem ~ as.factor(pvt_observed) + 
                    pop_density + terrain_ruggedness + ethnic_frac + 
                    isolation_simple + poverty_rate + night_lights + 
                    literacy_rate + as.factor(constituency_code), 
                  cluster = "constituency_code", weights = dgov$pvt_ipw)

### document problems, opp sample
m6 <- glm.cluster(dopp, document_problem ~ as.factor(pvt_observed) + 
                    pop_density + terrain_ruggedness + ethnic_frac + 
                    isolation_simple + poverty_rate + night_lights + 
                    literacy_rate + as.factor(constituency_code), 
                  cluster = "constituency_code", weights = dopp$pvt_ipw)

### procedure problems, opp sample
m7 <- glm.cluster(dopp, procedure_problem ~ as.factor(pvt_observed) + 
                    pop_density + terrain_ruggedness + ethnic_frac + 
                    isolation_simple + poverty_rate + night_lights + 
                    literacy_rate + as.factor(constituency_code), 
                  cluster = "constituency_code", weights = dopp$pvt_ipw)

### agent problems, opp sample
m8 <- glm.cluster(dopp, agent_problem ~ as.factor(pvt_observed) + 
                    pop_density + terrain_ruggedness + ethnic_frac + 
                    isolation_simple + poverty_rate + night_lights + 
                    literacy_rate + as.factor(constituency_code), 
                  cluster = "constituency_code", weights = dopp$pvt_ipw)

### edited results, opp sample
m9 <- glm.cluster(dopp, edited_results ~ as.factor(pvt_observed) + 
                    pop_density + terrain_ruggedness + ethnic_frac + 
                    isolation_simple + poverty_rate + night_lights + 
                    literacy_rate + as.factor(constituency_code), 
                  cluster = "constituency_code", weights = dopp$pvt_ipw)

### any problem, opp sample
m10 <- glm.cluster(dopp, any_problem ~ as.factor(pvt_observed) + 
                     pop_density + terrain_ruggedness + ethnic_frac + 
                     isolation_simple + poverty_rate + night_lights + 
                     literacy_rate + as.factor(constituency_code), 
                   cluster = "constituency_code", weights = dopp$pvt_ipw)

### document problems, mix sample
m11 <- glm.cluster(dmix, document_problem ~ as.factor(pvt_observed) + 
                     pop_density + terrain_ruggedness + ethnic_frac + 
                     isolation_simple + poverty_rate + night_lights + 
                     literacy_rate + as.factor(constituency_code), 
                   cluster = "constituency_code", weights = dmix$pvt_ipw)

### procedure problems, mix sample
m12 <- glm.cluster(dmix, procedure_problem ~ as.factor(pvt_observed) + 
                     pop_density + terrain_ruggedness + ethnic_frac + 
                     isolation_simple + poverty_rate + night_lights + 
                     literacy_rate + as.factor(constituency_code), 
                   cluster = "constituency_code", weights = dmix$pvt_ipw)

### agent problems, mix sample
m13 <- glm.cluster(dmix, agent_problem ~ as.factor(pvt_observed) + 
                     pop_density + terrain_ruggedness + ethnic_frac + 
                     isolation_simple + poverty_rate + night_lights + 
                     literacy_rate + as.factor(constituency_code), 
                   cluster = "constituency_code", weights = dmix$pvt_ipw)

### edited results, mixed sample
m14 <- glm.cluster(dmix, edited_results ~ as.factor(pvt_observed) + 
                     pop_density + terrain_ruggedness + ethnic_frac + 
                     isolation_simple + poverty_rate + night_lights + 
                     literacy_rate + as.factor(constituency_code), 
                   cluster = "constituency_code", weights = dmix$pvt_ipw)

### any problem, mixed sample
m15 <- glm.cluster(dmix, any_problem ~ as.factor(pvt_observed) + 
                     pop_density + terrain_ruggedness + ethnic_frac + 
                     isolation_simple + poverty_rate + night_lights + 
                     literacy_rate + as.factor(constituency_code), 
                   cluster = "constituency_code", weights = dmix$pvt_ipw)

### document problems, all sample
m16 <- glm.cluster(df, document_problem ~ as.factor(pvt_observed) + 
                     pop_density + terrain_ruggedness + ethnic_frac + 
                     isolation_simple + poverty_rate + night_lights + 
                     literacy_rate + as.factor(constituency_code), 
                   cluster = "constituency_code", weights = df$pvt_ipw)

### procedure problems, all sample
m17 <- glm.cluster(df, procedure_problem ~ as.factor(pvt_observed) + 
                     pop_density + terrain_ruggedness + ethnic_frac + 
                     isolation_simple + poverty_rate + night_lights + 
                     literacy_rate + as.factor(constituency_code), 
                   cluster = "constituency_code", weights = df$pvt_ipw)

### agent problems, all sample
m18 <- glm.cluster(df, agent_problem ~ as.factor(pvt_observed) + 
                     pop_density + terrain_ruggedness + ethnic_frac + 
                     isolation_simple + poverty_rate + night_lights + 
                     literacy_rate + as.factor(constituency_code), 
                   cluster = "constituency_code", weights = df$pvt_ipw)

### edited results, all sample
m19 <- glm.cluster(df, edited_results ~ as.factor(pvt_observed) + 
                     pop_density + terrain_ruggedness + ethnic_frac + 
                     isolation_simple + poverty_rate + night_lights + 
                     literacy_rate + as.factor(constituency_code), 
                   cluster = "constituency_code", weights = df$pvt_ipw)

### any problem, all sample
m20 <- glm.cluster(df, any_problem ~ as.factor(pvt_observed) + 
                     pop_density + terrain_ruggedness + ethnic_frac + 
                     isolation_simple + poverty_rate + night_lights + 
                     literacy_rate + as.factor(constituency_code), 
                   cluster = "constituency_code", weights = df$pvt_ipw)

### get the table together
tab <- data.frame(matrix(nrow = 5, ncol = 5))
names(tab) <- c("Irregularity", "Gov. strongholds", "Opp. strongholds", "Competitive", "All")
tab$Irregularity <- c("Document problems", "Procedure problems", "Agent problems",
                      "Edited results", "Any problem")
for(i in 1:20){
  # extract the estimate and significance of each model
  tmp <- eval(parse(text = paste("m", i, sep = "")))
  ef <- sprintf("%.2f", coef(tmp)[2])
  p <- summary(tmp)[2,4] # 2nd row is pvt, 4th col is p value
  # drop it in the table
  if(i %in% c(1:5)){
    tab[i,2] <- ifelse(p < .05, paste(ef, "*", sep = ""), ef)
  }
  if(i %in% c(6:10)){
    tab[(i-5),3] <- ifelse(p < .05, paste(ef, "*", sep = ""), ef)
  }
  if(i %in% c(11:15)){
    tab[(i-10),4] <- ifelse(p < .05, paste(ef, "*", sep = ""), ef)
  }
  if(i %in% c(16:20)){
    tab[(i-15),5] <- ifelse(p < .05, paste(ef, "*", sep = ""), ef)
  }
}

### save it
write.csv(tab, "/results/table-3.csv", row.names = F)

### save the full results for the appendix table A12 (government strongholds)
tab_a12 <- data.frame(rbind(summary(m1)[2:9,], summary(m2)[2:9,],
                           summary(m3)[2:9,], summary(m4)[2:9,],
                           summary(m5)[2:9,]))
# clean up
tab_a12$cov <- row.names(tab_a12); row.names(tab_a12) <- NULL
tab_a12$cov <- gsub("\\..*", "", tab_a12$cov)
tab_a12$cov[which(tab_a12$cov == "as")] <- "observed"
names(tab_a12)[1:4] <- c("est", "se", "t", "p")
# add model information
tab_a12$mod <- c(rep("Document problem", 8), rep("Procedure problem", 8),
             rep("Agent problem", 8), rep("Results edited", 8), 
             rep("Any problem", 8))
# round
tab_a12$est <- ifelse(tab_a12$p < .05, paste(sprintf("%.2f", tab_a12$est), 
                                             "*", sep = ""),
                      sprintf("%.2f", tab_a12$est))
tab_a12$se <- sprintf("%.2f", tab_a12$se)
# reorder and drop unnecessary columns
tab_a12 <- tab_a12[,c("mod", "cov", "est", "se")]
# save it - slightly reshaped in the appendix
write.csv(tab_a12, "/results/table-a12.csv", row.names = F)

### do the same for Table A13 (opposition strongholds)
tab_a13 <- data.frame(rbind(summary(m6)[2:9,], summary(m7)[2:9,],
                            summary(m8)[2:9,], summary(m9)[2:9,],
                            summary(m10)[2:9,]))
tab_a13$cov <- row.names(tab_a13); row.names(tab_a13) <- NULL
tab_a13$cov <- gsub("\\..*", "", tab_a13$cov)
tab_a13$cov[which(tab_a13$cov == "as")] <- "observed"
names(tab_a13)[1:4] <- c("est", "se", "t", "p")
tab_a13$mod <- c(rep("Document problem", 8), rep("Procedure problem", 8),
                 rep("Agent problem", 8), rep("Results edited", 8), 
                 rep("Any problem", 8))
tab_a13$est <- ifelse(tab_a13$p < .05, paste(sprintf("%.2f", tab_a13$est), 
                                             "*", sep = ""),
                      sprintf("%.2f", tab_a13$est))
tab_a13$se <- sprintf("%.2f", tab_a13$se)
tab_a13 <- tab_a13[,c("mod", "cov", "est", "se")]
write.csv(tab_a13, "/results/table-a13.csv", row.names = F)

### do the same for Table A14 (competitive areas)
tab_a14 <- data.frame(rbind(summary(m11)[2:9,], summary(m12)[2:9,],
                            summary(m13)[2:9,], summary(m14)[2:9,],
                            summary(m15)[2:9,]))
tab_a14$cov <- row.names(tab_a14); row.names(tab_a14) <- NULL
tab_a14$cov <- gsub("\\..*", "", tab_a14$cov)
tab_a14$cov[which(tab_a14$cov == "as")] <- "observed"
names(tab_a14)[1:4] <- c("est", "se", "t", "p")
tab_a14$mod <- c(rep("Document problem", 8), rep("Procedure problem", 8),
                 rep("Agent problem", 8), rep("Results edited", 8), 
                 rep("Any problem", 8))
tab_a14$est <- ifelse(tab_a14$p < .05, paste(sprintf("%.2f", tab_a14$est), 
                                             "*", sep = ""),
                      sprintf("%.2f", tab_a14$est))
tab_a14$se <- sprintf("%.2f", tab_a14$se)
tab_a14 <- tab_a14[,c("mod", "cov", "est", "se")]
write.csv(tab_a14, "/results/table-a14.csv", row.names = F)

### do the same for Table A15 (all areas)
tab_a15 <- data.frame(rbind(summary(m16)[2:9,], summary(m17)[2:9,],
                            summary(m18)[2:9,], summary(m19)[2:9,],
                            summary(m20)[2:9,]))
tab_a15$cov <- row.names(tab_a15); row.names(tab_a15) <- NULL
tab_a15$cov <- gsub("\\..*", "", tab_a15$cov)
tab_a15$cov[which(tab_a15$cov == "as")] <- "observed"
names(tab_a15)[1:4] <- c("est", "se", "t", "p")
tab_a15$mod <- c(rep("Document problem", 8), rep("Procedure problem", 8),
                 rep("Agent problem", 8), rep("Results edited", 8), 
                 rep("Any problem", 8))
tab_a15$est <- ifelse(tab_a15$p < .05, paste(sprintf("%.2f", tab_a15$est), 
                                             "*", sep = ""),
                      sprintf("%.2f", tab_a15$est))
tab_a15$se <- sprintf("%.2f", tab_a15$se)
tab_a15 <- tab_a15[,c("mod", "cov", "est", "se")]
write.csv(tab_a15, "/results/table-a15.csv", row.names = F)

##### EFFECT SIZE INTERPRETATION -- IN RESULTS SECTION TEXT #####

### how many document problems, government strongholds, by observation type
nrow(dgov[which(dgov$document_problem == 1),])

### clean up
rm(list=ls()[!(ls() %in% c("m_ag", "m_doc", "m_ed", "m_proc", "df"))])

##### ANALYSIS OF EDITS MADE -- IN RESULTS SECTION TEXT #####

### read in data
ef <- read.csv('/data/final-audit-edited-tallies.csv', stringsAsFactors = F)

### analysis
# median values
median(ef$registered_diff, na.rm = T)
median(ef$spoiled_diff, na.rm = T)
median(ef$cast_diff, na.rm = T)
median(ef$rejected_diff, na.rm = T)
median(ef$disputed_diff, na.rm = T)
median(ef$objected_diff, na.rm = T)
median(ef$valid_votes_diff, na.rm = T)
# global mean value
mean(c(ef$registered_diff, ef$spoiled_diff, ef$cast_diff, ef$rejected_diff,
       ef$disputed_diff, ef$objected_diff, ef$valid_votes_diff), na.rm = T)
# how many unique problematic forms are there?
probs <- sort(unique(c(ef$file_orig[which(ef$registered_diff > 10)],
                       ef$file_orig[which(ef$spoiled_diff > 10)],
                       ef$file_orig[which(ef$cast_diff > 10)],
                       ef$file_orig[which(ef$rejected_diff > 10)],
                       ef$file_orig[which(ef$disputed_diff > 10)],
                       ef$file_orig[which(ef$objected_diff > 10)],
                       ef$file_orig[which(ef$valid_votes_diff > 10)])))
length(probs)
# which ones do we need to inspect?
probs

##### FIGURE 4 #####

### Figure 4 is a screenshot from the first of these, 001-004-0016-004-5.pdf

### clean up
rm(list=ls()[!(ls() %in% c("m_ag", "m_doc", "m_ed", "m_proc", "df"))])

##### APPENDIX TABLE A1 #####

### This is coded directly in TeX from the output of our hyperparameter tuning.

##### APPENDIX TABLES A2-A10 #####

### These tables are imported directly from the confusion_matrix_* files in
### the /results/fit/ folder, and are produced by the python scripts that
### fit the deep neural networks.

##### APPENDIX  FIGURE A1 #####

### estimate null effects
null_agent <- glm(agent_problem ~ as.factor(constituency_code), data = df)
null_doc <- glm(document_problem ~ as.factor(constituency_code), data = df)
null_proc <- glm(procedure_problem ~ as.factor(constituency_code), data = df)
null_edit <- glm(edited_results ~ as.factor(constituency_code), data = df)

### estimate FE with SEs via MASS
agent_fe <- coef(null_agent); agent_vc <- vcov(null_agent)
doc_fe <- coef(null_doc); doc_vc <- vcov(null_doc)
proc_fe <- coef(null_proc); proc_vc <- vcov(null_proc)
edit_fe <- coef(null_edit); edit_vc <- vcov(null_edit)

### draw 1,000 values from the fitted model parameters
bhat_agent <- data.frame(mvrnorm(1000, agent_fe, agent_vc))
bhat_doc <- data.frame(mvrnorm(1000, doc_fe, doc_vc))
bhat_proc <- data.frame(mvrnorm(1000, proc_fe, proc_vc))
bhat_edit <- data.frame(mvrnorm(1000, edit_fe, edit_vc))

### cleanup
rm(null_agent, null_doc, null_proc, null_edit, agent_fe, doc_fe, proc_fe, 
   edit_fe, agent_vc, doc_vc, proc_vc, edit_vc)

### fix names
names(bhat_agent) <- gsub("as.factor.constituency_code.", "", names(bhat_agent),
                          fixed = T)
names(bhat_doc) <- gsub("as.factor.constituency_code.", "", names(bhat_doc),
                        fixed = T)
names(bhat_proc) <- gsub("as.factor.constituency_code.", "", names(bhat_proc),
                         fixed = T)
names(bhat_edit) <- gsub("as.factor.constituency_code.", "", names(bhat_edit),
                         fixed = T)

### reshape
df_agent <- data.frame(fe = names(bhat_agent),
                       est = apply(bhat_agent, 2, mean),
                       lo = apply(bhat_agent, 2, quantile, 0.025),
                       hi = apply(bhat_agent, 2, quantile, 0.975),
                       dv = "agent")
df_doc <- data.frame(fe = names(bhat_doc),
                     est = apply(bhat_doc, 2, mean),
                     lo = apply(bhat_doc, 2, quantile, 0.025),
                     hi = apply(bhat_doc, 2, quantile, 0.975),
                     dv = "doc")
df_proc <- data.frame(fe = names(bhat_proc),
                      est = apply(bhat_proc, 2, mean),
                      lo = apply(bhat_proc, 2, quantile, 0.025),
                      hi = apply(bhat_proc, 2, quantile, 0.975),
                      dv = "proc")
df_edit <- data.frame(fe = names(bhat_edit),
                      est = apply(bhat_edit, 2, mean),
                      lo = apply(bhat_edit, 2, quantile, 0.025),
                      hi = apply(bhat_edit, 2, quantile, 0.975),
                      dv = "edit")

### merge and clean up
dd <- rbind(df_agent, df_doc, df_proc, df_edit)
dd <- dd[-which(grepl("Intercept", dd$fe)),]
rm(bhat_agent, bhat_doc, bhat_proc, bhat_edit, df_agent, df_doc, df_proc, 
   df_edit)

### extract global means
agent_mean <- mean(dd$est[which(dd$dv == "agent")])
doc_mean <- mean(dd$est[which(dd$dv == "doc")])
proc_mean <- mean(dd$est[which(dd$dv == "proc")])
edit_mean <- mean(dd$est[which(dd$dv == "edit")])

### get county data - omit a few non-matches
const <- na.omit(unique(df[c("constituency_code", "county_name")]))

### add a color to keep it consistent
gg_color_hue <- function(n) {
  hues = seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}
counties <- sort(unique(const$county_name))
colrs <- gg_color_hue(length(counties))
const$color <- NA
for(i in 1:length(counties)){
  const$color[which(const$county_name == counties[i])] <- 
    rep(colrs[i], nrow(const[which(const$county_name == counties[i]),]))
}
names(colrs) <- counties

### merge with county data
dd <- merge(dd, const, by.x = "fe", by.y = "constituency_code", all = T)

### sort within category
dd <- split(dd, f = dd$dv)
dd <- lapply(dd, FUN = function(x){
  x <- x[order(x$county_name, x$est),]
  x$loc <- nrow(x):1
  x
})
dd <- do.call(rbind, dd)

### split into four dataframes just to make ggplot easier
agent_fe <- dd[which(dd$dv == "agent"),]
doc_fe <- dd[which(dd$dv == "doc"),]
proc_fe <- dd[which(dd$dv == "proc"),]
edit_fe <- dd[which(dd$dv == "edit"),]

### now plot
# agent problems
locs <- aggregate(loc ~ county_name + color, agent_fe, mean)
locs$county_name <- tools::toTitleCase(tolower(locs$county_name))
locs$county_name <- gsub(" - ", "/", locs$county_name)
p_agent <- ggplot(agent_fe, aes(color = county_name)) + 
  theme_minimal() + 
  theme(panel.grid.major.y = element_blank(),
        panel.grid.minor.y = element_blank(),
        panel.grid.minor.x = element_blank(),
        axis.text.y = element_text(color = locs$col)) +
  geom_vline(xintercept = agent_mean, color = 'grey60', linetype = "longdash") + 
  geom_linerange(aes(y = loc, xmin = lo, xmax =  hi), show.legend = F) +
  geom_point(aes(y = loc, x = est), size = 0.7, show.legend = F) +
  scale_y_continuous(breaks = locs$loc, labels = locs$county_name, 
                     expand = c(0.01, 0.01)) +
  coord_cartesian(xlim = c(0,1)) +
  labs(x = "Constituency mean", y = NULL, title = "Agent problems")

# document problems
locs <- aggregate(loc ~ county_name + color, doc_fe, mean)
locs$county_name <- tools::toTitleCase(tolower(locs$county_name))
locs$county_name <- gsub(" - ", "/", locs$county_name)
p_doc <- ggplot(doc_fe, aes(color = county_name)) + 
  theme_minimal() + 
  theme(panel.grid.major.y = element_blank(),
        panel.grid.minor.y = element_blank(),
        panel.grid.minor.x = element_blank(),
        axis.text.y = element_text(color = locs$col)) +
  geom_vline(xintercept = doc_mean, color = 'grey60', linetype = "longdash") + 
  geom_linerange(aes(y = loc, xmin = lo, xmax =  hi), show.legend = F) +
  geom_point(aes(y = loc, x = est), size = 0.7, show.legend = F) +
  scale_y_continuous(breaks = locs$loc, labels = locs$county_name, 
                     expand = c(0.01, 0.01)) +
  coord_cartesian(xlim = c(0,1)) +
  labs(x = "Constituency mean", y = NULL, title = "Document problems")

# procedure problems
locs <- aggregate(loc ~ county_name + color, proc_fe, mean)
locs$county_name <- tools::toTitleCase(tolower(locs$county_name))
locs$county_name <- gsub(" - ", "/", locs$county_name)
p_proc <- ggplot(proc_fe, aes(color = county_name)) + 
  theme_minimal() + 
  theme(panel.grid.major.y = element_blank(),
        panel.grid.minor.y = element_blank(),
        panel.grid.minor.x = element_blank(),
        axis.text.y = element_text(color = locs$col)) +
  geom_vline(xintercept = proc_mean, color = 'grey60', linetype = "longdash") + 
  geom_linerange(aes(y = loc, xmin = lo, xmax =  hi), show.legend = F) +
  geom_point(aes(y = loc, x = est), size = 0.7, show.legend = F) +
  scale_y_continuous(breaks = locs$loc, labels = locs$county_name, 
                     expand = c(0.01, 0.01)) +
  coord_cartesian(xlim = c(0,1)) +
  labs(x = "Constituency mean", y = NULL, title = "Procedure problems")

# edited results
locs <- aggregate(loc ~ county_name + color, edit_fe, mean)
locs$county_name <- tools::toTitleCase(tolower(locs$county_name))
locs$county_name <- gsub(" - ", "/", locs$county_name)
p_edit <- ggplot(edit_fe, aes(color = county_name)) + 
  theme_minimal() + 
  theme(panel.grid.major.y = element_blank(),
        panel.grid.minor.y = element_blank(),
        panel.grid.minor.x = element_blank(),
        axis.text.y = element_text(color = locs$col)) +
  geom_vline(xintercept = edit_mean, color = 'grey60', linetype = "longdash") + 
  geom_linerange(aes(y = loc, xmin = lo, xmax =  hi), show.legend = F) +
  geom_point(aes(y = loc, x = est), size = 0.7, show.legend = F) +
  scale_y_continuous(breaks = locs$loc, labels = locs$county_name, 
                     expand = c(0.01, 0.01)) +
  coord_cartesian(xlim = c(0,1)) +
  labs(x = "Constituency mean", y = NULL, title = "Edited results")

### save
p_out <- wrap_plots(p_doc, p_proc, p_agent, p_edit, ncol = 4)
ggsave(p_out, filename = '/results/figure-a1.pdf', height = 10, width = 12)

### clean up
rm(list=ls()[!(ls() %in% c("m_ag", "m_doc", "m_ed", "m_proc", "df"))])

##### APPENDIX TABLE A11 #####

### bind the estimate rows together
tab <- data.frame(rbind(summary(m_doc)[2:10,], summary(m_proc)[2:10,],
                        summary(m_ag)[2:10,], summary(m_ed)[2:10,]))

### clean up
# fix names
tab$cov <- row.names(tab); row.names(tab) <- NULL
tab$cov <- gsub("\\..*", "", tab$cov)
names(tab)[1:4] <- c("est", "se", "t", "p")
# add model information
tab$mod <- c(rep("Document problem", 9), rep("Procedure problem", 9),
             rep("Agent problem", 9), rep("Results edited", 9))
# round
tab$est <- ifelse(tab$p < .05, paste(sprintf("%.2f", tab$est), "*", sep = ""),
                  sprintf("%.2f", tab$est))
tab$se <- sprintf("%.2f", tab$se)
# reorder and drop unnecessary columns
tab <- tab[,c("mod", "cov", "est", "se")]

### save it - slightly reshaped in the appendix
write.csv(tab, "/results/table-a11.csv", row.names = F)

### clean up
rm(list=ls()[!(ls() %in% c("df"))])

##### APPENDIX FIGURE A2 #####

### make sure our stronghold variable is factored with "mixed" the baseline
df$sh_vs_75 <- factor(df$sh_vs_75, levels = c("mixed", "gov", "opp"))

### document problems
m_doc2 <- glm.cluster(df, document_problem ~ sh_vs_75 + pop_density + 
                        terrain_ruggedness + ethnic_frac + isolation_simple + 
                        poverty_rate + literacy_rate + night_lights +
                        as.factor(constituency_code), cluster = "constituency_code")

### procedure problems
m_proc2 <- glm.cluster(df, procedure_problem ~ sh_vs_75 + pop_density + 
                         terrain_ruggedness + ethnic_frac + isolation_simple + 
                         poverty_rate + literacy_rate + night_lights +
                         as.factor(constituency_code), cluster = "constituency_code")

### agent problems
m_ag2 <- glm.cluster(df, agent_problem ~ sh_vs_75 + pop_density + 
                       terrain_ruggedness + ethnic_frac + isolation_simple + 
                       poverty_rate + literacy_rate + night_lights +
                       as.factor(constituency_code), cluster = "constituency_code")

### edited results
m_ed2 <- glm.cluster(df, edited_results ~ sh_vs_75 + pop_density + 
                       terrain_ruggedness + ethnic_frac + isolation_simple + 
                       poverty_rate + literacy_rate + night_lights +
                       as.factor(constituency_code), cluster = "constituency_code")

### get them into shape for plotting
dd <- data.frame(mod = c("Document", "Document", "Procedure", "Procedure",
                         "Agent", "Agent", "Edit", "Edit"),
                 sh = rep(c("gov", "opp"), 4),
                 est = c(coef(m_doc2)[2:3],
                         coef(m_proc2)[2:3],
                         coef(m_ag2)[2:3],
                         coef(m_ed2)[2:3]),
                 se = c(sqrt(diag(vcov(m_doc2))[2:3]),
                        sqrt(diag(vcov(m_proc2))[2:3]),
                        sqrt(diag(vcov(m_ag2))[2:3]),
                        sqrt(diag(vcov(m_ed2))[2:3])))
# create hi-lo
dd$lo <- dd$est - qnorm(.975)*dd$se
dd$hi <- dd$est + qnorm(.975)*dd$se
# set x-axis locations
dd$loc <- sort(rep(c(1:4), 2))
dd$loc <- ifelse(dd$sh == "gov", dd$loc - .1, dd$loc + .1)

### plot the estimates
pa2 <- ggplot(dd) +
  theme_minimal() +
  theme(panel.grid.minor = element_blank(),
        axis.text = element_text(color = "black")) +
  geom_hline(yintercept = 0, color = 'grey60', linetype = "longdash") +
  geom_linerange(aes(x = loc, ymin = lo, ymax =  hi),
                 show.legend = F) +
  geom_point(aes(x = loc, y = est, shape = factor(sh)),
             show.legend = F) +
  scale_shape_manual(values = c("gov" = 15, "opp" = 19)) +
  scale_x_continuous(breaks = c(1:4),
                     labels = c("Document problem", "Procedure problem",
                                "Agent problem", "Edited results")) +
  labs(y = "Predicted effect", x = NULL, title = NULL)

ggsave(pa2, filename = '/results/figure-a2.pdf', height = 4, width = 6)

### clean up
rm(list=ls()[!(ls() %in% c("m_ag", "m_doc", "m_ed", "m_proc", "df"))])

##### APPENDIX FIGURE A3 #####

### make sure our stronghold variable is factored with "mixed" the baseline
df$sh_vs_85 <- factor(df$sh_vs_85, levels = c("mixed", "gov", "opp"))

### document problems
m_doc <- glm.cluster(df, document_problem ~ sh_vs_85 + pop_density + 
                       terrain_ruggedness + ethnic_frac + isolation_simple + 
                       poverty_rate + literacy_rate + night_lights +
                       as.factor(constituency_code), cluster = "constituency_code")

### procedure problems
m_proc <- glm.cluster(df, procedure_problem ~ sh_vs_85 + pop_density + 
                        terrain_ruggedness + ethnic_frac + isolation_simple + 
                        poverty_rate + literacy_rate + night_lights +
                        as.factor(constituency_code), cluster = "constituency_code")

### agent problems
m_ag <- glm.cluster(df, agent_problem ~ sh_vs_85 + pop_density + 
                      terrain_ruggedness + ethnic_frac + isolation_simple + 
                      poverty_rate + literacy_rate + night_lights +
                      as.factor(constituency_code), cluster = "constituency_code")

### edited results
m_ed <- glm.cluster(df, edited_results ~ sh_vs_85 + pop_density + 
                      terrain_ruggedness + ethnic_frac + isolation_simple + 
                      poverty_rate + literacy_rate + night_lights +
                      as.factor(constituency_code), cluster = "constituency_code")

### get them into shape for plotting
dd <- data.frame(mod = c("Document", "Document", "Procedure", "Procedure",
                         "Agent", "Agent", "Edit", "Edit"),
                 sh = rep(c("gov", "opp"), 4),
                 est = c(coef(m_doc)[2:3],
                         coef(m_proc)[2:3],
                         coef(m_ag)[2:3],
                         coef(m_ed)[2:3]),
                 se = c(sqrt(diag(vcov(m_doc))[2:3]),
                        sqrt(diag(vcov(m_proc))[2:3]),
                        sqrt(diag(vcov(m_ag))[2:3]),
                        sqrt(diag(vcov(m_ed))[2:3])))
# create hi-lo
dd$lo <- dd$est - qnorm(.975)*dd$se
dd$hi <- dd$est + qnorm(.975)*dd$se
# set x-axis locations
dd$loc <- sort(rep(c(1:4), 2))
dd$loc <- ifelse(dd$sh == "gov", dd$loc - .1, dd$loc + .1)

### plot the estimates
pa3 <- ggplot(dd) +
  theme_minimal() +
  theme(panel.grid.minor = element_blank(),
        axis.text = element_text(color = "black")) +
  geom_hline(yintercept = 0, color = 'grey60', linetype = "longdash") +
  geom_linerange(aes(x = loc, ymin = lo, ymax =  hi),
                 show.legend = F) +
  geom_point(aes(x = loc, y = est, shape = factor(sh)),
             show.legend = F) +
  scale_shape_manual(values = c("gov" = 15, "opp" = 19)) +
  scale_x_continuous(breaks = c(1:4),
                     labels = c("Document problem", "Procedure problem",
                                "Agent problem", "Edited results")) +
  labs(y = "Predicted effect", x = NULL, title = NULL)

ggsave(pa3, filename = '/results/figure-a3.pdf', height = 4, width = 6)

##### APPENDIX FIGURE A4 #####

### make sure our stronghold variable is factored with "mixed" the baseline
df$sh_80 <- factor(df$sh_80, levels = c("mixed", "gov", "opp"))

### document problems
m_doc <- glm.cluster(df, document_problem ~ sh_80 + pop_density + 
                       terrain_ruggedness + ethnic_frac + isolation_simple + 
                       poverty_rate + literacy_rate + night_lights +
                       as.factor(constituency_code), cluster = "constituency_code")

### procedure problems
m_proc <- glm.cluster(df, procedure_problem ~ sh_80 + pop_density + 
                        terrain_ruggedness + ethnic_frac + isolation_simple + 
                        poverty_rate + literacy_rate + night_lights +
                        as.factor(constituency_code), cluster = "constituency_code")

### agent problems
m_ag <- glm.cluster(df, agent_problem ~ sh_80 + pop_density + 
                      terrain_ruggedness + ethnic_frac + isolation_simple + 
                      poverty_rate + literacy_rate + night_lights +
                      as.factor(constituency_code), cluster = "constituency_code")

### edited results
m_ed <- glm.cluster(df, edited_results ~ sh_80 + pop_density + 
                      terrain_ruggedness + ethnic_frac + isolation_simple + 
                      poverty_rate + literacy_rate + night_lights +
                      as.factor(constituency_code), cluster = "constituency_code")

### get them into shape for plotting
dd <- data.frame(mod = c("Document", "Document", "Procedure", "Procedure",
                         "Agent", "Agent", "Edit", "Edit"),
                 sh = rep(c("gov", "opp"), 4),
                 est = c(coef(m_doc)[2:3],
                         coef(m_proc)[2:3],
                         coef(m_ag)[2:3],
                         coef(m_ed)[2:3]),
                 se = c(sqrt(diag(vcov(m_doc))[2:3]),
                        sqrt(diag(vcov(m_proc))[2:3]),
                        sqrt(diag(vcov(m_ag))[2:3]),
                        sqrt(diag(vcov(m_ed))[2:3])))
# create hi-lo
dd$lo <- dd$est - qnorm(.975)*dd$se
dd$hi <- dd$est + qnorm(.975)*dd$se
# set x-axis locations
dd$loc <- sort(rep(c(1:4), 2))
dd$loc <- ifelse(dd$sh == "gov", dd$loc - .1, dd$loc + .1)

### plot the estimates
pa4 <- ggplot(dd) +
  theme_minimal() +
  theme(panel.grid.minor = element_blank(),
        axis.text = element_text(color = "black")) +
  geom_hline(yintercept = 0, color = 'grey60', linetype = "longdash") +
  geom_linerange(aes(x = loc, ymin = lo, ymax =  hi),
                 show.legend = F) +
  geom_point(aes(x = loc, y = est, shape = factor(sh)),
             show.legend = F) +
  scale_shape_manual(values = c("gov" = 15, "opp" = 19)) +
  scale_x_continuous(breaks = c(1:4),
                     labels = c("Document problem", "Procedure problem",
                                "Agent problem", "Edited results")) +
  labs(y = "Predicted effect", x = NULL, title = NULL)

ggsave(pa4, filename = '/results/figure-a4.pdf', height = 4, width = 6)

### clean up
rm(list=ls()[!(ls() %in% c("df"))])

##### TABLES A12-A15 #####

### these are saved above in the code for TABLE 3 (at the end)

##### FIGURE A5 #####

### subset to cases where there were 2 or fewer streams
rd_df <- df[which(df$registered_voters_center <= 1200),]

### run the models
# document problem
m_doc <- rdrobust(y = rd_df$document_problem, 
                  x = rd_df$registered_voters_center, 
                  c = 800.5, bwselect = 'certwo')
# procedure problem
m_proc <- rdrobust(y = rd_df$procedure_problem, 
                   x = rd_df$registered_voters_center, 
                   c = 800.5, bwselect = 'certwo')
# agent problem
m_ag <- rdrobust(y = rd_df$agent_problem, 
                 x = rd_df$registered_voters_center, 
                 c = 800.5, bwselect = 'certwo')
# edited results
m_edit <- rdrobust(y = rd_df$edited_results, 
                   x = rd_df$registered_voters_center, 
                   c = 800.5, bwselect = 'certwo')

### reshape the results
res <- list("Document problems" = m_doc, 
            "Procedure problems" = m_proc, 
            "Agent problems" = m_ag, 
            "Edited results" = m_edit)
res <- lapply(res, FUN = function(x){
  x <- data.frame(coef = x$Estimate[1, 2], se = x$Estimate[1,4])
  x
})
res <- mapply(cbind, res, "outcome" = names(res), SIMPLIFY = F)
res <- do.call(rbind, res)

### create confidence intervals
res$lo95 <- res$coef - res$se*qnorm(0.975)
res$lo90 <- res$coef - res$se*qnorm(0.950)
res$hi95 <- res$coef + res$se*qnorm(0.975)
res$hi90 <- res$coef + res$se*qnorm(0.950)

### create the x location
res$outcome <- factor(res$outcome, 
                      levels = c('Document problems', 'Procedure problems', 
                                 'Agent problems', 'Edited results'))

### plot the results
pa5 <- ggplot(res) +
  theme_minimal() +
  theme(panel.grid.minor = element_blank(),
        axis.text = element_text(color = "black")) +
  geom_hline(yintercept = 0, color = 'grey60', linetype = "longdash") +
  geom_linerange(aes(x = outcome, ymin = lo90, ymax =  hi90), size = 1,
                 show.legend = F) +
  geom_linerange(aes(x = outcome, ymin = lo95, ymax =  hi95), size = 0.6,
                 show.legend = F) +
  geom_point(aes(x = outcome, y = coef), show.legend = F) +
  coord_flip() +
  labs(y = "Regression discontinuity effect", x = NULL, title = NULL)

### save the plot
ggsave(pa5, filename = '/results/figure-a5.pdf', height = 4, width = 6)

### clean up
rm(list=ls()[!(ls() %in% c("df"))])

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
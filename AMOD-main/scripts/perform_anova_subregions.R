#Factorial repeated-measures ANOVA: within subjects difference of performance by amygdala subregion (CM, SF, AStr, LB)

#----Set the working directory------
setwd("/Users/gracejang/Desktop/encoding_model_performance/amygdala_subregions_encoding_performance")

#----Install Packages-----
library(compute.es)
library(ez)
library(ggplot2)
library(multcomp)
library(nlme)
library(pastecs)
library(reshape)
library(emmeans)

library(dplyr)
library(ggpubr)
library(olsrr)
library(lmtest)

#----access data----
subregions_data<-read.delim("amygdala_subregions_avg_atanh_LONG.csv",',', header = TRUE)
subregions_data$subject<-factor(subregions_data$subject) #set subject as a factor
subregions_data$amy_subregion<-factor(subregions_data$amy_subregion) #set subregion as a factor

head(subregions_data)

#specify contrasts
contrasts(subregions_data$amy_subregion) <-cbind(c(-1, -1, 3, -1), c(-1, 2, 0, -1), c(-1, 0, 0, 1)) #contrast 1: LB vs others, contrast 2: CM vs others, contrast 3: SF vs AStr 

#Using lme()
baseline<-lme(performance_atanh ~ 1, random = ~1|subject, na.action = na.omit, data = subregions_data, method = "ML")
fullModel<-update(baseline, .~. + amy_subregion)

#comparing models with anova()
anova(baseline, fullModel)

#result summary of model fitting for final model
summary(fullModel)

#posthoc tests
postHocs<-glht(fullModel, linfct = mcp(amy_subregion = "Tukey"))
summary(postHocs)
confint(postHocs)
---
  title: 'Improving Olfactory Assessment: An Item Response Theory Analysis of the American
  English version of the Sniffin’ Sticks Identification subtest'
author: Eva Tolomeo, Leognano Ceraudo, Ryann Kolb, Pamela H. Dalton, Marco Tullio
Liuzza, Valentina Parma

---
  
  ```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
library(psych)
library(tidyverse)
library(ltm)
library(DescTools)
library(RcmdrMisc)
library(psych)
library(ppcor)
library(car)
library(lmtest)
library(MuMIn)
library(lavaan)
library(semPlot)
library(semTools)
library(polycor)
library(REdaS)
library(eRm)
library(patchwork)
library(FMP)
library(readr)
```

## Load the two Dataframes

#We encoded the same dataset in two distinct ways: 
#the first (IRT) is suited for IRT and DIF analyses, where the scores of the Sniffin' Stick Identification (ID) subtest can take a value of 0 (incorrect response) or 1 (correct response). 
#The second (IRT_NRM) is designed for polytomous IRT analysis, in which the scores of ID subtest can take a value of 4 if the response is correct, 
#and a value of 1, 2 or 3 for the distractors.
#Item scores will be re-coded in r for a better stability and interpretation of the Nominal Response Model: 
#Least chosen distractor= 1
#intermediate chosen distractor= 2
#most frequent distractor = 3
#correct option = 4


```{r}
## Dataset for IRT and DIF analyses

IRT<-IRT_LABELED_DICHOTOMOUS<- read_csv("IRT_LABELED_DICHOTOMOUS.csv")
IRT<-na.omit(IRT)
View(IRT)

## Dataset for Polynomial IRT
IRT_NRM<-IRT_NRM <- read_csv("IRT_NRM.csv")
IRT_NRM<-na.omit(IRT_NRM)
IRT_NRM<-IRT_NRM[,4:19]
IRT_NRM[IRT_NRM == 1] <- 5
IRT_NRM[IRT_NRM==4] <- 1#Least Chosen category (distractor)
IRT_NRM[IRT_NRM==2] <- 6
IRT_NRM[IRT_NRM==3] <- 7
IRT_NRM[IRT_NRM==6] <- 3#The most frequent response category (distractor)
IRT_NRM[IRT_NRM==7] <- 2#Intermediate response category (distractor)
IRT_NRM[IRT_NRM == 5] <- 4# Correct response category
View(IRT_NRM)
```

#Descriptive Analyses: Sex and Age Participants

```{r}
#Sex
with(IRT,Desc(sex)) 
#Age
with(IRT, Desc(age))
```

#Data Analysis

#Testing IRT assumption: Unidimensionality
```{r}
irtOI<-IRT[, c(4:19)]# create a subset with ID items only
View(irtOI)
#CFA
UNI_oi<- 'OI =~ orange + leather + cinnamon + peppermint + banana + lemon + licorice + turpentine + garlic + coffee + apple + clove + pineapple + rose + anise + fish
          '
CFA_OI<- cfa(UNI_oi, ordered = TRUE, data = irtOI)
summary(CFA_OI, standardize= T, ci= T)
# peppermint estimates negative variance: -0.015. Heywood Case

##heywood case cause: sampling fluctuactions?
# Sub-sample trial (70% of the sample) for the Heywood Case
set.seed(123)
subsample <- sample(1:nrow(irtOI), size=round(0.7*nrow(irtOI)))
fit_subsample <- cfa(CFA_OI, data=irtOI[subsample,])
summary(fit_subsample, ci= T, standardize= T)#check for negative variances# no more negative variances
subsample

#fit indexes of the unidimensional CFA
fitMeasures(CFA_OI, c("chisq.scaled", "df.scaled", "pvalue.scaled", 
                      "cfi.scaled", "tli.scaled","rmsea.ci.lower.scaled", "rmsea.scaled","rmsea.ci.upper.scaled",
                      "srmr"))
#accettable fit indexes, but DWLS estimator returns optimistics indexes

#Unidim Index
irtOI_noRowNames <- unname(irtOI)  #create a new df to remove raws and columnes
u<-psych::unidim(irtOI_noRowNames, cor = "tet", nfactors = 1)#tetrachoric because of the dichotomus nature of ID items
print(u)
#u index ranges from 0 to 1. A value of .7 means average unidimensionality

```

#Testing IRT assumption: Local Independence

```{r}
library(mirt)
model <- mirt(irtOI, 1, itemtype = "2PL")  # Create a 2PL undimensional model with the irtOI df
residuals <- residuals(model, type="Q3")# compute the model residuals. Look at Q3 summary statistics


residuals_matrix<-cor.plot(residuals, upper = FALSE)# create a correlation residual matrix plot
# and check for |Q3| > 0.2

cor_matrix <- residuals 

# Check for item pairs |Q3|  => 0.2 
which(abs(cor_matrix) > 0.2 & lower.tri(cor_matrix), arr.ind = TRUE)

# The Q3 statistic (Yen, 1984) yielded the following item pairs' residual correlation values: M = −0.046, Mdn = −0.047, range [−0.20, 0.11]. Only one item pair, rose-peppermint (r = −0.21), met the residual correlation threshold of |.2| proposed by Chen and Thiessen (1997, pp. 265–289). 

```

# Testing IRT assumption: Monotonicity

```{r}
library(mokken)
#Compute Monotonicity 
monotonicity_result <- check.monotonicity(irtOI)
monotonicity_result$violations #It detects the name of items that violates monotonicity. It returns NULL when no items violate the assumption.

```

# IRT Model Fitting and Comparison
```{r}
# IRT 1PL Model
model_1pl_OI <- ltm::rasch(irtOI)
summary(model_1pl_OI)

#IRT 2PL Model
model_2pl_OI <- ltm::ltm(irtOI~ z1)
summary(model_2pl_OI)

#compare IRT Models
compare_irt_model<-anova(model_1pl_OI, model_2pl_OI)
print(compare_irt_model)

#2PL model seems to fit better to the data

# Item Characteristic Curves of 2PL model

plot(model_2pl_OI, type = "ICC", cex = 0.9, cex.axis = 1.1, cex.lab = 1.2, lwd= 1.5,xlab= "Θ", y = "P (Θ)", main= "2PL Item Characteristic Curve", legend = T, cex.legend = 0.6)

# Graphics Customization
old_par <- par(cex = 0.8, cex.axis = 1.1, cex.lab = 1.5)


# Plot ICC
plot(model_2pl_OI, 
     type = "ICC", 
     lwd = 2.5,
     xlab = "Θ", 
     ylab = "P(Θ)", 
     main = "2PL Item Characteristic Curve", 
     legend = TRUE)

# previous graphics
par(old_par)
```

#Test Response Function 
```{r}
library(ggplot2)

# TRF function
trf <- function(theta_values) {
  # 2PL model parameters
  b <- c(-1.8232, -0.8245, -0.6988, -1.8111, -0.9871, -0.3105, 
         -1.3617, 0.2075, -2.0011, -1.7708, 0.1538, -1.6116, 
         -0.4793, -1.6818, -0.8284, -1.4741)
  
  a <- c(0.8968, 0.3094, 0.9544, 3.8759, 1.2337, 0.6425, 
         1.3468, 0.4583, 1.0846, 1.8761, 0.8925, 1.3864, 
         0.4004, 1.5827, 1.2630, 1.8282)
  
  results <- sapply(theta_values, function(theta) {
    probs <- 1 / (1 + exp(-a * (theta - b)))
    sum(probs)
  })
  
  return(data.frame(theta = theta_values, expected_score = results))
}

#Print expected scores in relation to theta
test_values <- c(-4,-3, -2, -1, 0, 1, 2, 3, 4)# theta range
quick_trf <- trf(test_values)
print(quick_trf)

#Plot
theta_smooth <- seq(-4, 4, by = 0.1)
trf_smooth <- trf(theta_smooth)

library(ggplot2)
plot1 <- ggplot(quick_trf, aes(x = theta, y = expected_score)) +
  geom_point(size = 4, color = "darkblue") +
  geom_line(size = 1.2, color = "blue") +
  labs(title = "Test Response Function (TRF)",
       
       x = "θ",
       y = "Expected scores") +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5, size = 12),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10)
  ) +
  scale_x_continuous(breaks = seq(-4, 4, 1)) +
  scale_y_continuous(breaks = seq(0, 16, 1))
print(plot1)

##The most important results of this analysis suggest that: a  value of -2 corresponds approximately to an expected score of 5, a  value of -1 corresponds approximately to an expected score of 9, and a  value of 0 corresponds approximately to an expected score of 12. Given that a total raw score of 8 or lower on the identification subtest indicates functional anosmia, a score between 9 and 11 indicates hyposmia, and a score of 12 or higher indicates normosmia, we can infer with reasonable precision that:  values ≤ -2 correspond to functional anosmia, = -1 corresponds to hyposmia, and  values ≥ 0 corresponds to normosmia. 


```

#Test Information Function
```{r}
model_2pl_OI_mirt<-mirt::mirt(irtOI, model = 1, itemtype = "2PL")#create a 2PL unidimensional model with irtOI df

theta<-seq(-4, 4, by=0.1)
tif_values<-mirt::testinfo(model_2pl_OI_mirt, Theta = theta)
tif_values

#TIF PLOT
ggplot(data.frame(theta = theta, information = tif_values), 
       aes(x = theta, y = information)) +
  geom_line(color = "black", linewidth = 1) +
  labs(x = expression(θ), y = "Test Information", title = "Test Information Function (TIF)") +  
  scale_x_continuous(breaks = seq(-4, 4, by = 1)) +  
  theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 12, face = "bold"),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
  )

##The TIF analysis suggested that the identification subtest shows greater measurement precision and information value within  levels ranging from -2 to -1.

```

#Differential Item Functioning
##To assess item measurement invariance across sex, we conducted a DIF analysis using the Logistic Regression (LR) method (Zumbo, 1999). 
#As stated by Zumbo (1999; p.12), "DIF occurs when examinees from different groups show differing probabilities of success on the item after matching on the underlying ability that the item is intended to measure". 
#For dichotomous items, LR accommodates for both uniform and non-uniform DIF within a unified framework (Swaminathan, 1994). 
#Non-uniform DIF happens when lower-scoring participants are more likely to succeed in the first group, while higher-scoring participants are more likely to succeed in the other group (Zumbo, 1999).

```{r}
library(difR)

IRT<-as.data.frame(IRT)#run this prompt if difLogistic does not work properly

log<-difLogistic(IRT[, c(4:19)], group = IRT$sex, focal.name = "Female", match = IRT$Score, type = "both", p.adjust.method = "bonferroni")
summary(log)

log$names
log$adjusted.p
log$DIFitems

# Create a df with two vectors
dif_table <- data.frame(
  Item = log$names,
  Adjusted_p = log$adjusted.p
)

# Table
print(dif_table)

dif_table$DIF_detected <- ifelse(dif_table$Adjusted_p < 0.05, "Yes", "No")

dif_table$DIF_detected

dif_table1 <- data.frame(
  Item = dif_table$Item,
  Adjusted_p = dif_table$Adjusted_p, DIF= dif_table$DIF_detected
)

print(dif_table1)

##After adjusting p-values using the Bonferroni correction, no items exhibited statistically significant DIF between males and females
```

#Polynomial IRT:(IRT_NRM) is designed for polytomous IRT analyses, in which the scores of the Odor Identification subtest can take a value of 4 if the response is correct, and a value of 1 for the least chosen distractor, 2 for the intermediate chosen distractor, and 3 for the most chosen distractor.

#Note: This coding scheme constrains the correct response to a fixed value of 3, while the rarest distractor is assigned a value of 0, establishing it as the reference category. Consequently, the remaining two distractors are parameterized in relation to this baseline.

```{r}
modello<-mirt::mirt(IRT_NRM, model=1, itemtype = "nominal", verbose= F)

modello

#expected: ak0<ak1<ak2<ak3

coef<-mirt::coef(modello, simplify= T)
coef

# category characteristic curves
library(mirt)

# graphics customization 
plot(modello, 
     type = "trace", 
     which.items = 1:16, 
     facet_items = TRUE, 
     auto.key = list(
       space = "right", 
       cex = 0.9,
       text = c("D1", "D2", "D3", "Correct"),  
       title = "Response Categories",
       border = TRUE
     ),
     layout = c(4, 4),                              
     lwd = 3,                                        
     ,main = "Category Characteristic Curves")




# item Fit NRM
mirt::itemfit(modello) 
#The individual item fit analysis, after applying the Bonferroni adjustment, 
#revealed that four items (e.g, orange; licorice; turpentine; pineapple), albeit non-significantly, 
#showed higher S-x2 statistical values, revealing potential discrepancies between observed 
#and model predicted response patterns and suggesting potential violations of the assumed item response function. 
#The remaining twelve items demonstrated adequate fit with adjusted p-values of 1.00, indicating that their response patterns
#were consistent with model expectations


#Bonferroni

p_values <- c(0.024, 0.333, 0.304, 0.083, 0.931, 0.648, 0.015, 0.030, 0.403, 0.258, 0.175, 0.188, 0.027, 0.056, 0.198, 0.077)
bonferroni<-p.adjust(p_values, method = "bonferroni")

modello_bonferroni<-data.frame(Item = c("orange", "leather", "cinnamon", "peppermint", "banana", "lemon", "licorice", "turpentine", "garlic", "coffee", "apple", "clove", "pineapple", "rose", "anise", "fish"),
                               P_Raw = p_values,
                               P_Bonferroni = bonferroni)

print(modello_bonferroni)

#NRM fit
mirt::M2(modello)

#The M2 test statistic  suggested no severe misfit of the NRM to the data.

```


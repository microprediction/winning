library(tidyverse)

# -------------------------------------------------------------------------------------------------

data <- read.csv2('data_experiment1.csv')
data$Correct = as.logical(data$Correct)
data <- as_tibble(data)

# ----------------------------------------------------------------------------------------------------

###---- Response times by difficulty & amount of alternatives ----####

data %>%
  group_by(Nsujeto, StimVal, Nalternativas) %>% 
  summarize(rt1 = mean(RT_type1)) %>% 
  ggplot(aes(x=as.factor(StimVal),
             y=rt1,
             group=as.factor(Nalternativas)))+
  stat_summary(fun = mean, geom = "line", na.rm=T, lwd=2, 
               col=c(rep('darkgreen',5), rep('orange',5)))+
  stat_summary(geom='errorbar', lwd=2, col=c(rep('darkgreen',5), rep('orange',5)),
               width=0.1)+
  xlab('Task difficulty (stimulus2/stimulus1)')+
  ylab('RT type 1')+
  ggtitle('Response times (type 1 task) by task difficulty & amount of alternatives')+
  theme_classic()+
  coord_cartesian(ylim=c(1000,2500))+
  annotate('text', x=4, y=c(1500, 1380), label=c('2 alternatives','3 alternatives'),
           col=c('darkgreen', 'orange'), size=9)

data %>%
  group_by(Nsujeto, StimVal, Nalternativas) %>% 
  summarize(rtconf = mean(RT_Confidence)) %>% 
  ggplot(aes(x=as.factor(StimVal),
             y=rtconf,
             group=as.factor(Nalternativas)))+
  stat_summary(fun = mean, geom = "line", na.rm=T, lwd=2, 
               col=c(rep('darkgreen',5), rep('orange',5)))+
  stat_summary(geom='errorbar', lwd=2, col=c(rep('darkgreen',5), rep('orange',5)),
               width=0.1)+
  xlab('Task difficulty (stimulus2/stimulus1)')+
  ylab('RT type 2')+
  ggtitle('Response times (type 2 task) by task difficulty & amount of alternatives')+
  theme_classic()+
  coord_cartesian(ylim=c(1000,2500))+
  annotate('text', x=2, y=c(2300, 2180), label=c('2 alternatives','3 alternatives'),
           col=c('darkgreen', 'orange'), size=9)

####---- Performance by difficulty & amount of alternatives ----####
data %>%
  group_by(Nsujeto, StimVal, Nalternativas) %>% 
  summarize(perf = mean(binary_correct)) %>% 
  ggplot(aes(x=as.factor(StimVal),
             y=perf,
             group=as.factor(Nalternativas)))+
  stat_summary(fun = mean, geom = "line", na.rm=T, lwd=2, 
               col=c(rep('darkgreen',5), rep('orange',5)))+
  stat_summary(geom='errorbar', lwd=2, col=c(rep('darkgreen',5), rep('orange',5)),
               width=0.1)+
  xlab('Task difficulty (stimulus2/stimulus1)')+
  ylab('Performance')+
  ggtitle('Performance by task difficulty & amount of alternatives')+
  theme_classic()+
  annotate('text', x=4, y=c(0.95, 0.92), label=c('2 alternatives','3 alternatives'),
           col=c('darkgreen', 'orange'), size=9)


####---- Confidence by difficulty & amount of alternatives ----####
data %>%
  group_by(Nsujeto, StimVal, Nalternativas) %>% 
  summarize(conf = mean(Confidence)) %>% 
  ggplot(aes(x=as.factor(StimVal),
             y=conf,
             group=as.factor(Nalternativas)))+
  stat_summary(fun = mean, geom = "line", na.rm=T, lwd=2, 
               col=c(rep('darkgreen',5), rep('orange',5)))+
  stat_summary(geom='errorbar', lwd=2, col=c(rep('darkgreen',5), rep('orange',5)),
               width=0.1)+
  xlab('Task difficulty (stimulus2/stimulus1)')+
  ylab('Confidence')+
  ggtitle('Confidence by task difficulty & amount of alternatives')+
  theme_classic()+
  coord_cartesian(ylim=c(0.5,0.9))+
  annotate('text', x=4, y=c(0.85, 0.82), label=c('2 alternatives','3 alternatives'),
           col=c('darkgreen', 'orange'), size=9)
  

####---- confidence 3 alternatives - 2 alternatives ----####

# this code can reproduce the subtraction between correct and incorrect trials
# to do that, add at indx3 and indx2:
# subj$binary_correct==0 (inc trials) // subj$binary_correct==1 (cor trials)


levels <- sort(unique(data$StimVal))
nsubj  <- max(data$Nsujeto)
conf_data <- matrix(NA, nsubj, 5) # 5 levels of difficulty

for(i in 1:nsubj){
  subj <- data[data$Nsujeto==i,]
  temp <- c()
  
  
  for(j in levels){
    indx3 <- which(subj$StimVal==j & subj$Nalternativas==3)
    indx2 <- which(subj$StimVal==j & subj$Nalternativas==2)
    temp <- c(temp, mean(subj[indx3,]$Confidence)-mean(subj[indx2,]$Confidence))
  }
  
  
  conf_data[i,] <- temp
  
}

# standard error of mean 
level1 <- mean_se(na.omit(conf_data[,1]))
level2 <- mean_se(na.omit(conf_data[,2]))
level3 <- mean_se(na.omit(conf_data[,3]))
level4 <- mean_se(na.omit(conf_data[,4]))
level5 <- mean_se(na.omit(conf_data[,5]))

# plot
graphdata <- data.frame(names = levels, 
                        values = colMeans(conf_data, na.rm = T))
graphdata %>%
  ggplot(aes(x=as.factor(levels),y=values))+
  geom_col(fill='gray28')+
  geom_errorbar(aes(x=as.factor(levels), ymin=c(level1$ymin, level2$ymin, level3$ymin,
                                                level4$ymin, level5$ymin), 
                    ymax=c(level1$ymax,level2$ymax,level3$ymax, level4$ymax, level5$ymax)),
                width=0.4, size=1.3, color='orange')+
  ylim(c(-0.12,0.10))+
  ggtitle('Confidence (3alternatives-2alternatives)')+
  ylab('Confidence 3alt-2alt')+
  xlab('Task difficulty (stimulus2/stimulus1)')+
  theme_classic()


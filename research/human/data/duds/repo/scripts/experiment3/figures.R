library(tidyverse)

# -----------------------------------------------------------------------------------------------

data <- read.csv2('data_experiment3.csv')
data$Correct = as.logical(data$Correct)
data <- as_tibble(data)

# ----------------------------------------------------------------------------------------------------

###---- Response times by difficulty & amount of alternatives ----####

data %>%
  group_by(Nsujeto, distance_ratio, Nalternativas) %>% 
  summarize(rt1 = mean(RT_type1)) %>% 
  ggplot(aes(x=rev(as.factor(distance_ratio)),
             y=rt1,
             group=as.factor(Nalternativas)))+
  stat_summary(fun = mean, geom = "line", na.rm=T, lwd=2, 
               col=c(rep('darkgreen',3), rep('orange',3)))+
  stat_summary(geom='errorbar', lwd=2, col=c(rep('darkgreen',3), rep('orange',3)),
               width=0.1)+
  scale_x_discrete(name='Task difficulty (distance to closest cloud)', 
                   labels=c('0.33', '0.4', '0.47'))+
  ylab('RT type 1')+
  ggtitle('Response times (type 1 task) by task difficulty & amount of alternatives')+
  theme_classic()+
  coord_cartesian(ylim=c(3500,5500))+
  annotate('text', x=3, y=c(4000, 3850), label=c('2 alternatives','3 alternatives'),
           col=c('darkgreen', 'orange'), size=9)

data %>%
  group_by(Nsujeto, distance_ratio, Nalternativas) %>% 
  summarize(rtconf = mean(RT_Confidence)) %>% 
  ggplot(aes(x=rev(as.factor(distance_ratio)),
             y=rtconf,
             group=as.factor(Nalternativas)))+
  stat_summary(fun = mean, geom = "line", na.rm=T, lwd=2, 
               col=c(rep('darkgreen',3), rep('orange',3)))+
  stat_summary(geom='errorbar', lwd=2, col=c(rep('darkgreen',3), rep('orange',3)),
               width=0.1)+
  scale_x_discrete(name='Task difficulty (distance to closest cloud)', 
                   labels=c('0.33', '0.4', '0.47'))+
  ylab('RT type 2')+
  ggtitle('Response times (type 2 task) by task difficulty & amount of alternatives')+
  theme_classic()+
  coord_cartesian(ylim=c(1000,3000))+
  annotate('text', x=2, y=c(2800, 2650), label=c('2 alternatives','3 alternatives'),
           col=c('darkgreen', 'orange'), size=9)

####---- Performance by difficulty & amount of alternatives ----####
data %>%
  group_by(Nsujeto, distance_ratio, Nalternativas) %>% 
  summarize(perf = mean(binary_correct)) %>% 
  ggplot(aes(x=rev(as.factor(distance_ratio)),
             y=perf,
             group=as.factor(Nalternativas)))+
  stat_summary(fun = mean, geom = "line", na.rm=T, lwd=2, 
               col=c(rep('darkgreen',3), rep('orange',3)))+
  stat_summary(geom='errorbar', lwd=2, col=c(rep('darkgreen',3), rep('orange',3)),
               width=0.1)+
  scale_x_discrete(name='Task difficulty (distance to closest cloud)', 
                   labels=c('0.33', '0.4', '0.47'))+
  ylab('Performance')+
  ggtitle('Performance by task difficulty & amount of alternatives')+
  theme_classic()+
  annotate('text', x=2.5, y=c(0.95, 0.92), label=c('2 alternatives','3 alternatives'),
           col=c('darkgreen', 'orange'), size=9)


####---- Confidence by difficulty & amount of alternatives ----####
data %>%
  group_by(Nsujeto, distance_ratio, Nalternativas) %>% 
  summarize(conf = mean(Confidence)) %>% 
  ggplot(aes(x=rev(as.factor(distance_ratio)),
             y=conf,
             group=as.factor(Nalternativas)))+
  stat_summary(fun = mean, geom = "line", na.rm=T, lwd=2, 
               col=c(rep('darkgreen',3), rep('orange',3)))+
  stat_summary(geom='errorbar', lwd=2, col=c(rep('darkgreen',3), rep('orange',3)),
               width=0.1)+
  scale_x_discrete(name='Task difficulty (distance to closest cloud)', 
                   labels=c('0.33', '0.4', '0.47'))+
  ylab('Confidence')+
  ggtitle('Confidence by task difficulty & amount of alternatives')+
  theme_classic()+
  coord_cartesian(ylim=c(0.5,0.9))+
  annotate('text', x=2, y=c(0.85, 0.82), label=c('2 alternatives','3 alternatives'),
           col=c('darkgreen', 'orange'), size=9)


####---- confidence 3 alternatives - 2 alternatives ----####

# this code can reproduce the subtraction between correct and incorrect trials
# to do that, add at indx3 and indx2:
# subj$binary_correct==0 (inc trials) // subj$binary_correct==1 (cor trials)


levels <- sort(unique(data$distance_ratio), decreasing = T)
nsubj  <- max(data$Nsujeto)
conf_data <- matrix(NA, nsubj, 3) # 3 levels of difficulty

for(i in 1:nsubj){
  subj <- data[data$Nsujeto==i,]
  temp <- c()
  
  for(j in levels){
    indx3 <- which(subj$distance_ratio==j & subj$Nalternativas==3)
    indx2 <- which(subj$distance_ratio==j & subj$Nalternativas==2)
    temp <- c(temp, mean(subj[indx3,]$Confidence)-mean(subj[indx2,]$Confidence))
  }
  
  conf_data[i,] <- temp
  
}

# standard error of mean 
level1 <- mean_se(na.omit(conf_data[,1]))
level2 <- mean_se(na.omit(conf_data[,2]))
level3 <- mean_se(na.omit(conf_data[,3]))

# plot
graphdata <- data.frame(names = c('0.33', '0.4', '0.47'), 
                        values = colMeans(conf_data, na.rm = T))
graphdata %>%
  ggplot(aes(x=c('0.33', '0.4', '0.47'),y=values))+
  geom_col(fill='gray28')+
  geom_errorbar(aes(x=c('0.33', '0.4', '0.47'), ymin=c(level1$ymin, level2$ymin, level3$ymin), 
                    ymax=c(level1$ymax,level2$ymax,level3$ymax)),
                width=0.4, size=1.3, color='orange')+
  ylim(c(-0.12,0.10))+
  ggtitle('Confidence (3alternatives-2alternatives)')+
  ylab('Confidence 3alt-2alt')+
  xlab('Task difficulty (distance to closest cloud)')+
  theme_classic()



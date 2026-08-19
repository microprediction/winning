library(tidyverse)

# -------------------------------------------------------------------------------------------------

data <- read.csv2('data_experiment2.csv')
data <- as_tibble(data)
data$Correct = as.logical(data$Correct)

# ----------------------------------------------------------------------------------------------------

colplots <- colorRampPalette(c('orange', 'darkorange3'))(3)

###---- Response times by difficulty & amount of alternatives ----####

data %>%
  group_by(Nsujeto, StimVal, Nalternativas) %>% 
  summarize(rt1 = mean(RT_type1)) %>% 
  ggplot(aes(x=as.factor(StimVal),
             y=rt1,
             group=as.factor(Nalternativas)))+
  stat_summary(fun = mean, geom = "line", na.rm=T, lwd=2, 
               col=c(rep('darkgreen',5), rep(colplots[1],5),
                     rep(colplots[2],5), rep(colplots[3],5)))+
  stat_summary(geom='errorbar', lwd=2, col=c(rep('darkgreen',5), rep(colplots[1],5),
               rep(colplots[2],5), rep(colplots[3],5)),width=0.1)+
  xlab('Task difficulty (stimulus2/stimulus1)')+
  ylab('RT type 1')+
  ggtitle('Response times (type 1 task) by task difficulty & amount of alternatives')+
  theme_classic()+
  coord_cartesian(ylim=c(1000,2500))+
  annotate('text', x=2, y=c(2400, 2300, 2200, 2100), 
           label=c('2 alternatives','3 alternatives','4 alternatives', '5 alternatives'),
           col=c('darkgreen', colplots[1], colplots[2], colplots[3]), size=7)

data %>%
  group_by(Nsujeto, StimVal, Nalternativas) %>% 
  summarize(rtconf = mean(RT_Confidence)) %>% 
  ggplot(aes(x=as.factor(StimVal),
             y=rtconf,
             group=as.factor(Nalternativas)))+
  stat_summary(fun = mean, geom = "line", na.rm=T, lwd=2, 
               col=c(rep('darkgreen',5), rep(colplots[1],5),
                     rep(colplots[2],5), rep(colplots[3],5)))+
  stat_summary(geom='errorbar', lwd=2, col=c(rep('darkgreen',5), rep(colplots[1],5),
               rep(colplots[2],5), rep(colplots[3],5)),width=0.1)+
  xlab('Task difficulty (stimulus2/stimulus1)')+
  ylab('RT type 2')+
  ggtitle('Response times (type 2 task) by task difficulty & amount of alternatives')+
  theme_classic()+
  coord_cartesian(ylim=c(1000,2500))+
  annotate('text', x=2, y=c(2400, 2300, 2200, 2100), 
           label=c('2 alternatives','3 alternatives','4 alternatives', '5 alternatives'),
           col=c('darkgreen', colplots[1], colplots[2], colplots[3]), size=7)

####---- Performance by difficulty & amount of alternatives ----####
data %>%
  group_by(Nsujeto, StimVal, Nalternativas) %>% 
  summarize(perf = mean(binary_correct)) %>% 
  ggplot(aes(x=as.factor(StimVal),
             y=perf,
             group=as.factor(Nalternativas)))+
  stat_summary(fun = mean, geom = "line", na.rm=T, lwd=2, 
               col=c(rep('darkgreen',5), rep(colplots[1],5),
                     rep(colplots[2],5), rep(colplots[3],5)))+
  stat_summary(geom='errorbar', lwd=2, col=c(rep('darkgreen',5), rep(colplots[1],5),
               rep(colplots[2],5), rep(colplots[3],5)),width=0.1)+
  xlab('Task difficulty (stimulus2/stimulus1)')+
  ylab('Performance')+
  ggtitle('Performance by task difficulty & amount of alternatives')+
  theme_classic()+
  annotate('text', x=1.5, y=c(0.8, 0.775, 0.75, 0.725), 
           label=c('2 alternatives','3 alternatives','4 alternatives', '5 alternatives'),
           col=c('darkgreen', colplots[1], colplots[2], colplots[3]), size=7)
         

####---- Confidence by difficulty & amount of alternatives ----####
data %>%
  group_by(Nsujeto, StimVal, Nalternativas) %>% 
  summarize(conf = mean(Confidence)) %>% 
  ggplot(aes(x=as.factor(StimVal),
             y=conf,
             group=as.factor(Nalternativas)))+
  stat_summary(fun = mean, geom = "line", na.rm=T, lwd=2, 
               col=c(rep('darkgreen',5), rep(colplots[1],5),
                     rep(colplots[2],5), rep(colplots[3],5)))+
  stat_summary(geom='errorbar', lwd=2, col=c(rep('darkgreen',5), rep(colplots[1],5),
               rep(colplots[2],5), rep(colplots[3],5)),width=0.1)+
  xlab('Task difficulty (stimulus2/stimulus1)')+
  ylab('Confidence')+
  ggtitle('Confidence by task difficulty & amount of alternatives')+
  theme_classic()+
  coord_cartesian(ylim=c(0.5,0.9))+
  annotate('text', x=4, y=c(0.8, 0.775, 0.75, 0.725), 
           label=c('2 alternatives','3 alternatives','4 alternatives', '5 alternatives'),
           col=c('darkgreen', colplots[1], colplots[2], colplots[3]), size=7)

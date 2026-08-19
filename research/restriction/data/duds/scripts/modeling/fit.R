fit <- function(data,model,a1,a2,a3,nd,n,m){
  
  # boundaries for the parameters that control the linear transformation
  b0low <- -2
  b0up  <- 4
  b1low <- -2
  b1up  <- 4
  
  # loop on subjects and find best-fitting parameter values
  fittedpars <- data.frame(sigma=NA,alpha=NA,b0=NA,b1=NA,loglik=NA)
  
  for(i in 1:99){ # loop on subjects
    
    cat('Fitting subject n', i, 'with', model,'model','\n')
    
    # get decisions and confidence per subject
    datatofit <- data[c('binary_correct','Confidence','StimVal','Nalternativas')][data$Nsujeto==i,]

    cat('Fitting decisions...', '\n')
    
    # use optim to find best-fitting values of the parameters by minimizing 
    # the negative log likelihood --- FITTING DECISIONS
    estimationD <- optim(c(0.01,20), 
                         fn=computeLogLikelihoodDecisions, n=nd, data=datatofit, 
                         model=model, method='SANN', control=list(maxit=100))
    
    cat('Simulating for fitting confidence...', '\n')
    
    # sim decisions & conf
    sim <- data.frame(decisions=c(),confidence=c(),difficulty=c(),nAlt=c())
    for(l in 1:length(a2)){
      temp2 <- matrix(NA,m,2)
      temp3 <- matrix(NA,m,2)
      for(o in 1:m){
        two   <- simNtrials(n,model,a1,a2[l],a3=F,estimationD$par[1],estimationD$par[2])
        three <- simNtrials(n,model,a1,a2[l],a3,estimationD$par[1],estimationD$par[2])
        
        temp2[o,] <- colMeans(two)
        temp3[o,] <- colMeans(three)
        
        
      }
      
      sim <- rbind(sim,data.frame(decisions=c(temp2[,1],temp3[,1]),
                                  confidence=c(temp2[,2],temp3[,2]),
                                  difficulty=c(rep(a2[l],m),rep(a2[l],m)),
                                  nAlt=c(rep(2,m),rep(3,m))))
      
    }
    
    
    cat('Starting the optimization algorithm for fitting confidence...', '\n')
    
    datatofit %>%
      group_by(StimVal, Nalternativas) %>%
      summarise(Confidence = mean(Confidence)) -> datatofit
    
    datatofit$Confidence <- round(datatofit$Confidence,1)
    
    # use optim to find best-fitting values of the parameters by minimizing 
    # the negative log likelihood --- FITTING CONFIDENCE
    estimationC_1 <- optim(c(-0.1,0.8), fn=computeLogLikelihoodConfidence, n=n, 
                           data=datatofit, sigma=estimationD$par[1], 
                           alpha=estimationD$par[2],b0low=b0low, b0up=b0up,
                           b1low=b1low,b1up=b1up,sim=sim,model=model, 
                           method='SANN', control=list(maxit=1000))
    
    estimationC_2 <- optim(c(0,1), fn=computeLogLikelihoodConfidence, n=n, 
                           data=datatofit, sigma=estimationD$par[1], 
                           alpha=estimationD$par[2],b0low=b0low, b0up=b0up,
                           b1low=b1low,b1up=b1up,sim=sim,model=model, 
                           method='SANN', control=list(maxit=1000))
    
    estimationC_3 <- optim(c(0.3,0.3), fn=computeLogLikelihoodConfidence, n=n, 
                           data=datatofit, sigma=estimationD$par[1], 
                           alpha=estimationD$par[2],b0low=b0low, b0up=b0up,
                           b1low=b1low,b1up=b1up,sim=sim,model=model, 
                           method='SANN', control=list(maxit=1000))
    
    best.estimation <- min(c(estimationC_1$value, estimationC_2$value, estimationC_3$value))
    if(best.estimation==estimationC_1$value){
      estimationC <- estimationC_1
    }
    if(best.estimation==estimationC_2$value){
      estimationC <- estimationC_2
    }
    if(best.estimation==estimationC_3$value){
      estimationC <- estimationC_3
    }
    
    # get best fitting values
    fittedpars[i,] <- c(estimationD$par[1], estimationD$par[2],
                        estimationC$par[1], estimationC$par[2],
                        -estimationC$value)
    
    cat(c('Best values:', '\n'))
    cat(c('Sigma =', estimationD$par[1], '\n')) 
    cat(c('Alpha =',estimationD$par[2], '\n'))
    cat(c('b0 =', estimationC$par[1], '\n'))
    cat(c('b1 =', estimationC$par[2], '\n'))
    cat(c('Log-likelihood for this parameters =', -estimationC$value), '\n')
    
    cat('Done.', '\n')
    
    cat('--------------------------------------------', '\n')
  }  
  
  return(fittedpars)
}

computeLogLikelihoodDecisions <- function(parameters,n,data,model){
  
  # this function is for fitting only the decisions
  
  if((parameters[1]<0.001) | (parameters[2]<4) | (parameters[1]>1)){
    return(NA)
  }
  
  # sim decisions & conf
  sim <- data.frame(decisions=c(),confidence=c(),difficulty=c(),nAlt=c())
  for(i in 1:length(a2)){
    temp2 <- simNtrials(n,model,a1,a2[i],a3=F,parameters[1],parameters[2])
    temp3 <- simNtrials(n,model,a1,a2[i],a3,parameters[1],parameters[2])
    sim <- rbind(sim,data.frame(decisions=c(temp2$decision,temp3$decision),
                                confidence=c(temp2$confidence,temp3$confidence),
                                difficulty=c(rep(a2[i],n),rep(a2[i],n)),
                                nAlt=c(rep(2,n),rep(3,n))))
    
  }
  
  # 1) calculate p(correct | sigma, alpha) for each condition (difficulty x alternatives)
  pc <- data.frame(p_one=rep(NA,10),difficulty=rep(NA,10),nAlt=rep(NA,10))
  for(o in 1:length(a2)){
    temp2 <- sim[sim$difficulty==a2[o] & sim$nAlt==2,]
    temp3 <- sim[sim$difficulty==a2[o] & sim$nAlt==3,]
    
    pc[o,] <- c(mean(temp2$decisions),a2[o],2)
    pc[o+5,] <- c(mean(temp3$decisions),a2[o],3)
  }
  
  
  # 2) compute likelihood for each condition (diff x nalt)
  p <- loopOnConditionsD(data,pc,sim)
  
  # 3) return loglikelihood
  loglikelihood <- sum(log(p))
  loglikelihood <- -loglikelihood
  
  
  return(loglikelihood)
}

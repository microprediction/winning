computeLogLikelihoodConfidence <- function(parameters,n,m,data,model,sigma,alpha,b0low,b0up,b1low,b1up,sim){

  # this function is for fitting only confidence (with fixed alpha & sigma obtained
  # from fitting the decisions first)
  
  if((parameters[1] < b0low) | (parameters[2] < b1low) | (parameters[1] > b0up) | (parameters[2] > b1up)){
    return(NA)
  }
  
  sim$confidenceLT <- parameters[1] + parameters[2]*sim$confidence # linear transformation
  sim$confidenceLT <- round(sim$confidenceLT,1) # round up to 1 decimal
  
  
  # 1) calculate p(correct | sigma, alpha) for each condition (difficulty x alternatives)
  pc <- data.frame(p_one=rep(NA,10),difficulty=rep(NA,10),nAlt=rep(NA,10))
  for(o in 1:length(a2)){
    temp2 <- sim[sim$difficulty==a2[o] & sim$nAlt==2,]
    temp3 <- sim[sim$difficulty==a2[o] & sim$nAlt==3,]
    
    pc[o,] <- c(mean(temp2$decisions),a2[o],2)
    pc[o+5,] <- c(mean(temp3$decisions),a2[o],3)
  }
  
  
  # 2) compute likelihood for each condition (diff x nalt)
  p <- loopOnConditionsC(data,pc,sim)
  
  # 3) return loglikelihood
  loglikelihood <- sum(log(p))
  loglikelihood <- -loglikelihood
  
  
  return(loglikelihood)
}
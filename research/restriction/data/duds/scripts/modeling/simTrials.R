simNtrials <- function(n,model,a1,a2,a3,sigma,alpha,theta=F){
  if(model=='max'){
    source('max_model.R')
  } else if(model=='diff'){
    source('diff_model.R')
  } else if(model=='contrast'){
    source('contrast_model.R')
  } else if(model=='av_res'){
    source('average_residual_model.R')
  } else if(model=='contrast_model_theta'){
    source('contrast_model_theta.R')
  } else{
    source('average_residual_model_theta.R')
  }
  
  decisions <- rep(NA,n)
  confidence <- rep(NA,n)
  if((model == 'contrast_model_theta') | (model == 'av_res_theta')){
    for(i in 1:n){
      x <- simulate_trial(s=sigma,a=alpha,A1=a1,A2=a2,A3=a3,theta)
      decisions[i] <- x[1]
      confidence[i] <- x[2]
    }
  } else{
    for(i in 1:n){
      x <- simulate_trial(s=sigma,a=alpha,A1=a1,A2=a2,A3=a3)
      decisions[i] <- x[1]
      confidence[i] <- x[2]
    }
  }
  
  output <- data.frame(decision=decisions,confidence=confidence)
  
  return(output)
}
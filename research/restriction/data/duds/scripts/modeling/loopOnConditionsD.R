loopOnConditionsD <- function(datatofit, pc, sim){
  # calculate likelihood of parameters across conditions
  
  p <- rep(NA,length(datatofit$binary_correct))
  
  for(l in 1:length(datatofit$binary_correct)){
    if(datatofit[l,]$binary_correct==1 & datatofit[l,]$StimVal==0.7 &
       datatofit[l,]$Nalternativas==2){
      
      d <- pc[pc$difficulty==0.7 & pc$nAlt==2,]$p_one # likelihood of decision
      p[l] <- d # likelihood
      
    } else if(datatofit[l,]$binary_correct==1 & datatofit[l,]$StimVal==0.7 &
              datatofit[l,]$Nalternativas==3){
      
      d <- pc[pc$difficulty==0.7 & pc$nAlt==3,]$p_one # likelihood of decision
      p[l] <- d # likelihood
      
    } else if(datatofit[l,]$binary_correct==1 & datatofit[l,]$StimVal==0.8 &
              datatofit[l,]$Nalternativas==2){
      d <- pc[pc$difficulty==0.8 & pc$nAlt==2,]$p_one  # likelihood of decision
      p[l] <- d # likelihood
      
      
    } else if(datatofit[l,]$binary_correct==1 & datatofit[l,]$StimVal==0.8 &
              datatofit[l,]$Nalternativas==3){ 
      
      d <- pc[pc$difficulty==0.8 & pc$nAlt==3,]$p_one  # likelihood of decision
      p[l] <- d # likelihood
      
      
    } else if(datatofit[l,]$binary_correct==1 & datatofit[l,]$StimVal==0.9 &
              datatofit[l,]$Nalternativas==2){
      
      d <- pc[pc$difficulty==0.9 & pc$nAlt==2,]$p_one  # likelihood of decision
      p[l] <- d # likelihood
      
      
    } else if(datatofit[l,]$binary_correct==1 & datatofit[l,]$StimVal==0.9 &
              datatofit[l,]$Nalternativas==3){
      
      d <- pc[pc$difficulty==0.9 & pc$nAlt==3,]$p_one  # likelihood of decision
      p[l] <- d # likelihood
      
      
    } else if(datatofit[l,]$binary_correct==1 & datatofit[l,]$StimVal==0.93 &
              datatofit[l,]$Nalternativas==2){
      
      d <- pc[pc$difficulty==0.93 & pc$nAlt==2,]$p_one  # likelihood of decision
      p[l] <- d # likelihood
      
    } else if(datatofit[l,]$binary_correct==1 & datatofit[l,]$StimVal==0.93 &
              datatofit[l,]$Nalternativas==3){
      
      d <- pc[pc$difficulty==0.93 & pc$nAlt==3,]$p_one  # likelihood of decision
      p[l] <- d  # likelihood
      
      
    } else if(datatofit[l,]$binary_correct==1 & datatofit[l,]$StimVal==0.95 &
              datatofit[l,]$Nalternativas==2){
      
      d <- pc[pc$difficulty==0.95 & pc$nAlt==2,]$p_one  # likelihood of decision
      p[l] <- d # likelihood
      
    } else if(datatofit[l,]$binary_correct==1 & datatofit[l,]$StimVal==0.95 &
              datatofit[l,]$Nalternativas==3){
      
      d <- pc[pc$difficulty==0.95 & pc$nAlt==3,]$p_one  # likelihood of decision
      p[l] <- d # likelihood
      
      
      # incorrect response -> 1 - p(correct|sigma,alpha)
    } else if(datatofit[l,]$binary_correct==0 & datatofit[l,]$StimVal==0.7 &
              datatofit[l,]$Nalternativas==2){
      
      d <- 1 - pc[pc$difficulty==0.7 & pc$nAlt==2,]$p_one # likelihood of decision
      p[l] <- d # likelihood
      
      
    } else if(datatofit[l,]$binary_correct==0 & datatofit[l,]$StimVal==0.7 &
              datatofit[l,]$Nalternativas==3){
      
      d <- 1 - pc[pc$difficulty==0.7 & pc$nAlt==3,]$p_one # likelihood of decision
      p[l] <- d # likelihood
      
      
    } else if(datatofit[l,]$binary_correct==0 & datatofit[l,]$StimVal==0.8 &
              datatofit[l,]$Nalternativas==2){
      
      d <- 1 - pc[pc$difficulty==0.8 & pc$nAlt==2,]$p_one # likelihood of decision
      p[l] <- d # likelihood
      
      
    } else if(datatofit[l,]$binary_correct==0 & datatofit[l,]$StimVal==0.8 &
              datatofit[l,]$Nalternativas==3){
      
      d <- 1 - pc[pc$difficulty==0.8 & pc$nAlt==3,]$p_one # likelihood of decision
      p[l] <- d # likelihood
      
      
    } else if(datatofit[l,]$binary_correct==0 & datatofit[l,]$StimVal==0.9 &
              datatofit[l,]$Nalternativas==2){
      
      d <- 1 - pc[pc$difficulty==0.9 & pc$nAlt==2,]$p_one # likelihood of decision
      p[l] <- d # likelihood
      
    } else if(datatofit[l,]$binary_correct==0 & datatofit[l,]$StimVal==0.9 &
              datatofit[l,]$Nalternativas==3){
      
      d <- 1 - pc[pc$difficulty==0.9 & pc$nAlt==3,]$p_one # likelihood of decision
      p[l] <- d # likelihood
      
      
    } else if(datatofit[l,]$binary_correct==0 & datatofit[l,]$StimVal==0.93 &
              datatofit[l,]$Nalternativas==2){
      
      d <- 1 - pc[pc$difficulty==0.93 & pc$nAlt==2,]$p_one # likelihood of decision
      p[l] <- d # likelihood
    
    } else if(datatofit[l,]$binary_correct==0 & datatofit[l,]$StimVal==0.93 &
              datatofit[l,]$Nalternativas==3){
      
      d <- 1 - pc[pc$difficulty==0.93 & pc$nAlt==3,]$p_one # likelihood of decision
      p[l] <- d # likelihood
      
      
    } else if(datatofit[l,]$binary_correct==0 & datatofit[l,]$StimVal==0.95 &
              datatofit[l,]$Nalternativas==2){
      
      d <- 1 - pc[pc$difficulty==0.95 & pc$nAlt==2,]$p_one # likelihood of decision
      p[l] <- d # likelihood
      
      
    } else if(datatofit[l,]$binary_correct==0 & datatofit[l,]$StimVal==0.95 &
              datatofit[l,]$Nalternativas==3){
      
      d <- 1 - pc[pc$difficulty==0.95 & pc$nAlt==3,]$p_one # likelihood of decision
      p[l] <- d # likelihood
      
      
    }
    
    if(p[l]==0){
      p[l] <- 0.0001  # to avoid log(0)
    }
    
  }
  
  return(p)
}
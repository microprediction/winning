loopOnConditionsC <- function(datatofit, pc, sim){
  # calculate likelihood of parameters across conditions
  
  p <- rep(NA,length(datatofit$Confidence))
  
  for(l in 1:length(datatofit$Confidence)){
    if(datatofit[l,]$StimVal==0.7 & datatofit[l,]$Nalternativas==2){
      
      c <- length(which(sim[sim$difficulty==0.7 & sim$nAlt==2,]$confidenceLT==datatofit[l,'Confidence'])) 
      c <- c / length(sim[sim$difficulty==0.7 & sim$nAlt==2,]$confidenceLT) # likelihood of confidence
      
      p[l] <- c # likelihood
      
    } else if(datatofit[l,]$StimVal==0.7 & datatofit[l,]$Nalternativas==3){
      
      c <- length(which(sim[sim$difficulty==0.7 & sim$nAlt==3,]$confidenceLT==datatofit[l,'Confidence'])) 
      c <- c / length(sim[sim$difficulty==0.7 & sim$nAlt==3,]$confidenceLT) # likelihood of confidence
      
      p[l] <- c # likelihood
      
    } else if(datatofit[l,]$StimVal==0.8 & datatofit[l,]$Nalternativas==2){
      
      c <- length(which(sim[sim$difficulty==0.8 & sim$nAlt==2,]$confidenceLT==datatofit[l,'Confidence'])) 
      c <- c / length(sim[sim$difficulty==0.8 & sim$nAlt==2,]$confidenceLT) # likelihood of confidence
      
      p[l] <- c # likelihood
      
      
    } else if(datatofit[l,]$StimVal==0.8 & datatofit[l,]$Nalternativas==3){ 
      
      c <- length(which(sim[sim$difficulty==0.8 & sim$nAlt==3,]$confidenceLT==datatofit[l,'Confidence'])) 
      c <- c / length(sim[sim$difficulty==0.8 & sim$nAlt==3,]$confidenceLT) # likelihood of confidence
      
      p[l] <- c # likelihood
      
      
    } else if(datatofit[l,]$StimVal==0.9 & datatofit[l,]$Nalternativas==2){
      
      c <- length(which(sim[sim$difficulty==0.9 & sim$nAlt==2,]$confidenceLT==datatofit[l,'Confidence'])) 
      c <- c / length(sim[sim$difficulty==0.9 & sim$nAlt==2,]$confidenceLT) # likelihood of confidence
      
      p[l] <- c # likelihood
      
      
    } else if(datatofit[l,]$StimVal==0.9 & datatofit[l,]$Nalternativas==3){
      
      c <- length(which(sim[sim$difficulty==0.9 & sim$nAlt==3,]$confidenceLT==datatofit[l,'Confidence'])) 
      c <- c / length(sim[sim$difficulty==0.9 & sim$nAlt==3,]$confidenceLT) # likelihood of confidence
      
      p[l] <- c # likelihood
      
      
    } else if(datatofit[l,]$StimVal==0.93 & datatofit[l,]$Nalternativas==2){
      
      c <- length(which(sim[sim$difficulty==0.93 & sim$nAlt==2,]$confidenceLT==datatofit[l,'Confidence'])) 
      c <- c / length(sim[sim$difficulty==0.93 & sim$nAlt==2,]$confidenceLT) # likelihood of confidence
      
      p[l] <- c # likelihood
      
    } else if(datatofit[l,]$StimVal==0.93 & datatofit[l,]$Nalternativas==3){
      
      c <- length(which(sim[sim$difficulty==0.93 & sim$nAlt==3,]$confidenceLT==datatofit[l,'Confidence'])) 
      c <- c / length(sim[sim$difficulty==0.93 & sim$nAlt==3,]$confidenceLT) # likelihood of confidence
      
      p[l] <- c # likelihood
      
      
    } else if(datatofit[l,]$StimVal==0.95 & datatofit[l,]$Nalternativas==2){
      
      c <- length(which(sim[sim$difficulty==0.95 & sim$nAlt==2,]$confidenceLT==datatofit[l,'Confidence'])) 
      c <- c / length(sim[sim$difficulty==0.95 & sim$nAlt==2,]$confidenceLT) # likelihood of confidence
      
      p[l] <- c # likelihood
      
    } else if(datatofit[l,]$StimVal==0.95 & datatofit[l,]$Nalternativas==3){
      
      c <- length(which(sim[sim$difficulty==0.95 & sim$nAlt==3,]$confidenceLT==datatofit[l,'Confidence'])) 
      c <- c / length(sim[sim$difficulty==0.95 & sim$nAlt==3,]$confidenceLT) # likelihood of confidence
      
      p[l] <- c # likelihood
      
    }   
   
    if(p[l]==0){
      p[l] <- 0.0001  # to avoid log(0)
    }
    
  }
  
  return(p)
}
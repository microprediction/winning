compareModelvsData <- function(data,fittedpars,model,n,m,a1,a2,a3){
  
  # sim decisions & conf
  nsubj <- 99
  simd <- matrix(NA,nsubj,10)
  simc <- matrix(NA,nsubj,10)
  dec <- matrix(NA,nsubj,10)
  conf <- matrix(NA,nsubj,10)
  
  for(i in 1:nsubj){
    subj <- data[data$Nsujeto==i,]
    
    cat('Simulating subject n ', i, '\n')
    
    sim <- data.frame(decisions=c(),confidence=c(),difficulty=c(),nAlt=c())
    temp <- fittedpars[i,]
    for(l in 1:length(a2)){
      temp2 <- matrix(NA,m,2)
      temp3 <- matrix(NA,m,2)
      for(o in 1:m){
        two   <- simNtrials(n,model,a1,a2[l],a3=F,temp$sigma,temp$alpha)
        three <- simNtrials(n,model,a1,a2[l],a3,temp$sigma,temp$alpha)
        
        temp2[o,] <- colMeans(two)
        temp3[o,] <- colMeans(three)
        
        
      }
      
      sim <- rbind(sim,data.frame(decisions=c(temp2[,1],temp3[,1]),
                                  confidence=c(temp2[,2],temp3[,2]),
                                  difficulty=c(rep(a2[l],m),rep(a2[l],m)),
                                  nAlt=c(rep(2,m),rep(3,m))))
      
    }
    
    sim$confidenceLT <- temp$b0 + temp$b1*sim$confidence  # linear transformation
    sim$confidenceLT <- round(sim$confidenceLT,1) # round up
    
    for(l in 1:length(a2)){
      simd[i,l] <- mean(sim[sim$difficulty==a2[l] & sim$nAlt == 2,]$decisions)
      simd[i,l+5] <- mean(sim[sim$difficulty==a2[l] & sim$nAlt == 3,]$decisions)
      
      simc[i,l] <- mean(sim[sim$difficulty==a2[l] & sim$nAlt == 2,]$confidenceLT)
      simc[i,l+5] <- mean(sim[sim$difficulty==a2[l] & sim$nAlt == 3,]$confidenceLT)
      
      dec[i,l] <- mean(subj[subj$StimVal==a2[l] & subj$Nalternativas==2,]$binary_correct)
      dec[i,l+5] <- mean(subj[subj$StimVal==a2[l] & subj$Nalternativas==3,]$binary_correct)
      
      conf[i,l] <- mean(subj[subj$StimVal==a2[l] & subj$Nalternativas==2,]$Confidence)
      conf[i,l+5] <- mean(subj[subj$StimVal==a2[l] & subj$Nalternativas==3,]$Confidence)
    }
    
  }
  
  # get mse
  d <- apply(simd, 2, mean_se)
  c <- apply(simc, 2, mean_se)
  dataD <- apply(dec, 2, mean_se)
  dataC <- apply(conf, 2, mean_se)
  
  msedataD <- matrix(NA,3,10)
  msedataC <- matrix(NA,3,10)
  msemodelD <- matrix(NA,3,10)
  msemodelC <- matrix(NA,3,10)
  for(i in 1:10){
    msedataD[,i] <- c(dataD[[i]][1,]$y, dataD[[i]][1,]$ymin, dataD[[i]][1,]$ymax)
    msedataC[,i] <- c(dataC[[i]][1,]$y, dataC[[i]][1,]$ymin, dataC[[i]][1,]$ymax)
    msemodelD[,i] <- c(d[[i]][1,]$y, d[[i]][1,]$ymin, d[[i]][1,]$ymax)
    msemodelC[,i] <- c(c[[i]][1,]$y, c[[i]][1,]$ymin, c[[i]][1,]$ymax)
  }
  
  par(bty='l', pty='s')
  # plot model predictions vs data (performance)
  plot(msedataD[1,1:5], col='darkgreen',type='l',lwd=3,ylim=c(0,1), xaxt='n',
       main=model, ylab='P(correct)', xlab='Task difficulty (area2/area1)')
  axis(1,at=1:5,labels=a2)
  lines(msedataD[1,6:10], col='orange',lwd=3)
  arrows(x0=1:5,y0=msedataD[2,1:5],y1=msedataD[3,1:5], col='darkgreen',
         code=0, angle=90, lwd=3)
  arrows(x0=1:5,y0=msedataD[2,6:10],y1=msedataD[3,6:10], col='orange',
         code=0, angle=90, lwd=3)
  polygon(c(1:5,rev(1:5)), c(msemodelD[2,1:5],rev(msemodelD[3,1:5])),
          col=alpha('darkgreen',0.5), border=NA)
  polygon(c(1:5,rev(1:5)), c(msemodelD[2,6:10],rev(msemodelD[3,6:10])),
          col=alpha('orange',0.5), border=NA)
  
  # plot model predictions vs data (confidence)
  plot(msedataC[1,1:5], col='darkgreen',type='l',lwd=3,ylim=c(0,1), xaxt='n',
       main=model, ylab='Confidence', xlab='Task difficulty (area2/area1)')
  axis(1,at=1:5,labels=a2)
  lines(msedataC[1,6:10], col='orange',lwd=3)
  arrows(x0=1:5,y0=msedataC[2,1:5],y1=msedataC[3,1:5], col='darkgreen',
         code=0, angle=90, lwd=3)
  arrows(x0=1:5,y0=msedataC[2,6:10],y1=msedataC[3,6:10], col='orange',
         code=0, angle=90, lwd=3)
  polygon(c(1:5,rev(1:5)), c(msemodelC[2,1:5],rev(msemodelC[3,1:5])),
          col=alpha('darkgreen',0.5), border=NA)
  polygon(c(1:5,rev(1:5)), c(msemodelC[2,6:10],rev(msemodelC[3,6:10])),
          col=alpha('orange',0.5), border=NA)
  
  
}
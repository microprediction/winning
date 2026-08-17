# calculate P(max(x)=x1 | {x1,x2,x3,s})
pmax.3opt <- function(x, s, mu1, mu2, mu3=F, id) {
  if(mu3){
    if (id == 1){
      out <- pnorm(x, mu2, s) * pnorm(x, mu3, s) * dnorm(x, mu1, s)}
    if (id == 2){
      out <- pnorm(x, mu1, s) * pnorm(x, mu3, s) * dnorm(x, mu2, s)}
  }
  else{
    if (id == 1){
      out <- pnorm(x, mu2, s) * dnorm(x, mu1, s)}
    if (id == 2){
      out <- pnorm(x, mu1, s) * dnorm(x, mu2, s)}
  }
  return(out)
}

# 1 trial simulation
simulate_trial <- function(s,A1,A2,A3=F,a){
  
  ## generate internal responses
  x1 <- rnorm(1, A1, s)
  x2 <- rnorm(1, A2, s)
  if(A3){
    x3 <- rnorm(1, A3, s)
    x  <- cbind(x1,x2,x3)}
  else{
    x  <- cbind(x1,x2)}
  
  ## type 1 decision
  
  if(A3){
    post      <- rep(NA, 3)
    post[1]   <- integrate(pmax.3opt, -10, 10, s+0.15, x1, x2, x3, 1)$value
    post[2]   <- integrate(pmax.3opt, -10, 10, s+0.15, x1, x2, x3, 2)$value
    post[3]   <- 1 - post[1] - post[2]} 
  else{
    post      <- rep(NA, 2)
    post[1]   <- integrate(pmax.3opt, -10, 10, s+0.15, x1, x2, id=1)$value
    post[2]   <- integrate(pmax.3opt, -10, 10, s+0.15, x1, x2, id=2)$value
  }
  
  if(any(sign(post)==-1) | any(sign(post)==0)){  
    indx <- which(sign(post)==-1 | sign(post)==0)
    post[indx] <- 0.000001
  }
  
  # decision noise (p ---> q)
  q <- rdirichlet(n = 1, alpha = a*post)
  d <- which.max(q) # type 1 decision
  
  
  if(d != 1){  # if decision wasn't correct return a 0
    d <- 0
  }
  
  ## type 2 decision --> confidence
  o <- order(q, decreasing = T)
  if(A3){
    confidence <- q[o[1]] - mean(c(q[o[2]], q[o[3]])) # average residual model
  }
  else{
    confidence <- q[o[1]] - q[o[2]]} # equal to diff model
  
  return(c(d,confidence))
  
}

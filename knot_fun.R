rate.fun <- function(x1, x, y, d0, n0=20, deg=1){ #x1:測試點，看要不要在此放knot
  # d0 x1附近的小範圍的x
  ind1 <- which((x>(x1-d0)) & (x<x1))
  if (length(ind1)<n0) { return(NA)}

  ind2 <- which((x<(x1+d0)) & (x>x1))
  if (length(ind2)<n0) { return(NA)}

  yy <- y[ind1]
  xx <- x[ind1]
  if (deg>1){ for (j in 2:deg){ xx <- cbind(xx, x[ind1]^j)}}

  y1_lm <- lm(yy~xx)
  ans1 <- y1_lm$coef
  
  yy <- y[ind2]
  xx <- x[ind2]
  if (deg>1){ for (j in 2:deg){ xx <- cbind(xx, x[ind2]^j)}}

  y2_lm <- lm(yy~xx)
  ans2 <- y2_lm$coef
  
  sigma <- summary(y1_lm)$cov.unscaled + summary(y2_lm)$cov.unscaled
  d <- (ans2-ans1)
  d <- d %*% solve(sigma) %*% d
  return(d[1,1])
}

cl.fun <- function(xx, score, d){
  # xx想放節點的位置；score: xx對應的檢定統計量
  # 剔除範圍內已有節點的其他節點
  sol <- xx[which.max(score)[1]]
  ind <- which(abs(xx-sol)>d)
  xnew <- xx
  snew <- score
  while( length(ind)>0 ){
    xnew <- xnew[ind]
    snew <- snew[ind]
    sol1 <- xnew[which.max(snew)[1]]
    sol <- c(sol, sol1)
    ind <- which(abs(xnew-sol1)>d)
  }
  return(sol)
}

rss0 <- function(xi, x, y, deg=3){
  bxx <- bs(x,knots=xi,Boundary.knots = c(0,1), deg=deg, intercept=T)
  y_lm <- lm(y~bxx-1)
  return(sum(y_lm$resid^2))
}

rss1 <- function(xi, x, y, deg=3){
  bxx <- bs(x, knots=xi, Boundary.knots=c(0,1), deg=deg, intercept=T)
  y_lm <- lm(y~bxx-1)
  return(y_lm$fitted)
}

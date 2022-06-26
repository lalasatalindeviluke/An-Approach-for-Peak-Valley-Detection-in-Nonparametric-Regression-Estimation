require("splines")
load("xy.RData")        # xy.RData has x and y
source("knot_fun.R")    # knot_fun.R has the functions called by get_knot

get_knot <- function(x, y){
  deg0 = 3
  deg1 = 1
  J = 5
  n <- length(y)
  evenind <- seq(2, n, by=2)
  sig <- median(abs(y[evenind] - y[evenind-1])) / (sqrt(2)*qnorm(.75))
  sol_list <- vector("list", length=J)
  bic_val <- rep(0, J)
  for (j in 1:J){
    m0 = 2^j
    newx <- matrix(0, n, 2)
    for (i in 1:n){
      newx[i,] <- c(x[i], rate.fun(x[i], x, y, 1/m0, deg=deg1))
    }
    ind1 <- (newx[,2]>(qchisq(.95, df=deg1+1)*sig^2)) & (!is.na(newx[,2]))
    sol_list[[j]] <- cl.fun(x[ind1], newx[ind1, 2], 1/m0)
    bic_val[j] <- n*log(rss0(sol_list[[j]], x, y, deg=deg0)/n) + length(sol_list[[j]])*log(n)
  }
  sol <- sol_list[[which.min(bic_val)[1]]]
  return(sol)
}

#transform x to x1 so that the range of x1 is in [0,1]
x1 <- x/2
knots_1 <- get_knot(x1, y)

#plot the data and the fitted spline function
plot(x1,y, type="n")
lines(x1, rss1(knots_1, x1, y), col=2)
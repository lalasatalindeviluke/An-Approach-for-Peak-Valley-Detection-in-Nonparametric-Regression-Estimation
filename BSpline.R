library(splines)
bs(women$height, df = 3)
bs(women$weight, df = 4)

bspl_model <- bs(women$height, df = 3)
summary(bspl_model)
plot(bspl_model[,1]~women$height)
for (j in 1:ncol(bspl_model)) lines(bspl_model[,j]~women$height, lwd=2, col=j)

#---------------------------------------------------------------------
x <- (1:1000)/1001
knotlist = c(0.1, 0.2)
ord=3
bx = bs(x, deg=ord-1, knots=knotlist, Boundary.knots=c(0,1), intercept=T)

knotlist2 = c(0, 0, 0, 1)
sd = splineDesign(knotlist2, x, ord=3, outer.ok=T)[,1]
bx2 = bs(x, deg=ord-1, knots=knotlist2, Boundary.knots=c(0,1))

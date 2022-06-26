from patsy import bs, dmatrix
import numpy as np
import matplotlib.pyplot as plt

# Compare with R performance
women_height = np.arange(58, 73)
women_weight = np.array([115 ,117 ,120 ,123 ,126 ,
                         129 ,132 ,135 ,139 ,142 ,
                         146 ,150 ,154 ,159 ,164])
bs(women_height, df=3)
bs(women_weight, df=4)
target = dmatrix("bs(x, df=3)",
                 {"x": women_height})
plt.figure(figsize=(16, 9))
plt.title("B-spline basis example (df=3)")
plt.plot(women_height, target)
plt.grid()
plt.show()

#--------------------------------------------------------
t = np.linspace(0, 1, 20)
h_t = t**2
# y = np.sin(h_t) + np.cos(h_t) + 0.02*np.random.randn(20)

y = dmatrix("bs(x, df=4)",
            {"x": t})
plt.figure(figsize=(16, 9))
plt.title("B-spline basis example (df=3)")
plt.plot(t, y)
plt.grid()
plt.show()
#---------------------------------------------------------
x = np.arange(1, 1001) / 1001
knotlist = [0.1, 0.2]
knot_all = [0, 0, 0, 0.1, 0.2, 1, 1, 1]
order=3

bx = bs(x=x, knots=knotlist, degree=2, include_intercept=True,
        lower_bound=0, upper_bound=1)

knotlist2 = [0,0,0,1]
splinedesign = bs(x=x, knots=knotlist2, degree=order-1,
                  lower_bound=0, upper_bound=1)


# def splineDesign(knots, x):
    
#     if len(set(knots[:-2])) == 1:
#         x_copy1 = np.where(knots[0]<=x<knots[-1], x, 0)
#         return (knots[-1]-x_copy1)**(knots.shape[0]-1) / (knots[-1]-knots[0])**(order-1)
    
#     if len(set(knots[1:])) == 1:
#         x_copy2 = np.where(knots[0]<=x<knots[-1], x, 0)
#         return (x_copy2-knots[0])**(knots.shape[0]-1) / (knots[-1]-knots[0])**(order-1)
        
    
#     result = (x-knots[0])/(knots[-1]-knots[0]) * splineDesign(knots[:-2], x, order=knots.shape[0]) \
#         + (knots[-1]-x)/(knots[-1]-knots[1]) * splineDesign(knots[1:], x, order=knots.shape[0])
    
#     return result
    
    


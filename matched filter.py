# %%
import numpy as np
import matplotlib.pyplot as plt
from sympy import *

# %% [markdown]
# $$ f(t) = \frac{1}{\sqrt{2\pi}}\exp{\frac{-t^2}{2}} $$

# %%
t = Symbol('t')
t1 = Symbol('t1')
f = (1/sqrt(2*pi))*exp(-t**2/2)
# f = (1/sqrt(2*pi))*exp(-t**2/2) + (1/(sqrt(2*pi)*sqrt(0.5)))*exp(-(t-2.5)**2/(2*0.5))

# %%
f1 = diff(f, t)

# %% [markdown]
# ---

# %%
f2 = diff(f, t, 2)
# f2.evalf(subs={'t':t+t1})

# %%
f3 = diff(f, t, 3)
# f3.evalf(subs={'t':t+t1})

# %%
# (f3/4).evalf(subs={'t': t1/sqrt(2)})

# # %%
# graph1 = plot(f.evalf(subs={'t':t1}), adaptive=False, nb_of_points=1000, show=False)
# backend1 = graph1.backend(graph1)
# backend1.process_series()
# backend1.fig.savefig('.\\figure\\1.png', dpi=300)
# backend1.show()

# # %%
# graph2 = plot((f3/4).evalf(subs={'t': t1/sqrt(2)}), adaptive=False, nb_of_points=1000, show=False)
# backend2 = graph2.backend(graph2)
# backend2.process_series()
# backend2.fig.savefig('.\\figure\\2.png', dpi=300)
# backend2.show()

# # %%
# graph3 = plot((f2/(2*sqrt(2))).evalf(subs={'t': t1/sqrt(2)}), adaptive=False, nb_of_points=1000, show=False)
# backend3 = graph3.backend(graph3)
# backend3.process_series()
# backend3.fig.savefig('.\\figure\\3.png', dpi=300)
# backend3.show()

# %%
graph0 = plot(f.evalf(subs={'t':t1}), adaptive=False, nb_of_points=1000, show=False, label="F(t1)", size=(16,9))
graph0.extend(plot((f3/4).evalf(subs={'t': t1/sqrt(2)}), adaptive=False, nb_of_points=1000, show=False, label="beta(t1)"))
graph0.extend(plot((f2/(2*sqrt(2))).evalf(subs={'t': t1/sqrt(2)}), adaptive=False, nb_of_points=1000, show=False, label="alpha(t1)"))
graph0.legend = True

backend0 = graph0.backend(graph0)
backend0.process_series()
backend0.fig.savefig('.\\figure\\matched filter1.png', dpi=300)
backend0.show()


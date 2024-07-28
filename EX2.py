import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

#For uniform weight we have got already the result:
N=11 #even
u_z = np.linspace(-1, 1, 300)
B_sq_uniform = (1/(N**2))*np.square(np.divide(np.sin((N/2)*np.pi*u_z), np.sin((1/2)*np.pi*u_z)))
# we have seen the results for non-uniform weighting in the exercise in class:
B_weghting = 1/2*np.sin(np.pi/(2*N))*(np.divide(np.sin((N*np.pi/2)*(u_z + 1/N)), np.sin((np.pi/2)*(u_z + 1/N)))
                                      + np.divide(np.sin((N*np.pi/2)*(u_z - 1/N)), np.sin((np.pi/2)*(u_z - 1/N))))
B_sq_weghting = np.real(B_weghting*np.conjugate(B_weghting))

fig = make_subplots(rows=1, cols=2,
                    subplot_titles=(r'$\text{Power Pattern Vs } \Psi$',r'$\text{PowerPattern Vs } \Psi \text{ in dB}$')
                    )


fig.add_trace(
    go.Scatter(
        x=u_z,
        y=B_sq_weghting,
        name="Non-uniform weights"
    ), row=1, col=1)

fig.add_trace(
    go.Scatter(
        x=u_z,
        y=B_sq_uniform,
        name="Uniform weights"
    ), row=1, col=1)
fig.update_yaxes(title_text=r'$|B|^2$', row=1, col=1)



fig.add_trace(
    go.Scatter(
        x=u_z,
        y=10*np.log10(B_sq_weghting),
        name="Non-uniform weights in dB"
    ), row=1, col=2)

fig.add_trace(
    go.Scatter(
        x=u_z,
        y=10*np.log10(B_sq_uniform),
        name="Uniform weights"
    ), row=1, col=2)
fig.update_xaxes(title_text=r'$u_z$')
fig.update_yaxes(title_text=r'$|B|^2_{dB}$', row=1, col=2)

fig.update_yaxes(range=[-70, 1], row=1, col=2)


fig.show()
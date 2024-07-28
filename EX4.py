
import numpy as np
import plotly.graph_objects as go
from numpy.linalg import inv
from plotly.subplots import make_subplots

def Hermitian(mat):
    return np.conjugate(np.transpose(mat))

def CalcWmvdr(inter_pos, res, INR):
    res = res
    inter_pos = inter_pos
    N = 10 #Num of sensor
    st = np.arange(N).T #column
    v1 = np.exp(1j*np.pi*inter_pos)**st #interfernce steering vector
    vs = np.exp(1j*np.pi*0)**st #desired steering vector, broadside, u=0
    INR = INR #In dB
    sigma_w = 1
    sigma_1 = 10**((INR + 20*np.log10(sigma_w))/20)
    Sn = sigma_w*np.eye(N) + sigma_1*np.outer(v1,np.conjugate(v1))
    Sn_inv = inv(Sn)
    Delta = 1/(Hermitian(vs)@Sn_inv@vs) #scalar
    W_mvdr = Delta*Sn_inv@vs
    #I did that with dft but of course I could do that simply with B=W_mvdr.T@u with u=np.exp(1j*np.pi)**np.outer(st, u_var.T) and u_var=np.linspace(0.001,0.5,res)
    u = np.fft.fftshift(np.fft.fftfreq(res))*2 # the k-th frequency
    W = np.pad(np.squeeze(W_mvdr), (0,res-N),constant_values=0)
    B = np.conjugate(np.fft.fftshift(np.fft.fft(W)))
    BP = np.real(B*np.conjugate(B))
    return u, np.sqrt(BP)
INR = [70,0]
res = 1000
for inr in INR:
    u, BP_u03 = CalcWmvdr(inter_pos=0.3, res=res, INR=inr)
    u, BP_u0004 = CalcWmvdr(inter_pos=0.004, res=res, INR=inr) #near the desired direction--> can be problematic
    fig = fig = make_subplots(rows=1, cols=2,
                                subplot_titles=("Interference Direction=0.3", "Interference Direction=0.004", "Plot 3", "Plot 4"))

    fig.add_trace(
        go.Scatter(
            x=u,
            y=BP_u03,
            name="Power Pattern, Interference Direction=0.3"), row=1, col=1
        )
    fig.add_trace(
        go.Scatter(
            x=u,
            y=BP_u0004,
            name="Power Pattern, Interference Direction=0.004"), row=1, col=2
        )
    fig.update_layout(title=r'$\text{{Power Pattern Vs }} u_z, \text{{ INR={}dB}}$'.format(inr),
                       xaxis_title=r'$u_z$',
                       yaxis_title=r'$|B|^2$')
    fig.update_xaxes(range=[-1,1])


    fig.update_xaxes(title_text=r'$u_z$')
    fig.update_yaxes(title_text=r'$|B|^2$')
    bp = [BP_u03, BP_u0004]
    dis = [0.3, 0.004]
    for i, b in enumerate(bp):
        fig.add_annotation(x=dis[i], y=-0.05,
                           text=r"$u_z={}$".format(dis[i]),
                           showarrow=False,
                           xshift=-28, row=1, col=i+1)
        fig.add_vline(x=dis[i], line_dash="dash", row=1, col=i+1)

    #fig.update_yaxes(range=[-100,0])
    fig.show()

INR = [70, 0]
res = 1000
for inr in INR:

    inter_positions = np.linspace(0.001, 0.5, res)
    BP = np.zeros((len(inter_positions), res))
    for i, inter_pos in enumerate(inter_positions):
        u, BPi = CalcWmvdr(inter_pos=inter_pos, res=res, INR=inr)
        BP[i] = BPi

    '''
    fig = go.Figure(data=[go.Surface(z=BP, x=u, y=inter_positions)])

    fig.update_layout(title=f'Power Pattern Vs u, INR=70dB', autosize=False,
                      width=500, height=500,
                      margin=dict(l=65, r=50, b=65, t=90), scene=dict(
            xaxis_title=r'$x=u_z$',
            yaxis_title='y=Interfernce Direction',
            zaxis_title='z=Power Pattern'))
    # fig.update_traces(contours_z=dict(show=True, usecolormap=True,
    #                                  highlightcolor="limegreen", project_z=True))
    # fig.update_zaxes(range=[0,8])
    fig.update_layout(
        scene=dict(zaxis=dict(nticks=8))
    )
    fig.show()
    '''
    layout = dict(
        annotations=[dict(
            x=1.08,
            y=1.05,
            align="right",
            valign="top",
            text=r'$|B|^2_{dB}$',
            showarrow=False,
            xref="paper",
            yref="paper",
            xanchor="center",
            yanchor="top"
        )
        ]
    )
    fig = go.Figure(data=
    go.Contour(
        z=20*np.log10(BP), x=u, y=inter_positions, contours_coloring='heatmap'
           ), layout=layout)
    #fig.data[0].colorbar.title.text = r'$|B|^2_{dB}$'
    fig.update_xaxes(range=[-1, 1])
    fig.update_yaxes(title_text='Interfernce Direction')
    fig.update_xaxes(title_text=r'$u_z$')

    fig.update_layout(title=r'$\text{{Power Pattern Vs }} u_z, \text{{ INR={}dB}}$'.format(inr), autosize=False,
                      width=600, height=500)

    fig.show()

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

#For N odd, N=11:

#magnitude of the BP
# a)
N = 11

theta = np.linspace(0, np.pi, 300)
pssay = np.linspace(-np.pi, np.pi, 300)
u_z = np.linspace(-1, 1, 300)
k_z = np.linspace(-np.pi, np.pi, 300)
var = [theta, pssay, u_z, k_z]
names = [r'\theta', r'\Psi', r'u_z',r'k_z']
B_sq_odd_theta = np.abs(1/N*np.divide(np.sin((N/2)*np.pi*np.cos(theta)), np.sin((1/2)*np.pi*np.cos(theta))))
B_sq_odd_pssay = np.abs(1/N*np.divide(np.sin((N/2)*pssay), np.sin((1/2)*pssay)))
B_sq_odd_u_z = np.abs(1/N*np.divide(np.sin((N/2)*np.pi*u_z), np.sin((1/2)*np.pi*u_z)))
B_sq_odd_k_z = np.abs(1/N*np.divide(np.sin((N/2)*-k_z), np.sin((1/2)*-k_z)))
values = [B_sq_odd_theta, B_sq_odd_pssay, B_sq_odd_u_z, B_sq_odd_k_z]
fig = make_subplots(rows=4, cols=1,subplot_titles=(r"$|B| \ vs \ \theta$", r"$|B| \ vs \ \psi$",
                                                   r"$|B| \ vs \ u_z$", r"$|B| \ vs \ k_z$"))

#fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15,15))

for i in range(4):
    a = fig.add_trace(
    go.Scatter(name=f"$|B| \ vs \ {names[i]}$",x=var[i], y=values[i]),
    row=i+1, col=1)
    fig.update_xaxes(title_text=f"${names[i]}$", row=i+1, col=1)
    fig.update_yaxes(title_text="|B|", row=i+1, col=1)

    
    #ax.plot(var[i], values[i], 'b')
    #df = px.data.gapminder().query("country=='Canada'")
    #fig = px.line(var[i], B_sq_odd, title='Life expectancy in Canada')
    #fig.show()
    #ax.set_xlabel(names[i])
    #ax.set_ylabel(f"$B^{2}$")
    #ax.set_title(f"$B^{2}$  Vs  {names[i]}")
    #ax.grid(b=True)
fig.update_layout(height=1500)
fig.update_layout(title_text="N=11")
fig.show()
#fig.tight_layout()


#magnitude of the BP
#Assuming d=1 (lambda=2)
N = 10

theta = np.linspace(0, np.pi, 300)
pssay = np.linspace(-np.pi, np.pi, 300)
u_z = np.linspace(-1, 1, 300)
k_z = np.linspace(-np.pi, np.pi, 300)
var = [theta, pssay, u_z, k_z]
names = [r'\theta', r'\Psi', r'u_z',r'k_z']

B_sq_odd_theta = np.abs(1/N*(1 +  np.exp(1j*N/2*np.pi*np.cos(theta))*np.divide(np.sin((N-1)/2*np.pi*np.cos(theta)), np.sin((1/2)*np.pi*np.cos(theta)))))
B_sq_odd_u_z = np.abs(1/N*(1 + np.exp(1j*N/2*np.pi*u_z)*np.divide(np.sin((N-1)/2*np.pi*u_z), np.sin((1/2)*np.pi*u_z))))
B_sq_odd_k_z = np.abs(1/N*(1 +  np.exp(1j*N/2*-pssay)*np.divide(np.sin((N-1)/2*-pssay), np.sin((1/2)*-pssay))))
B_sq_odd_pssay =np.abs(1/N*(1 +  np.exp(1j*N/2*pssay)*np.divide(np.sin((N-1)/2*pssay), np.sin((1/2)*pssay))))
values = [B_sq_odd_theta, B_sq_odd_pssay, B_sq_odd_u_z, B_sq_odd_k_z]
fig = make_subplots(rows=4, cols=1,subplot_titles=(r"$|B| \ vs \ \theta$", r"$|B| \ vs \ \Psi$",
                                                   r"$|B| \ vs \ u_z$", r"$|B| \ vs \ k_z$"))
#fig = make_subplots(rows=4, cols=1, column_titles=["N=10"])

#fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15,15))

for i in range(4):
    a = fig.add_trace(
    go.Scatter(name=f"$|B| \ vs \ {names[i]}$",x=var[i], y=values[i]),
    row=i+1, col=1)
    fig.update_xaxes(title_text=f"${names[i]}$", row=i+1, col=1)
    fig.update_yaxes(title_text="|B|", row=i+1, col=1)
    
    #ax.plot(var[i], values[i], 'b')
    #df = px.data.gapminder().query("country=='Canada'")
    #fig = px.line(var[i], B_sq_odd, title='Life expectancy in Canada')
    #fig.show()
    #ax.set_xlabel(names[i])
    #ax.set_ylabel(f"$B^{2}$")
    #ax.set_title(f"$B^{2}$  Vs  {names[i]}")
    #ax.grid(b=True)
fig.update_layout(height=1500)
fig.update_layout(title_text="N=10")

fig.show()
#fig.tight_layout()

# b)

N = 11
theta = np.linspace(0, 2*np.pi, 300)
names = ["theta"]
B_sq_odd_theta = 10*np.log10((1/(N**2))*np.square(np.divide(np.sin((N/2)*np.pi*np.cos(theta)), np.sin((1/2)*np.pi*np.cos(theta)))))
#Only the value above -40dB
B_sq_odd_theta[B_sq_odd_theta<-40] = B_sq_odd_theta[B_sq_odd_theta<-40]/10 -40
trace1 = go.Scatterpolar(
    r=B_sq_odd_theta,
    theta=np.linspace(0, 360, 300) , name='N=11')

data = trace1
layout = go.Layout(
    title='Power Pattern in dB, N=11',
    font=dict(
        family='Arial, sans-serif;',
        size=12,
        color='#000'
    ),
    polar=dict(
      radialaxis_tickfont_size = 8,
      angularaxis = dict(
        tickfont_size = 8,
        rotation = 90,
        direction = "clockwise"
      )
))
fig = go.Figure(data=data, layout=layout)
fig.show()

#PowerPattern
N = 10
theta = np.linspace(0, 2*np.pi, 300)
B_sq_odd_theta = np.abs(1/N*(1 +  np.exp(1j*N/2*np.pi*np.cos(theta))*np.divide(np.sin((N-1)/2*np.pi*np.cos(theta)), np.sin((1/2)*np.pi*np.cos(theta)))))
names = ["theta"]
B_sq_odd_theta = 10*np.log10(np.square(B_sq_odd_theta))
#Only the value above -40dB
B_sq_odd_theta[B_sq_odd_theta<-40] = B_sq_odd_theta[B_sq_odd_theta<-40]/100 -40
trace2 = go.Scatterpolar(
    r=B_sq_odd_theta,
    theta=np.linspace(0, 360, 300), name='N=10')


data = trace2
layout = go.Layout(
    title='Power Pattern in dB, N=10',
    font=dict(
        family='Arial, sans-serif;',
        size=12,
        color='#000'
    ),
    polar=dict(
      radialaxis_tickfont_size = 8,
      angularaxis = dict(
        tickfont_size = 8,
        rotation = 90,
        direction = "clockwise"
      )
))
fig = go.Figure(data=data, layout=layout)
fig.show()

layout = go.Layout(
    title='Power Pattern in dB',
    font=dict(
        family='Arial, sans-serif;',
        size=12,
        color='#000'
    ),
    polar=dict(
      radialaxis_tickfont_size = 8,
      angularaxis = dict(
        tickfont_size = 8,
        rotation = 90,
        direction = "clockwise"
      )
))
fig = go.Figure(data=[trace1, trace2], layout=layout)
fig.show()

# c)
#No need
N = 11
theta = np.linspace(0, 2*np.pi, 300)
names = ["theta"]
B_sq_odd_theta = 10*np.log10((1/(N**2))*np.square(np.divide(np.sin((N/2)*(np.pi*np.cos(theta)-np.pi)), np.sin((1/2)*(np.pi*np.cos(theta)-np.pi)))))
#Only the value above -40dB
B_sq_odd_theta[B_sq_odd_theta<-40] = B_sq_odd_theta[B_sq_odd_theta<-40]/100 -40
trace1 = go.Scatterpolar(
    r=B_sq_odd_theta,
    theta=np.linspace(0, 360, 300))

data = trace1
layout = go.Layout(
    title='Power Pattern in dB, N=11',
    font=dict(
        family='Arial, sans-serif;',
        size=12,
        color='#000'
    ),
    polar=dict(
      radialaxis_tickfont_size = 8,
      angularaxis = dict(
        tickfont_size = 8,
        rotation = 90,
        direction = "clockwise"
      )
))
fig = go.Figure(data=data, layout=layout)
fig.show()

#try
#Display of go.Surface

N = 10
d_divide_lambda = np.linspace(0.001, 1, 300)
theta = np.linspace(0, np.pi, 300)
names = ["theta"]
steering = np.tile(2*np.pi*d_divide_lambda,(300,1))
B_sq_even_thet_90 = np.square(np.abs(1/N*(1 +  np.exp(1j*N/2*2*np.pi*np.outer(np.cos(theta),d_divide_lambda))
                                          *np.divide(np.sin((N-1)/2*2*np.pi*np.outer(np.cos(theta),d_divide_lambda)), 
                                           np.sin((1/2)*(2*np.pi*np.outer(np.cos(theta),d_divide_lambda)))))))

B_sq_even_thet_0 = np.square(np.abs(1/N*(1 +  np.exp(1j*N/2*(2*np.pi*np.outer(np.cos(theta),d_divide_lambda) - np.pi))
                                          *np.divide(np.sin((N-1)/2*(2*np.pi*np.outer(np.cos(theta),d_divide_lambda) - np.pi)), 
                                           np.sin((1/2)*(2*np.pi*np.outer(np.cos(theta),d_divide_lambda) - np.pi))))))

B_sq_even_thet_90 =  10*np.log10(B_sq_even_thet_90)
B_sq_even_thet_0 =  10*np.log10(B_sq_even_thet_0)
#B_sq_odd_theta_90 = 10*np.log10((1/(N**2))*np.square(np.divide(np.sin((N/2)*(2*np.pi*np.outer(np.cos(theta),d_divide_lambda))), np.sin((1/2)*(2*np.pi*np.outer(np.cos(theta),d_divide_lambda))))))
#B_sq_odd_theta_0 = 10*np.log10((1/(N**2))*np.square(np.divide(np.sin((N/2)*(2*np.pi*np.outer(np.cos(theta) - 1,d_divide_lambda))), np.sin((1/2)*(2*np.pi*np.outer(np.cos(theta) - 1,d_divide_lambda))))))
#Only the value above -40dB
B_sq_even_thet_90[B_sq_even_thet_90<-40] = B_sq_even_thet_90[B_sq_even_thet_90<-40]/100 -40
B_sq_even_thet_0[B_sq_even_thet_0<-40] = B_sq_even_thet_0[B_sq_even_thet_0<-40]/100 -40
#Broadside, N=11
fig = go.Figure(data=[go.Surface(z=B_sq_even_thet_90, x=d_divide_lambda, y=theta)])

fig.update_layout(title=r"$Power \ Pattern \ Vs \ \frac{d}{\lambda} \ and  \ \theta, Broadside$", autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90), scene = dict(
                    xaxis_title='x=d_divide_lambda',
                    yaxis_title='y=theta',
                    zaxis_title='z=Beam Pattern'))
#fig.update_traces(contours_z=dict(show=True, usecolormap=True,
#                                  highlightcolor="limegreen", project_z=True))
fig.show()


'''
fig = go.Figure(data=
    go.Contour(z=BP, x=u, y=inter_positions))
fig.update_xaxes(range=[-1, 1])
fig.update_yaxes(title_text='Interfernce Direction')
fig.update_xaxes(title_text='u')
fig.update_layout(title=f'Beam Pattern Vs u, INR={inr}dB', autosize=False,
                  width=600, height=500)
'''

#try
#Display of go.Contour

N = 10
d_divide_lambda = np.linspace(0.001, 1, 300)
theta = np.linspace(0, np.pi, 300)
names = ["theta"]
steer_angle = 0
steering = np.tile(2*np.pi*d_divide_lambda,(300,1))
B_sq_even_thet_90 = np.square(np.abs(1/N*(1 +  np.exp(1j*N/2*2*np.pi*np.outer(np.cos(theta),d_divide_lambda))
                                          *np.divide(np.sin((N-1)/2*2*np.pi*np.outer(np.cos(theta),d_divide_lambda)), 
                                           np.sin((1/2)*(2*np.pi*np.outer(np.cos(theta),d_divide_lambda)))))))

B_sq_even_thet_0 = np.square(np.abs(1/N*(1 +  np.exp(1j*N/2*(2*np.pi*np.outer(np.cos(theta) - np.cos(steer_angle),d_divide_lambda)))
                                          *np.divide(np.sin((N-1)/2*(2*np.pi*np.outer(np.cos(theta) - np.cos(steer_angle) ,d_divide_lambda))), 
                                           np.sin((1/2)*(2*np.pi*np.outer(np.cos(theta) - np.cos(steer_angle),d_divide_lambda)))))))

B_sq_even_thet_90 =  10*np.log10(B_sq_even_thet_90)
B_sq_even_thet_0 =  10*np.log10(B_sq_even_thet_0)
#B_sq_odd_theta_90 = 10*np.log10((1/(N**2))*np.square(np.divide(np.sin((N/2)*(2*np.pi*np.outer(np.cos(theta),d_divide_lambda))), np.sin((1/2)*(2*np.pi*np.outer(np.cos(theta),d_divide_lambda))))))
#B_sq_odd_theta_0 = 10*np.log10((1/(N**2))*np.square(np.divide(np.sin((N/2)*(2*np.pi*np.outer(np.cos(theta) - 1,d_divide_lambda))), np.sin((1/2)*(2*np.pi*np.outer(np.cos(theta) - 1,d_divide_lambda))))))
#Only the value above -40dB
B_sq_even_thet_90[B_sq_even_thet_90<-40] = B_sq_even_thet_90[B_sq_even_thet_90<-40]/100 -40
B_sq_even_thet_0[B_sq_even_thet_0<-40] = B_sq_even_thet_0[B_sq_even_thet_0<-40]/100 -40
fig = make_subplots(rows=2, cols=1,subplot_titles=("Broadside"
                                                   ,"Endfire"))
fig.add_trace(
    go.Contour(z=B_sq_even_thet_90.T,  x=theta, y=d_divide_lambda, coloraxis = "coloraxis"),
    row=1, col=1)
fig.add_trace(
    go.Contour(z=B_sq_even_thet_0.T,  x=theta, y=d_divide_lambda, coloraxis="coloraxis"),
    row=2, col=1)
'''
fig.update_layout(title=r"$Beam \ Pattern \ Vs \ \frac{d}{\lambda} \ and  \ \theta$", autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90), scene = dict(
                    xaxis_title='x=d_divide_lambda',
                    yaxis_title='y=theta',
                    zaxis_title='z=Beam Pattern'))
#fig.update_traces(contours_z=dict(show=True, usecolormap=True,
#    '                             highlightcolor="limegreen", project_z=True))'''

#fig.update_xaxes(range=[-1,1])
fig.update_yaxes(title_text=r'$\frac{d}{\lambda}$')
fig.update_xaxes(title_text=r'$\theta$')
fig.update_layout(title=r"$Power \ Pattern \ vs \ \frac{d}{\lambda} \ and  \ \theta$", autosize=False,
                  width=500, height=600, coloraxis_colorscale="plasma")
fig.show()


#Display of go.Surface

N = 11
d_divide_lambda = np.linspace(0.001, 1, 300)
theta = np.linspace(0, np.pi, 300)
names = ["theta"]
steering = np.tile(2*np.pi*d_divide_lambda,(300,1))
B_sq_odd_theta_90 = 10*np.log10((1/(N**2))*np.square(np.divide(np.sin((N/2)*(2*np.pi*np.outer(np.cos(theta),d_divide_lambda))), np.sin((1/2)*(2*np.pi*np.outer(np.cos(theta),d_divide_lambda))))))
B_sq_odd_theta_0 = 10*np.log10((1/(N**2))*np.square(np.divide(np.sin((N/2)*(2*np.pi*np.outer(np.cos(theta) - 1,d_divide_lambda))), np.sin((1/2)*(2*np.pi*np.outer(np.cos(theta) - 1,d_divide_lambda))))))
#Only the value above -40dB
B_sq_odd_theta_90[B_sq_odd_theta_90<-40] = B_sq_odd_theta_90[B_sq_odd_theta_90<-40]/100 -40
B_sq_odd_theta_0[B_sq_odd_theta_0<-40] = B_sq_odd_theta_0[B_sq_odd_theta_0<-40]/100 -40
#Broadside, N=11
fig = go.Figure(data=[go.Surface(z=B_sq_odd_theta_90, x=d_divide_lambda, y=theta)])

fig.update_layout(title=r"$Beam \ Pattern \ Vs \ \frac{d}{\lambda} \ and  \ \theta, Broadside$", autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90), scene = dict(
                    xaxis_title='x=d_divide_lambda',
                    yaxis_title='y=theta',
                    zaxis_title='z=Beam Pattern'))
#fig.update_traces(contours_z=dict(show=True, usecolormap=True,
#                                  highlightcolor="limegreen", project_z=True))
fig.show()

#Endfire, N=11
fig = go.Figure(data=[go.Surface(z=B_sq_odd_theta_0, x=d_divide_lambda, y=theta)])

fig.update_layout(title='Beam Pattern Vs d/lambda and theta, Endfire', autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90), scene = dict(
                    xaxis_title='x=d_divide_lambda',
                    yaxis_title='y=theta',
                    zaxis_title='z=Beam Pattern'))
#fig.update_traces(contours_z=dict(show=True, usecolormap=True,
#                                  highlightcolor="limegreen", project_z=True))

fig.show()

# d)

N=11
d_divide_lambda = np.linspace(0.001, 1, 300)
theta = np.linspace(0, np.pi, 300)
names = ["theta"]
angles_steer = np.radians([0,30,60,90])
traces = [None]*4
fig = make_subplots(rows=2, cols=2,  specs=[[{'is_3d': True}, {'is_3d': True}], [{'is_3d': True}, {'is_3d': True}]],
                    subplot_titles=(r'$\theta_t=0$', r'$\theta_t=30$', r'$\theta_t=60$', r'$\theta_t=90$')
                    )
for j in range(2):
    for i in range(2):
        B_sq_odd_theta_steered = 10*np.log10((1/(N**2))*np.square(np.divide(np.sin((N/2)*(2*np.pi*np.outer(np.cos(theta) - np.cos(angles_steer[j+i]),d_divide_lambda))),
                                                                            np.sin((1/2)*(2*np.pi*np.outer(np.cos(theta) - np.cos(angles_steer[j+i]),d_divide_lambda))))))
        B_sq_odd_theta_steered[B_sq_odd_theta_steered<-40] = B_sq_odd_theta_steered[B_sq_odd_theta_steered<-40]/100 -40
        fig.append_trace(go.Surface(z=B_sq_odd_theta_steered, x=d_divide_lambda, y=theta, coloraxis="coloraxis"), row=i+1, col=j+1)
        fig.update_scenes(
                    xaxis_title='x=d_divide_lambda',
                    yaxis_title='y=theta',
                    zaxis_title='z=Beam Pattern',domain_row=i+1, domain_column=j+1)


fig.update_layout(title='Beam Pattern Vs d/lambda and theta', autosize=True,
                  margin=dict(l=65, r=50, b=65, t=90), coloraxis_colorscale="plasma"
                 )
fig.update_layout(height=1000, width=1000, margin=dict(r=10, l=10, b=10, t=100), )

fig.show()    


N=10
d_divide_lambda = np.linspace(0.001, 1, 300)
theta = np.linspace(0, np.pi, 300)
names = ["theta"]
angles_steer = np.radians([0,30,60,90])
traces = [None]*4
fig = make_subplots(rows=1, cols=4,
                    subplot_titles=(r'$\theta_t=0$', r'$\theta_t=30$', r'$\theta_t=60$', r'$\theta_t=90$'), 
                    horizontal_spacing = 0.08)
for j in range(4):
    
        
    B_sq_odd_theta_steered = np.square(np.abs(1/N*(1 +  np.exp(1j*N/2*(2*np.pi*np.outer(np.cos(theta) - np.cos(angles_steer[j]),d_divide_lambda)))
                                      *np.divide(np.sin((N-1)/2*(2*np.pi*np.outer(np.cos(theta) - np.cos(angles_steer[j]) ,d_divide_lambda))), 
                                       np.sin((1/2)*(2*np.pi*np.outer(np.cos(theta) - np.cos(angles_steer[j]),d_divide_lambda)))))))
    B_sq_odd_theta_steered = 10*np.log10(B_sq_odd_theta_steered)
    B_sq_odd_theta_steered[B_sq_odd_theta_steered<-40] = B_sq_odd_theta_steered[B_sq_odd_theta_steered<-40]/100 -40
    fig.add_trace(
        go.Contour(z=B_sq_odd_theta_steered.T,  x=theta, y=d_divide_lambda, coloraxis="coloraxis"),
        row=1, col=j+1)


fig.update_layout(title=r"$Power \ Pattern \ vs \ \frac{d}{\lambda} \ and  \ \theta$", autosize=True,
                  margin=dict(l=65, r=50, b=65, t=90), coloraxis_colorscale="plasma"
                 )
fig.update_layout(height=500, width=1000, margin=dict(r=40, l=40, b=40, t=100))
fig.update_yaxes(title_text=r'$\frac{d}{\lambda}$')
fig.update_xaxes(title_text=r'$\theta$')

fig.show()    

# e)

pssay = np.linspace(-np.pi, np.pi, 300)
N = 10 #even
B_failure = 1/N*np.exp(4j*pssay)*(2*np.cos(4*pssay) + 2*np.cos(3*pssay) + 1 + 2*np.exp(1.5j*pssay)*np.cos(3.5*pssay)) 
#B_sq_pssay = (1/N**2)*(1 + 2*np.divide(np.sin((N-1)/2*pssay), np.sin((1/2)*pssay))
#                       + np.square(np.divide(np.sin((N-1)/2*pssay), np.sin((1/2)*pssay))))
B_sq_pssay =np.square(np.abs(1/N*(1 +  np.exp(1j*N/2*pssay)*np.divide(np.sin((N-1)/2*pssay), np.sin((1/2)*pssay)))))
B_sq_pssay_failure = np.real(B_failure*np.conjugate(B_failure))
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=(r'$\text{Power Pattern Vs } \Psi$',r'$\text{PowerPattern Vs } \Psi \text{ in dB}$')
                    )
#fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=pssay,
        y=B_sq_pssay,
        name="Without Failure"
    ),  row=1, col=1)

fig.add_trace(
    go.Scatter(
        x=pssay,
        y=B_sq_pssay_failure,
        name="With Failure"
    ), row=1, col=1)
fig.update_yaxes(title_text=r'$|B|^2$', row=1, col=1)

#fig.show()

#fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=pssay,
        y=10*np.log10(B_sq_pssay),
        name="Without Failure"
    ), row=1, col=2)

fig.add_trace(
    go.Scatter(
        x=pssay,
        y=10*np.log10(B_sq_pssay_failure),
        name="With Failure"
    ), row=1, col=2)

fig.update_xaxes(title_text=r'$\Psi$')
fig.update_yaxes(title_text=r'$|B|^2_{dB}$', row=1, col=2)
fig.update_yaxes(range=[-70,1.5], row=1, col=2)

fig.show()
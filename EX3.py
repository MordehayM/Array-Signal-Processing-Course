import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

#Endfire, N=11
Nd_div_lambda = np.linspace(1, 1000, 50000)
c0 = 0.443
HPBW_scan_lim = np.arccos(1 - 2*c0/Nd_div_lambda)
HPBW_endfire = 2*np.arccos(1 - c0/Nd_div_lambda)
HPBW_lower_scan_limit = lambda angle_rad: np.arccos(np.cos(angle_rad) - c0/Nd_div_lambda)
HPBW_greater_scan_limit = lambda angle_rad: np.arccos(np.cos(angle_rad) - c0/Nd_div_lambda) \
                                            - np.arccos(np.cos(angle_rad) + c0/Nd_div_lambda)

scan_lim_theta = np.arccos(1 - c0/Nd_div_lambda) # in radians

fig = go.Figure()

fig.add_trace(
    go.Scatter(x=Nd_div_lambda, y=np.degrees(HPBW_scan_lim), name="Scan limit"))
fig.add_trace(
    go.Scatter(x=Nd_div_lambda, y=np.degrees(HPBW_endfire), name="Endfire"))


fig.update_layout(title='Half-power beamwidth in degress for various angles - log scale',
                   xaxis_title=r'$\frac{d}{\lambda N}$',
                   yaxis_title='Half-power beamwidth in degrees')

de = [2.5, 5, 10, 20, 30, 45, 90]
theta_t = np.radians([2.5, 5, 10, 20, 30, 45, 90])
t = '\theta_t'
for i, steer_angle_rad in enumerate(theta_t):
    trace = np.zeros(50000)
    trace[scan_lim_theta<=steer_angle_rad] = HPBW_greater_scan_limit(steer_angle_rad)[scan_lim_theta<=steer_angle_rad]
    #trace[scan_lim_theta>steer_angle_rad] = HPBW_lower_scan_limit(steer_angle_rad)[scan_lim_theta>steer_angle_rad]
    fig.add_trace(
    go.Scatter(x=Nd_div_lambda[scan_lim_theta<=steer_angle_rad],
               y=np.degrees(trace[scan_lim_theta<=steer_angle_rad]), name=r"$\theta_t = {}^\circ$".format(de[i])))
    #go.Scatter(x=Nd_div_lambda,
    #           y=np.degrees(trace), name=f"Steering angle = {de[i]}"))
    
    
    
    
fig.update_layout(xaxis_type="log", yaxis_type="log")
fig.show()
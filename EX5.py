import matplotlib.pyplot as plt
import numpy as np

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from scipy import signal
import math
from scipy import interpolate

"""L = 151; # FIR order
#numtaps = Length of the filter (number of coefficients, i.e. the filter order + 1)
numtaps = L + 1
f = 16e3
#fs/2 in at analog in f domain is pi at digital in w domain
cutoff = 1/500; # bandwidth
h_fir = signal.firwin(numtaps=numtaps, cutoff=cutoff, window='hamming')
u = np.fft.fftshift(np.fft.fftfreq(numtaps))*f # the k-th frequency
B = np.fft.fftshift(np.fft.fft(h_fir))
BP = np.real(B*np.conjugate(B))"""

'''
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=u,
        y=20*np.log10(BP),
        name="Beam Power, Interference Direction=0.3")
    )



fig.update_layout(title=f'Filter in dB, logscale',
                   xaxis_title='Frequency',
                   yaxis_title='Magnitude (dB)')
#fig.update_layout(xaxis_type="log", yaxis_type="log")
fig.update_yaxes(range=[-300,0])
fig.update_xaxes(range=[-16e03/2,16e03/2])
fig.show()
'''


def NB_signal(fs=16e3, f0=7e03/3, N=2 ** 10):
    """

    :param fs: Sampling frequency in Hz
    :param f0: Carrier frequency in Hz
    :param N: Signal length in samples
    :return: real-valued NB signals (bandlimited white noise)
    """

    L = 151  # FIR order
    # numtaps = Length of the filter (number of coefficients, i.e. the filter order + 1)
    numtaps = L + 1
    f = 16e3
    # fs/2 in at analog in f domain is pi at digital in w domain
    cutoff = 1 / 500  # bandwidth
    h_fir = signal.firwin(numtaps=numtaps, cutoff=cutoff, window='hamming')  # Lowpass filter

    w0_d = 2 * np.pi * f0 / fs  # discrete center frequency

    carrier = np.exp(1j * w0_d * np.arange(N))
    s = np.random.normal(size=N)
    s_nb = signal.lfilter(h_fir, 1, s)  # narrow band white noise
    # sa = s_nb+j*hilbert(s_nb); # analytical signal no carrier
    sa = signal.hilbert(s_nb)  # analytical signal no carrier
    sa_c = sa * carrier  # analytical signal with carrier

    y = sa_c.real

    return y, h_fir


def frac_delay(x, delay, fs):
    """

    :param x: discrete input signal
    :param delay: signal delay in seconds
    :param fs: sampling frequency in Hz
    :return: Delayed discrete input signal
    """
    tau_disc = delay*fs   # Delay in samples
    # calculation of integer delay (quantizatin)
    res = 500 # precision of the quantization.
    temp = round(tau_disc*res)
    tau_num = temp/math.gcd(temp, res)
    tau_den = res/math.gcd(temp, res)
    y = sub_delay(x, tau_num, tau_den)
    return y


def sub_delay(x, N, D):
    """
    Delay of a time series, by sub-sample
    :param x: time series to be delayed
    :param N: N/D - Delay amount
    :param D: N/D - Delay amount
    :return: Delay of a time series, by sub-sample
    """
    i = np.arange(len(x))
    #f = interpolate.interp1d(i, x, kind='linear')
    gaps = len(x) - 1
    new_num_samples = int(gaps*(D-1) + len(x))
    new_index_predict = np.linspace(i[0], i[-1], new_num_samples)
    #z = f(new_index_predict) #we can do also this command: np.interp(new_index_predict, i, x)
    z = np.interp(new_index_predict, i, x)
    zz = delay(z, N)
    if D==1:
        return zz
    y = signal.decimate(zz, int(D), ftype='fir')
    return y


def delay(x, D):
    """

    :param x: the original signal
    :param D: amount of delay. if D > 0 there are D zeros in the begining and the last D samples are lost.
                        if D < 0 the first D samples are lost and there are D zeros in the end.
    :return: The delayed version of x
    """
    D = int(D)
    x_delayed = np.zeros_like(x)

    if D > 0:
        x_delayed[D:] = x[:len(x) - D]

    elif D <= 0:
        x_delayed[:len(x)-abs(D)] = x[abs(D):len(x)]

    return x_delayed

#Generate a narrowband source signal
NarrowB_signal, h_fir = NB_signal()
print("Finish creating NarrowB_signal")
#NarrowB_signal
d = 0.05 # [m]
N = 10
c = 342 #[m/s]
fs = 16e03
f0 = 7e03/3
NarrowB_signal_delayed = np.zeros((N, len(NarrowB_signal)))
theta_deg = np.arange(0,95,5) #from 0 up to 90
theta_rad = np.radians(theta_deg)

NarrowB_reco = np.zeros((N, len(NarrowB_signal)))
#NarrowB_signal_delayed_rev = np.zeros((N, len(NarrowB_signal)))
steering = [0, 90]
names = ["Endfire", "Broadside"]
fig = go.Figure()
for name, steer in zip(names, steering):
    OIR_arr = []
    beampattern_f0 = []
    #steer = 2 * np.pi / c * d * np.cos(np.radians(steer)) * np.arange(N)
    for theta in theta_rad:
        print(theta)
        delays = -1 / c * d * np.cos(theta) * np.arange(N)
        for i, de in enumerate(delays):
            NarrowB_signal_delayed[i] = frac_delay(NarrowB_signal, de, fs)
            #NarrowB_signal_delayed_rev[i] = frac_delay(NarrowB_signal_delayed[i], -steer[i], fs) #reverse

            #NarrowB_reco[i] = frac_delay(NarrowB_signal_delayed[i], -de, fs)
        ##Processing signal with DSB
        NarrowB_signal_delayed_h = signal.hilbert(NarrowB_signal_delayed)
        W_dsb = np.expand_dims(np.exp(-2j * np.pi * f0 / c * d * np.cos(np.radians(steer)) * np.arange(N)), 1)
        #W_dsb = np.ones((N,1))
        NarrowB_reco = 1 / N * W_dsb.T @ NarrowB_signal_delayed_h
        NarrowB_reco_real = NarrowB_reco.real
        #output_power = np.real(NarrowB_reco@np.conjugate(NarrowB_reco.T))
        ## Calc output to input power, OIR
        #input_power = np.var()
        input_power = np.sum(abs(NarrowB_signal) ** 2)
        output_power = np.sum(abs(NarrowB_reco_real) ** 2)
        OIR = output_power / input_power
        OIR_arr.append(OIR)
        #print(OIR)
        mono_f0 = np.expand_dims(np.exp(2j * np.pi * f0 / c * d * np.cos(theta) * np.arange(N)), 1)
        response_mono_f0 = 1 / N * W_dsb.T @ mono_f0
        response_mono_f0_real = response_mono_f0.real
        response_mono_f0_real = np.abs(response_mono_f0_real.squeeze().item())
        beampattern_f0.append(response_mono_f0_real)


    fig.add_trace(go.Scatter(x=theta_deg, y=beampattern_f0,
                        mode='lines',
                        name=f'Mono {name} DSB'))
    fig.add_trace(go.Scatter(x=theta_deg, y=OIR_arr,
                        mode='lines',
                        name=f'{name} DSB'))

fig.update_layout(title=f'OIR Vs Directions of arrival',
                       xaxis_title=r'$\theta [deg]$',
                       yaxis_title='OIR')
fig.show()



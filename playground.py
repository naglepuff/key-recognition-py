import matplotlib.pyplot as plt 
import scipy.fftpack as fft 
from scipy.io import wavfile 
import numpy as np 

def find_nearest_index(array, value):
    array = np.asarray(array) 
    index = (np.abs(array - value)).argmin() 
    return index 


fs, data = wavfile.read('i_iv_v_keyC.wav') 
a = data.T[0] 

Y = fft.rfft(a) 
x = fft.fftfreq(Y.size)
freqs_hz = x * fs 

c3_index = find_nearest_index(freqs_hz, 130.8)

print(Y[c3_index]) 

index = np.argmax(np.abs(Y)) 
freq = x[index] 
freq_hz = abs(freq * fs) 
print(Y[index]) 

# plt.plot(Y[4300:4500])  
# plt.show()
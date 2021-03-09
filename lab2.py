import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sg

fs = 48e3
frequency1 = 20
frequency2 = 200
signalLength = 1 # in seconds
timeArray = np.arange(0, signalLength, 1/fs)

print(timeArray.size)

signalA = np.sin(timeArray * 2 * np.pi * frequency1)
signalB = np.sin(timeArray * 2 * np.pi * frequency2)
signalOutput = signalA + signalB

powerSpectrum = np.square(np.abs(np.fft.rfft(signalOutput))/ len(timeArray))
frequencyArray = np.linspace(0, fs/2, len(powerSpectrum))

peaks = sg.find_peaks(powerSpectrum, threshold=0.2)
print(peaks)


plt.figure()
plt.semilogx(frequencyArray, powerSpectrum)
plt.savefig('./lab2data/zad1.png', format='png')





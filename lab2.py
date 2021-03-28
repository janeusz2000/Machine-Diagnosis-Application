import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sg
from nptdms import TdmsFile

def Spectrum(signal):
   return np.square(np.abs(np.fft.fft(signal)) / signal.size)

print("========= ZAD 1 ==========")

frequency1 = 30
frequency2 = 200
fs = int(48e3)
signalLength = 1 # in seconds
timeArray = np.arange(0, signalLength, 1/fs)
signalA = np.sin(timeArray * 2 * np.pi * frequency1)
signalB = np.sin(timeArray * 2 * np.pi * frequency2)
signalOutput = signalA + signalB

maxAmplitude = np.max(signalOutput)
print("Max amplitude: {}".format(maxAmplitude))
powerSpectrum = np.square(np.abs(np.fft.rfft(signalOutput))/ len(timeArray))
frequencyArray = np.linspace(0, fs/2, len(powerSpectrum))

plt.figure()
plt.semilogx(frequencyArray, powerSpectrum)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.savefig('./lab2data/zad1.png', format='png')

print("======== ZAD 2 ==========")
peaks = sg.find_peaks(powerSpectrum, threshold=0.005)
print("Peaks where found at: {}".format(peaks[0]))

print("========= ZAD 3 ==========")
signalOutput = list()
time = list()

with TdmsFile.open("./lab2data./Signal_GenSig.tdms") as tdms_file:
      current = 1
      group = tdms_file["Untitled"]
      all_group_channels = group.channels();
      channel = all_group_channels[0]
      signalOutput= channel[:]
      time = channel.time_track()

fs = int((time[1] - time[0])**-1)
print(fs)

signalLength = 1 # in seconds
timeArray = np.arange(0, signalLength, 1/fs)

print(timeArray.size)
maxAmplitude = np.max(signalOutput)
print("Max amplitude: {}".format(maxAmplitude))
powerSpectrum = np.square(np.abs(np.fft.rfft(signalOutput))/ len(timeArray))
frequencyArray = np.linspace(0, fs/2, len(powerSpectrum))

peaks = sg.find_peaks(powerSpectrum, threshold=0.001)
print(peaks)

for peak in peaks[0]:
   frequency = frequencyArray[peak]
   print("Frequency: {}".format(frequency))


plt.figure()
plt.semilogx(frequencyArray, powerSpectrum)
plt.savefig('./lab2data/zad3WidmoOryginalne.png', format='png')


# RECONSTRUCTION
signalA = 10.0 * np.sin(timeArray * 2 * np.pi * 14.0)
signalA += 6.0 * np.sin(timeArray * 2 * np.pi * 125.0)
signalA += 4.0 * np.sin(timeArray * 2 * np.pi * 328.0)
signalA /= max(signalA)
signalA *= maxAmplitude
plt.figure()
plt.plot(signalA)
plt.savefig('./lab2data/zad3SignalReconstructed.png', format='png')

powerSpectrum = np.square(np.abs(np.fft.rfft(signalA))/ len(timeArray))
frequencyArray = np.linspace(0, fs/2, len(powerSpectrum))
plt.figure()
plt.semilogx(frequencyArray, powerSpectrum)
plt.savefig('./lab2data/zad3WidmoReconstructed.png', format='png')

print("====== ZAD 4 ======")

plt.figure(figsize=(10,10))
plt.subplot(2, 1, 1)
plt.title("Widmo Amplitudowo-częstotliwościowe")
plt.ylabel("Amplituda")
plt.xlabel("Częstotliwość [Hz]")
spectrum = Spectrum(signalOutput)
frequencyArray = np.linspace(0, fs/2, len(spectrum))
plt.grid(True, which='both')
plt.plot(frequencyArray, spectrum)

plt.subplot(2, 1, 2)
plt.title("Widmo fazowo-czętotliwościowe")
plt.phase_spectrum(signalOutput, Fs=fs)
plt.grid(True, which='both')
plt.tight_layout(pad=1.1)
plt.savefig('./lab2data/zad4.png', format='png')

plt.figure()
plt.title("Widmo Uśredniane amplitudowo")
plt.xlabel("częstotliwość [Hz]")
plt.ylabel("Magnitude")
[frequencies, Pxx] = sg.welch(signalOutput, fs=fs, window=
                              'hamming')
plt.plot(frequencies, Pxx)
plt.savefig('./lab2data/zad4Averaged.png', format='png')
















from nptdms import TdmsFile
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sg

global fs
fs = 51200

def Spectrum(signal):
   return np.square(np.abs(np.fft.rfft(signal)) / signal.size)

dictionary = { "KWP/ai0" : "przyspieszenie (os x) - kierunek prostopadły do osi wału poziomy",
"KWP/ai1": "przyśpieszenie (os Y) - kierunek prostopadły do osi wału pionowy",
"KWP/ai2": "przyśpieszenie (os Z) - kierunek równoległy do osi wału poziomy",
"KWP/ai3" : "znacznik obrotów - jeden znacznik na obrót",
}

channel_length = 0
for key in dictionary.keys():

   with TdmsFile.open("./files/lab1/wentylator-dobry.tdms") as tdms_file:
      current = 1
      group = tdms_file["Untitled"]
      channel = group[key]

      rmsValues = list()
      peakValues = list()
      crestFactorValues = list()
      stft = list()

      dataSize = fs
      while dataSize == fs:
         data = channel[(current-1)*fs: current * fs]
         dataSize = data.size

         rms_value = np.sqrt(np.dot(data, data) / data.size)
         print("RMS value %f" % rms_value)
         rmsValues.append(rms_value)

         peak_value = np.max(data)
         print("Peak value %f" % peak_value)
         peakValues.append(peak_value)

         crest_factor = np.abs(peak_value) / np.abs(rms_value)
         print("crest Factor value %f" % crest_factor)
         crestFactorValues.append(crest_factor)

         stft = data

         current += 1

   plt.figure(figsize=(12, 16))
   plt.subplot(4, 1, 1)
   plt.title("Rms Values: %s" % dictionary[key])
   plt.xlabel("Time [s]")
   plt.ylabel("Rms Value [g]")
   plt.grid("True")
   plt.plot(np.arange(len(rmsValues)), rmsValues)

   plt.subplot(4, 1, 2)
   plt.title("Peak Values: %s" % dictionary[key])
   plt.xlabel("Time [s]")
   plt.ylabel("Amplitude [g]")
   plt.grid("True")
   plt.plot(np.arange(len(peakValues)), peakValues)

   plt.subplot(4, 1, 3)
   plt.title("Crest Factor Values: %s" % dictionary[key])
   plt.xlabel("Time [s]")
   plt.ylabel("Crest Factor [-]")
   plt.grid("True")
   plt.plot(np.arange(len(crestFactorValues)), crestFactorValues)

   plt.subplot(4, 1, 4)
   data_fft = np.abs(np.real(np.fft.rfft(stft)))
   plt.title("STFT: %s" % dictionary[key])
   plt.xlabel("Frequency [Hz]")
   plt.ylabel("Amplitude [g]")
   plt.grid("True", which='both')
   plt.semilogx(np.arange(len(data_fft)) / 2 / fs, data_fft / fs)


   plt.tight_layout()
   plt.savefig("./data/%s.png" % key[4:], format='png')


   plt.figure(figsize=(10,10))
   plt.subplot(2, 1, 1)
   plt.title("Widmo Amplitudowo-częstotliwościowe: {}".format(dictionary[key]))
   plt.ylabel("Amplituda")
   plt.xlabel("Częstotliwość [Hz]")
   spectrum = Spectrum(stft)
   frequencyArray = np.linspace(0, fs/2, len(spectrum))
   plt.grid(True, which='both')
   plt.semilogx(frequencyArray, spectrum)

   plt.subplot(2, 1, 2)
   plt.title("Widmo fazowo-czętotliwościowe: {}".format(dictionary[key]))
   plt.phase_spectrum(stft, Fs=fs)
   plt.grid(True, which='both')
   plt.tight_layout(pad=1.1)
   plt.savefig('./lab2data/zad4{}.png'.format(dictionary[key]), format='png')

   peaks = sg.find_peaks(spectrum, threshold=0.05 * np.max(spectrum))
   print("in: \"{}\" peaks where found at: {}".format(dictionary[key], [str(i) + "Hz" for i in peaks[0]]))

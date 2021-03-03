from nptdms import TdmsFile
import matplotlib.pyplot as plt
import numpy as np

global fs
fs = 51200

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
   plt.xlabel("Time [s]")
   plt.ylabel("Amplitude [g]")
   plt.grid("True", which='both')
   plt.semilogx(np.arange(len(data_fft)) / 2 / fs, data_fft / fs)


   plt.tight_layout()
   plt.savefig("./data/%s.png" % key[4:], format='png')


# for key in dictionary.keys():
#    group = tdms_file["Untitled"]
#    channel = group[key]
#    channel_data = channel[:]

#

#    currentLimit = 1
#    data = list()
#    while (currentLimit - 1) * fs < len(channel_data):
#       if (currentLimit * fs) > len(channel_data):
#          data = channel_data[(currentLimit-1) * fs:]
#       else:
#          data = channel_data[(currentLimit-1) * fs: (currentLimit) * fs]

#




# group = tdms_file['group name']
# channel = group['channel name']
# channel_data = channel[:]
# channel_properties = channel.properties
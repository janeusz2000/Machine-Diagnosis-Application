import numpy as np
import matplotlib.pyplot as plt
import nidaqmx as ni
import time
import keyboard
import queue
import threading
import logging
import json
import sys
import scipy.signal as sg

from enum import Enum
from datetime import datetime
from nptdms import TdmsWriter, ChannelObject, TdmsFile

logging.getLogger().setLevel(logging.INFO)

# PROGRAM CONFIGURATION
ACQUISITION = False
READING = True
SAVE_TO_JSON = True
FAULT_DETECTION_THRESHOLD = 0.2
PATH_TO_TDMS = "./lab3App/lab3Went/went_OK.tdms"
DATABASE_PATH = "./ApplicationData/database"

# !This must be checked everytime we run diagnosis
sensorDictionary = {
   0 : "acc/x - przyśpieszenie (oś x) - kierunek prostopadły do osi wału poziomy",
   1 : "acc/y - przyśpieszenie (oś Y) - kierunek równoległy do osi wału poziomy",
   2 : "acc/z - przyśpieszenie (oś Z) - kierunek prostopadły do osi wału pionowy",
}

GRAVITY = 9.8
CURRENT_SENSOR = 0
CURRENT_SENSOR_LOCK = threading.Lock()
LOCK = threading.Lock()
GUILOCK = threading.Lock()

class DiagnosisFlags(Enum):
   DIAGNOSIS_INIT = 0
   PEAKS_FREQUENCIES_ARE_OK = 1
   PEAKS_FREQUENCIES_ARE_INVALID = 2
   AMPLITUDES_ARE_OK = 3
   AMPLITUDES_ARE_INVALID = 4

PERFECT_DIAGNOSIS = [
   DiagnosisFlags.PEAKS_FREQUENCIES_ARE_OK,
   DiagnosisFlags.AMPLITUDES_ARE_OK]

diagnosisDictionary = {
   DiagnosisFlags.DIAGNOSIS_INIT : "Diagnosis INIT!",
   DiagnosisFlags.PEAKS_FREQUENCIES_ARE_OK : "Peaks frequencies are OK!",
   DiagnosisFlags.PEAKS_FREQUENCIES_ARE_INVALID : "Peaks Frequencies are INVALID!",
   DiagnosisFlags.AMPLITUDES_ARE_OK : "Amplitudes are OK!",
   DiagnosisFlags.AMPLITUDES_ARE_INVALID : "Amplitudes are INVALID!",
}

class DiagnosisTracker:
   def __init__(self):
      self.peaks = None
      self.fft = None


   def runDiagnosis(self, buffer):
      outputFlags = list()

      if self.peaks is None or self.fft is None:
         outputFlags.append(self.getInitDiagnosisFlag())
         spec = Spectrum(buffer)
         self.peaks = sg.find_peaks(spec)[0]
         self.fft = spec
         return outputFlags
      else:
         outputFlags.append(self.checkPeaksFrequencies(buffer))
         # This does not help to detect fault of the device:
         # outputFlags.append(self.checkPeaksAmplitudes(buffer))
      return outputFlags

   def getInitDiagnosisFlag(self):
      return DiagnosisFlags.DIAGNOSIS_INIT

   def checkPeaksFrequencies(self, buffer):
      spec = Spectrum(buffer)
      peakDetectionThreshold = np.max(spec) * FAULT_DETECTION_THRESHOLD
      tempPeaks = sg.find_peaks(spec, threshold=peakDetectionThreshold)[0]
      if len(tempPeaks) == len(self.peaks) and \
         sorted(tempPeaks) == sorted(self.peaks):
         return DiagnosisFlags.PEAKS_FREQUENCIES_ARE_OK
      else:
         self.peaks = tempPeaks
         return DiagnosisFlags.PEAKS_FREQUENCIES_ARE_INVALID

   def checkPeaksAmplitudes(self, buffer):
      spec = Spectrum(buffer)
      tempSum = np.sum(np.abs(self.fft - spec))

      self.fft = spec
      if tempSum < FAULT_DETECTION_THRESHOLD * np.max(spec):
         return DiagnosisFlags.AMPLITUDES_ARE_OK
      else:
         return DiagnosisFlags.AMPLITUDES_ARE_INVALID


def logDiagnosis(flagList, sensorNumber):
   loggingStream = ""
   isEverythingAlright = True

   for flagPosition in range(len(flagList)):
      if flagList[flagPosition] not in PERFECT_DIAGNOSIS:
         isEverythingAlright = False

      loggingStream += diagnosisDictionary[flagList[flagPosition]]

      if flagPosition + 1 < len(flagList):
         loggingStream += "\n"

   if isEverythingAlright:
      logging.info("OK")
   else:
      logging.info("DIAGNOSIS: \n" + loggingStream)


def Spectrum(signal):
   return np.square(np.abs(np.fft.rfft(signal)) / signal.size)


def Freq(signal):
   return np.arange(len(signal) / 2 + 1)


def Time(signal):
   return np.arange(len(signal))


def animationGui(que):
   plt.ion()
   fig = plt.figure(figsize=(10, 10))

   ax1 = fig.add_subplot(211)
   ax1.set_title("Time Domain")
   ax1.set_xlabel("Buffer Sample")
   ax1.set_ylabel("Amplitude $\dfrac{m}{s^2}$")
   ax1.grid(True, which='both')

   ax2 = fig.add_subplot(212)
   ax2.grid(True, which='both')
   ax2.set_title("Frequency Domain")
   ax2.set_xlabel("Frequency [Hz]")
   ax2.set_ylabel("Amplitude $\dfrac{m}{s^2}$")

   graph1, graph2 = None, None
   first = True
   data = np.array([])
   while not keyboard.is_pressed('q'):
      if first and not que.empty():

         with GUILOCK:
            data = que.get()

         first = False
         spectrum = Spectrum(data)
         frequency = Freq(data)
         graph1, = ax1.plot(Time(data), data * GRAVITY)
         graph2, = ax2.semilogx(frequency, spectrum * GRAVITY)

         with CURRENT_SENSOR_LOCK:
            fig.suptitle(sensorDictionary[CURRENT_SENSOR], fontsize=16)

         plt.show()

      elif not que.empty():

         with GUILOCK:
            data = que.get()

         spectrum = Spectrum(data)
         graph1.set_ydata(data)
         ax1.relim()
         ax1.autoscale_view()
         graph2.set_ydata(spectrum)
         ax2.relim()
         ax2.autoscale_view()

         with CURRENT_SENSOR_LOCK:
            fig.suptitle(sensorDictionary[CURRENT_SENSOR], fontsize=16)

         fig.canvas.draw()

      fig.canvas.flush_events()
      time.sleep(0.1)


def saveToDatabase(data, tdms_writer=None):
   currentTime = datetime.now().strftime("%H:%M:%S")
   logging.info(f"Saving data at time: {currentTime}");

   if SAVE_TO_JSON:
      outputdata = {
         "time": currentTime,
         "data": data.tolist()}

      with open(DATABASE_PATH + ".js", 'a') as f:
         f.write(json.dumps(outputdata) + ",\n")

   if tdms_writer is not None:
      channel = ChannelObject('Undefined', 'Channel1', data)
      tdms_writer.write_segment([channel])


def acquiringData(que):
   t = threading.current_thread()
   name = str(t.getName())

   logging.info(f"ACQUISITION Thread: {name} started!")

   system = ni.system.system.System.local()
   taskName = system.tasks.task_names[0]
   logging.info(f"Task Name: {taskName}")
   prepareTask = ni.system.storage.persisted_task.PersistedTask(
      taskName)
   task = prepareTask.load()

   while not keyboard.is_pressed('q'):

      with LOCK:
         que.put(np.array(task.read(ni.constants.READ_ALL_AVAILABLE)))

      time.sleep(0.3)
   task.stop()

   logging.info(f"ACQUISITION Thread: {name} ended!")


def savingThread(que, guiQueue):
   t = threading.current_thread()
   name = str(t.getName())
   logging.info(f"GUI Thread: {name} started!")

   with TdmsWriter(DATABASE_PATH + ".tdms") as writer:
      while not keyboard.is_pressed('q'):

         with LOCK:
            if not que.empty():
               temp = que.get()
               saveToDatabase(data=temp, tdms_writer=writer)
               guiQueue.put(temp)

         time.sleep(0.2)

   logging.info(f"GUI Thread: {name} ended!")


def readTDMS(pathToData, guiQueue):

   with TdmsFile.open(pathToData) as tdms_file:
      for group in tdms_file.groups():
         all_group_channels = group.channels()
         sensorNumber = 0
         for channel in all_group_channels:
            logging.info("Reading sensor: {}".format(sensorDictionary[sensorNumber]))

            tracker = DiagnosisTracker()

            for chunk in channel.data_chunks():
               channel_chunk_data = chunk[:]

               with GUILOCK:
                  guiQueue.put(channel_chunk_data)

               diagnosisFlags = tracker.runDiagnosis(channel_chunk_data)
               logDiagnosis(diagnosisFlags, sensorNumber)

               time.sleep(1.0)
               if keyboard.is_pressed('q'):
                  sys.exit()

            with CURRENT_SENSOR_LOCK:
               global CURRENT_SENSOR
               CURRENT_SENSOR += 1
            sensorNumber += 1

if __name__ == "__main__":

   dataQueue = queue.Queue()
   guiQueue = queue.Queue()

   threads = list()
   if ACQUISITION:

      if SAVE_TO_JSON:
         # Clearing and creating new database inside .js
         # file at DATABASE_PATH

         with open(DATABASE_PATH + ".js", "w+") as f:
            f.write("const data = [")

      threads.append(threading.Thread(
         target=acquiringData, args=(dataQueue,)))
      threads.append(threading.Thread(
         target=savingThread, args=(dataQueue, guiQueue)))

   if READING:
      pathToData = PATH_TO_TDMS
      threads.append(threading.Thread(
         target=readTDMS, args=(pathToData, guiQueue)))

   for thread in threads:
      thread.start()

   animationGui(guiQueue)

   for thread in threads:
      thread.join()

   if SAVE_TO_JSON and ACQUISITION:
      # Closing database
      with open(DATABASE_PATH + ".js", "a") as f:
         f.write("]")

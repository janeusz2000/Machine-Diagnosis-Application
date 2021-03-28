import numpy as np
import matplotlib.pyplot as plt
import nidaqmx as ni
import time
import keyboard
import queue
import threading
import logging
import json

from datetime import datetime
from nptdms import TdmsWriter, ChannelObject


logging.getLogger().setLevel(logging.INFO)


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
   ax1.set_ylabel("Amplitude $m/s^2")
   ax1.grid(True, which='both')

   ax2 = fig.add_subplot(212)
   ax2.grid(True, which='both')
   ax2.set_title("Frequency Domain")
   ax2.set_xlabel("Frequency [Hz]")
   ax2.set_ylabel("Amplitude $m/s^2")

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
         graph1, = ax1.plot(Time(data), data)
         graph2, = ax2.semilogx(frequency, spectrum)
         plt.show()
      elif not que.empty():
         with GUILOCK:
            data = que.get()
         spectrum = Spectrum(data)
         graph1.set_ydata(data)
         graph2.set_ydata(spectrum)
         fig.canvas.draw()
      fig.canvas.flush_events()
      time.sleep(0.1)


def saveToDatabase(data, tdms_writer=None):
   currentTime = datetime.now().strftime("%H:%M:%S")
   logging.info(f"Saving data at time: {currentTime}");

   if JSON:
      outputdata = {
         "time": currentTime,
         "data": data.tolist()}
      with open(DATABASEPATH + ".js", 'a') as f:
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
   prepareTask = ni.system.storage.persisted_task.PersistedTask(taskName)
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
   with TdmsWriter(DATABASEPATH + ".tdms") as writer:
      while not keyboard.is_pressed('q'):
         with LOCK:
            if not que.empty():
               temp = que.get()
               saveToDatabase(data=temp, tdms_writer=writer)
               guiQueue.put(temp)
         time.sleep(0.2)
      logging.info(f"GUI Thread: {name} ended!")


def readTDMS(pathToData, guiQueue):

   # TODO: create function that will iterate
   # for each second in given tdms file at path
   # and put data into guiQueue
   with GUILOCK:
      pass


if __name__ == "__main__":

   global DATABASEPATH
   global DATAQUEUE
   global JSON
   global LOCK
   global GUILOCK
   global ACQUISITION
   global READING

   ACQUISITION = True
   READING = False
   DATABASEPATH = "./ApplicationData/database"
   LOCK = threading.Lock()
   GUILOCK = threading.Lock()
   JSON = True

   dataQueue = queue.Queue()
   guiQueue = queue.Queue()

   threads = list()
   if ACQUISITION:

      if JSON:
         # Clearing and creating new database inside .js file at DATABASEPATH
         with open(DATABASEPATH + ".js", "w+") as f:
            f.write("const data = [")

      threads.append(threading.Thread(
         target=acquiringData, args=(dataQueue,)))
      threads.append(threading.Thread(
         target=savingThread, args=(dataQueue, guiQueue)))

   if READING:
      pathToData = ""
      threads.append(threading.Thread(
         target=readTDMS, args=(pathToData, guiQueue)))

   for thread in threads:
      thread.start()

   animationGui(guiQueue)

   for thread in threads:
      thread.join()

   if JSON and ACQUISITION:
      # Closing database
      with open(DATABASEPATH + ".js", "a") as f:
         f.write("]")

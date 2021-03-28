import numpy as np
import matplotlib.pyplot as plt
import nidaqmx as ni
import nidaqmx.stream_readers as streams
import time
import keyboard
import queue
import threading
import logging
import json

from datetime import datetime
from nptdms import TdmsFile

logging.getLogger().setLevel(logging.INFO)

def Spectrum(signal):
   return np.square(np.abs(np.fft.rfft(signal)) / signal.size)

def Freq(signal):
   return np.arange(len(signal) / 2 + 1)

def Time(signal):
   return np.arange(len(signal))

def animationGui(que):
   plt.ion()
   fig = plt.figure(figsize=(10,10))

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

   while not keyboard.is_pressed('q'):
      if first and not que.empty():
         data = que.get()
         first = False
         spectrum = Spectrum(data)
         frequency = Freq(data)
         graph1, = ax1.plot(Time(data), data)
         graph2, = ax2.semilogx(frequency, spectrum )
         plt.show()
      elif not que.empty():
         data = que.get()
         spectrum = Spectrum(data)
         graph1.set_ydata(data)
         graph2.set_ydata(spectrum )
         fig.canvas.draw()
         fig.canvas.flush_events()
      time.sleep(0.1)

def saveToDatabase(data):
   currentTime = datetime.now().strftime("%H:%M:%S")
   logging.info(f"Saving data at time: {currentTime}");
   data = {
      "time" : currentTime,
      "data" : data.tolist()}
   with open(databasePath, 'a') as f:
      f.write(json.dumps(data) + ",\n")

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
   while not keyboard.is_pressed('q'):
      with LOCK:
         if not que.empty():
            temp = que.get()
            saveToDatabase(data=temp)
            guiQueue.put(temp)
      time.sleep(0.2)
   logging.info(f"GUI Thread: {name} ended!")

global databasePath
databasePath = "./ApplicationData/database.js"

global LOCK
LOCK = threading.Lock()

global dataQue
dataQue = queue.Queue()

guiQueue = queue.Queue()

# Clearing and creating new database
with open(databasePath, "w+") as f:
   f.write("const data = [")

threads = list()
threads.append(threading.Thread(target=acquiringData, args=(dataQue,)))
threads.append(threading.Thread(target=savingThread, args=(dataQue, guiQueue)))

for thread in threads:
   thread.start()

animationGui(guiQueue)

for thread in threads:
   thread.join()

# Closing database
with open(databasePath, "a") as f:
   f.write("]")
import numpy as np
import matplotlib as plt
import nidaqmx as ni
import nidaqmx.stream_readers as streams
import time
import keyboard
import queue
import threading
import logging

logging.getLogger().setLevel((logging.INFO))

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

def guiThread(que):
   t = threading.current_thread()
   name = str(t.getName())
   logging.info(f"GUI Thread: {name} started!")
   while not keyboard.is_pressed('q'):
      with LOCK:
         if not que.empty():
            print(que.get())
      time.sleep(0.2)
   logging.info(f"GUI Thread: {name} ended!")


global LOCK
LOCK = threading.Lock()

global dataQue
dataQue = queue.Queue()

threads = list()
threads.append(threading.Thread(target=acquiringData, args=(dataQue,)))
threads.append(threading.Thread(target=guiThread, args=(dataQue,)))

for thread in threads:
   thread.start()

for thread in threads:
   thread.join()




# with ni.Task() as task:
#    task.ai_channels.add_ai_voltage_chan("Dev1/ai0")
#    task.read()


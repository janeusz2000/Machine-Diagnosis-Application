import numpy as np
import matplotlib as plt
import nidaqmx as ni
import nidaqmx.stream_readers as streams
import time
import keyboard


fs = 25.6e3
bufferSize = int(fs);

system = ni.system.system.System.local()
taskName = system.tasks.task_names[0]
print(f"Task Name: {taskName}")
prepareTask = ni.system.storage.persisted_task.PersistedTask(taskName)
task = prepareTask.load()
while not keyboard.is_pressed('q'):
   print(np.array(task.read(ni.constants.READ_ALL_AVAILABLE)))
   time.sleep(0.1)
task.stop()




# with ni.Task() as task:
#    task.ai_channels.add_ai_voltage_chan("Dev1/ai0")
#    task.read()


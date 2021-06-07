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
REFERENCE = True
ACQUISITION = False
READING = True
SAVE_TO_JSON = True
FAULT_DETECTION_THRESHOLD = 1.5
PATH_TO_REFERENCE_TDMS = "./source/dane_analiza_rzedow/went_dobry.tdms"
PATH_TO_TDMS = "./source/dane_analiza_rzedow/went_uszk.tdms"
DATABASE_PATH = "./ApplicationData/database"

# Order Analiz parameters
NUMBER_OF_SAMPLES_PER_ROTATION = 50
DETECTION_THRESHOLD = -0.48
LASER_SAMPLE_RATE = 51200
SYNCHRONIZATION_CHANNEL = 3
RESAMPLED_BUFFER_SIZE = 2500

# !This must be checked everytime we run diagnosis
sensorDictionary = {
    0: "acc/x - przyśpieszenie (oś x) - kierunek prostopadły do osi wału poziomy",
    1: "acc/y - przyśpieszenie (oś Y) - kierunek równoległy do osi wału poziomy",
    2: "acc/z - przyśpieszenie (oś Z) - kierunek prostopadły do osi wału pionowy",
}


GRAVITY = 9.8
CURRENT_SENSOR = 0


DIAGNOSIS_LOCK = threading.Lock()
CURRENT_SENSOR_LOCK = threading.Lock()
LOCK = threading.Lock()
GUILOCK = threading.Lock()
REFERENCE_LOCK = threading.Lock()
SYNCHRONIZATION_QUEUE_LOCK = threading.Lock()


class Configuration:
    def __init__(self):
        with TdmsFile.open(pathToData) as tdms_file:
            for group in tdms_file.groups():
                all_group_channels = group.channels()
                sensorNumber = 0
                for channel in all_group_channels:
                    logging.info(
                        "Reading sensor: {}".format(
                            sensorDictionary[sensorNumber])
                    )

                    for chunk in channel.data_chunks():
                        channel_chunk_data = chunk[:]
                        self.fs = channel_chunk_data.size
                        break
                    break
                break

        self.STFTBufferlength = 5 * self.fs


class DiagnosisFlags(Enum):
    DIAGNOSIS_INIT = 0
    REFERENCE_DIAGNOSIS_INIT = 1
    REFERENCE_FREQUENCIES_ACQUIRED = 2
    REFERENCE_AMPLITUDE_ACQUIRED = 3
    PEAKS_FREQUENCIES_ARE_OK = 4
    PEAKS_FREQUENCIES_ARE_INVALID = 5
    AMPLITUDES_ARE_OK = 6
    AMPLITUDES_ARE_INVALID = 7


PERFECT_DIAGNOSIS = [
    DiagnosisFlags.PEAKS_FREQUENCIES_ARE_OK,
    DiagnosisFlags.AMPLITUDES_ARE_OK,
]

diagnosisDictionary = {
    DiagnosisFlags.REFERENCE_DIAGNOSIS_INIT: "Reference Diagnosis Init",
    DiagnosisFlags.REFERENCE_FREQUENCIES_ACQUIRED: "Reference frequencies acquired",
    DiagnosisFlags.REFERENCE_AMPLITUDE_ACQUIRED: "Reference amplitude acquired",
    DiagnosisFlags.DIAGNOSIS_INIT: "Diagnosis INIT!",
    DiagnosisFlags.PEAKS_FREQUENCIES_ARE_OK: "Peaks frequencies are OK!",
    DiagnosisFlags.PEAKS_FREQUENCIES_ARE_INVALID: "Peaks Frequencies are INVALID!",
    DiagnosisFlags.AMPLITUDES_ARE_OK: "Amplitudes are OK!",
    DiagnosisFlags.AMPLITUDES_ARE_INVALID: "Amplitudes are INVALID!",
}


class ReferenceData:
    def __init__(self):
        self.peaks = {}
        self.amplitudes = {}
        self.sensorListNumbers = list()
        self.currentBufferCount = 0

    def __repr__(self):
        outputBuffer = "PEAKS: "
        for key in self.peaks.keys():
            outputBuffer += f"Sensor Number: {key}, peaks: {self.peaks[key]}, "
            outputBuffer += f"Max amplitudes {np.max(self.amplitudes[key])}, "
            outputbuffer += f"Buffers size: {self.amplitudes[key].size()}\n"
        return outputBuffer

    def registerBuffer(self, buffer):
        spectrum = Spectrum(buffer)
        self._CheckAndPrepareClassMembersForNewSensor(buffer.size)
        self._acquireSpectrum(spectrum)

    def _acquireSpectrum(self, spectrum):
        with CURRENT_SENSOR_LOCK:
            peaks = sg.find_peaks(spectrum)

            # FIXME: make this average value, not updated value
            # list TODO:
            # [ ] find 3 max values inside spectrum array
            # [ ] sort them? - probably not needed since
            #        every time peaks are in increasing order
            # [ ] find out if acquired values corespond to
            #        already acquired values
            self.peaks[CURRENT_SENSOR] = peaks
            self.amplitudes[CURRENT_SENSOR] += spectrum

            self.currentBufferCount += 1

    def _CheckAndPrepareClassMembersForNewSensor(self, bufferSize):
        with CURRENT_SENSOR_LOCK:
            if CURRENT_SENSOR not in self.sensorListNumbers:

                if CURRENT_SENSOR > 0:
                    self._finishCurrentBuffer()
                self.currentBufferCount = 0
                self.sensorListNumbers.append(CURRENT_SENSOR_LOCK)
                self.peaks[CURRENT_SENSOR] = list()
                self.amplitudes[CURRENT_SENSOR] = np.zeros(bufferSize // 2 + 1)

    def _finishCurrentBuffer(self):
        self.amplitudes[CURRENT_SENSOR-1] /= self.currentBufferCount


class DiagnosisTracker:
    def __init__(self, referenceData):
        self.referenceData = referenceData

    def runDiagnosis(self, buffer):
        outputFlags = list()
        # outputFlags.append(self.checkPeaksFrequencies(buffer))
        outputFlags.append(self.checkPeaksAmplitudes(buffer))
        return outputFlags

    def getInitDiagnosisFlag(self):
        return DiagnosisFlags.DIAGNOSIS_INIT

    def checkPeaksFrequencies(self, buffer):
        spec = Spectrum(buffer)
        peakDetectionThreshold = np.max(spec) * FAULT_DETECTION_THRESHOLD
        bufferPeaks = sg.find_peaks(spec, threshold=peakDetectionThreshold)[0]
        referencePeaks = None
        with CURRENT_SENSOR_LOCK:
            referencePeaks = self.referenceData.peaks[CURRENT_SENSOR]
            logging.info(f"REFERENCE PEAKS: {referencePeaks}")
        if len(bufferPeaks) == len(referencePeaks) and sorted(
                bufferPeaks) == sorted(referencePeaks):
            return DiagnosisFlags.PEAKS_FREQUENCIES_ARE_OK
        else:
            return DiagnosisFlags.PEAKS_FREQUENCIES_ARE_INVALID

    def checkPeaksAmplitudes(self, buffer):
        spec = Spectrum(buffer)
        tempSum = np.sum(
            np.abs(
                self.referenceData.amplitudes[CURRENT_SENSOR] - spec)) / np.sum(
                    self.referenceData.amplitudes[CURRENT_SENSOR])
        logging.info(f"Peak in Spectrum Error: {tempSum}")
        if tempSum < FAULT_DETECTION_THRESHOLD:
            return DiagnosisFlags.AMPLITUDES_ARE_OK
        else:
            return DiagnosisFlags.AMPLITUDES_ARE_INVALID


def logDiagnosis(flagList):
    loggingStream = ""
    isEverythingAlright = True

    for flagPosition in range(len(flagList)):
        if flagList[flagPosition] not in PERFECT_DIAGNOSIS:
            isEverythingAlright = False

        loggingStream += diagnosisDictionary[flagList[flagPosition]]

        if flagPosition + 1 < len(flagList):
            loggingStream += "\n"

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")

    if isEverythingAlright:
        logging.info(f" TIME: {current_time} OK")
    else:
        logging.info(
            f" TIME: {current_time} BAD DIAGNOSIS: \n" + loggingStream)


def Spectrum(signal):
    return np.square(np.abs(np.fft.rfft(signal)) / signal.size)


def Freq(signal):
    return np.arange(len(signal) / 2 + 1)


def Time(signal):
    return np.arange(len(signal))


def findAllMinimas(time):
    return np.argwhere(time < DETECTION_THRESHOLD)


def resampleBufferPerTime(buffer, time):
    minimas = findAllMinimas(time)
    newBuffer = np.zeros(len(minimas) * NUMBER_OF_SAMPLES_PER_ROTATION)
    for index, minim in enumerate(minimas):
        newMinimum = minim[0]
        previousMinima = minimas[index - 1][0] if index > 0 else 0
        newBuffer[index * NUMBER_OF_SAMPLES_PER_ROTATION:(
            index + 1) * NUMBER_OF_SAMPLES_PER_ROTATION] = sg.resample(buffer[previousMinima:newMinimum], NUMBER_OF_SAMPLES_PER_ROTATION)

    # newBuffer[len(minimas)-1*NUMBER_OF_SAMPLES_PER_ROTATION:] = sg.resample(
    #     buffer[minimas[-1][0]:], NUMBER_OF_SAMPLES_PER_ROTATION)

    newBuffer = np.pad(newBuffer, pad_width=(
        0, RESAMPLED_BUFFER_SIZE - newBuffer.size), mode='constant', constant_values=0)
    return newBuffer


def animationGui(que, synchronizationQueue):
    configuration = Configuration()
    maxSTFTBufferSize = 5 * configuration.fs
    stftBuffer = np.zeros(maxSTFTBufferSize)

    plt.ion()
    fig = plt.figure(figsize=(10, 10), tight_layout=True)

    ax1 = fig.add_subplot(221)
    ax1.set_title("Time Domain")
    ax1.set_xlabel("Buffer Sample")
    ax1.set_ylabel("Amplitude $\dfrac{m}{s^2}$")
    ax1.grid(True, which="both")

    ax2 = fig.add_subplot(222)
    ax2.grid(True, which="both")
    ax2.set_title("Frequency Domain")
    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_ylabel("Amplitude $\dfrac{m}{s^2}$")

    ax3 = fig.add_subplot(223)
    ax3.set_ylabel("frequency [Hz]")
    ax3.set_xlabel("Time [s]")

    ax4 = fig.add_subplot(224)
    ax4.set_ylabel("Amplitude $\dfrac{m}{s^2}$")
    ax4.set_xlabel("Order Number [-]")
    ax4.set_title("Order Analisis")

    graph1, graph2, graph3, graph4 = None, None, None, None
    first = True
    data = np.array([])
    synchronization_data = None

    while not keyboard.is_pressed("q"):
        if (not synchronizationQueue.empty()):
            with SYNCHRONIZATION_QUEUE_LOCK:
                synchronization_data = synchronizationQueue.get()

        if first and not que.empty():

            with GUILOCK:
                data = que.get()

            first = False
            spectrum = Spectrum(data)
            frequency = Freq(data)
            (graph1,) = ax1.plot(Time(data), data * GRAVITY)
            (graph2,) = ax2.semilogx(frequency, spectrum * GRAVITY)

            stftBuffer = np.roll(stftBuffer, -data.size)
            stftBuffer[stftBuffer.size -
                       data.size:] = np.kaiser(data.size, 1.27) * data
            f, t, Zxx = sg.stft(stftBuffer, configuration.fs, nperseg=256)
            graph3 = ax3.imshow(np.abs(Zxx), origin='lower', aspect='auto')

            frequencies = np.round(np.arange(7) / 7 * configuration.fs, 2)
            ax3.set_yticks([0, 20, 40, 60, 80, 100, 120])
            ax3.set_yticklabels(frequencies)

            ax3.set_xticks([0, 33.33, 66.66, 99.99, 133.33, 166.66, 199.99, ])
            ax3.set_xticklabels(np.flip(np.arange(7)))

            # TODO: create Labels for the resampled buffer
            resampledBuffer = resampleBufferPerTime(data, synchronization_data)
            (graph4,) = ax4.plot(Spectrum(resampledBuffer))

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

            stftBuffer = np.roll(stftBuffer, -data.size)
            stftBuffer[stftBuffer.size -
                       data.size:] = np.kaiser(data.size, 1.27) * data
            f, t, Zxx = sg.stft(stftBuffer, configuration.fs, nperseg=256)
            graph3.set_data(np.abs(Zxx))

            # ORDER ANALISIS
            resampledBuffer = resampleBufferPerTime(data, synchronization_data)
            graph4.set_ydata(Spectrum(resampledBuffer))
            ax4.relim()
            ax4.autoscale_view()

            with CURRENT_SENSOR_LOCK:
                fig.suptitle(sensorDictionary[CURRENT_SENSOR], fontsize=16)

            fig.canvas.draw()

        fig.canvas.flush_events()
        time.sleep(0.05)


def saveToDatabase(data, tdms_writer=None):
    currentTime = datetime.now().strftime("%H:%M:%S")
    logging.info(f"Saving data at time: {currentTime}")

    if SAVE_TO_JSON:
        outputdata = {"time": currentTime, "data": data.tolist()}

        with open(DATABASE_PATH + ".js", "a") as f:
            f.write(json.dumps(outputdata) + ",\n")

    if tdms_writer is not None:
        channel = ChannelObject("Undefined", "Channel1", data)
        tdms_writer.write_segment([channel])


def acquiringData(guiQueue, diagnosisQueue):
    t = threading.current_thread()
    name = str(t.getName())

    logging.info(f"ACQUISITION Thread: {name} started!")
    # Brings card profile configured with NI MAX
    system = ni.system.system.System.local()
    taskName = system.tasks.task_names[0]

    logging.info(f"Task Name: {taskName}")
    # prepare task for acquisition
    prepareTask = ni.system.storage.persisted_task.PersistedTask(taskName)
    task = prepareTask.load()
    # reads data
    while not keyboard.is_pressed("q"):
        # if q is pressed, data acquisition is stopped
        with LOCK:
            tempData = np.array(task.read(ni.constants.READ_ALL_AVAILABLE))
            guiQueue.put(tempData)

        time.sleep(0.3)
    task.stop()

    logging.info(f"ACQUISITION Thread: {name} ended!")


def savingThread(que, guiQueue):
    t = threading.current_thread()
    name = str(t.getName())
    logging.info(f"GUI Thread: {name} started!")

    with TdmsWriter(DATABASE_PATH + ".tdms") as writer:
        while not keyboard.is_pressed("q"):

            with LOCK:
                if not que.empty():
                    temp = que.get()
                    saveToDatabase(data=temp, tdms_writer=writer)
                    guiQueue.put(temp)

            time.sleep(0.2)

    logging.info(f"GUI Thread: {name} ended!")


def diagnosisThread(diagnosisQueue, referenceData):
    t = threading.currentThread()
    logging.info(f"Reference Data Acquisition {t.getName()} started!")

    tracker = DiagnosisTracker(referenceData)
    buffer = None
    currentSenor = CURRENT_SENSOR
    while not keyboard.is_pressed("q"):

        with CURRENT_SENSOR_LOCK:
            if currentSenor != CURRENT_SENSOR:
                currentSenor = CURRENT_SENSOR

        with DIAGNOSIS_LOCK:
            if not diagnosisQueue.empty():
                buffer = diagnosisQueue.get()

        if buffer is not None:
            diagnosisResultFlags = tracker.runDiagnosis(buffer)
            buffer = None
            logDiagnosis(diagnosisResultFlags)

        time.sleep(0.5)


def referenceDataAcquisition(dataQueue, reference):
    t = threading.currentThread()
    logging.info(f"reference Acquisition {t.getName()} Started!")
    while not (keyboard.is_pressed("q") or not REFERENCE):

        buffer = None
        with REFERENCE_LOCK:
            if not dataQueue.empty():
                buffer = dataQueue.get()

        if buffer is not None:
            reference.registerBuffer(buffer)

        time.sleep(0.9)
        if dataQueue.empty():
            break
    logging.info(f"Reference Acquisition {t.getName()} Ended!")


def readTDMS(pathToData, guiQueue, diagnosisQueue):

    t = threading.currentThread()
    logging.info(
        f"Reading TDMS file at: {pathToData} started in {t.getName()}")

    def startReading(pathToData, guiQueue, diagnosisQueue):
        with TdmsFile.open(pathToData) as tdms_file:
            for group in tdms_file.groups():
                all_group_channels = group.channels()
                sensorNumber = 0
                for channel in all_group_channels:
                    logging.info(
                        "Reading sensor: {}".format(
                            sensorDictionary[sensorNumber])
                    )

                    for chunk in channel.data_chunks():
                        channel_chunk_data = chunk[:]

                        if guiQueue is not None:
                            with GUILOCK:
                                guiQueue.put(channel_chunk_data)
                        if diagnosisQueue is not None:
                            with DIAGNOSIS_LOCK:
                                diagnosisQueue.put(channel_chunk_data)

                        time.sleep(0.9)
                        if keyboard.is_pressed("q"):
                            sys.exit()

                    with CURRENT_SENSOR_LOCK:
                        global CURRENT_SENSOR
                        if CURRENT_SENSOR != 2:
                            CURRENT_SENSOR += 1
                            sensorNumber += 1
                        else:
                            return

    startReading(pathToData=pathToData, guiQueue=guiQueue,
                 diagnosisQueue=diagnosisQueue)

    logging.info(f"Reading TDMS file at: {pathToData} ended in {t.getName()}")

# Reads data acquired via laser from last available channel in tdms file


def synchronizationThread(synchronizationQueue):
    t = threading.currentThread()
    logging.info(f" Synchronization {t.getName()} started!")
    for _ in range(len(sensorDictionary.keys())):
        with TdmsFile.open(pathToData) as tdms_file:
            for group in tdms_file.groups():
                all_group_channels = group.channels()
                synchronization_channel = all_group_channels[SYNCHRONIZATION_CHANNEL]

                for chunk in synchronization_channel.data_chunks():
                    current_chunk = chunk[:]
                    with SYNCHRONIZATION_QUEUE_LOCK:
                        synchronizationQueue.put(current_chunk)

                    if keyboard.is_pressed("q"):
                        sys.exit()

                    time.sleep(0.9)

    logging.info(f" Synchronization {t.getName()} ended!")


if __name__ == "__main__":

    # DATA PREPARATION PHASE
    logging.info("DATA PREPARATION PHASE STARTED!")
    reference = ReferenceData()
    referenceQueue = queue.Queue()
    programPreparationThreads = list()
    if REFERENCE:
        programPreparationThreads.append(
            threading.Thread(
                target=readTDMS, args=(
                    PATH_TO_REFERENCE_TDMS, None, referenceQueue)
            )
        )

        programPreparationThreads.append(
            threading.Thread(
                target=referenceDataAcquisition, args=(
                    referenceQueue, reference)
            )
        )

    for thread in programPreparationThreads:
        thread.start()

    for thread in programPreparationThreads:
        thread.join()
    logging.info("DATA PREPARATION PHASE ENDED!")

    # MEASURMENT PHASE
    logging.info("MEASURMENT PHASE STARTED!")

    CURRENT_SENSOR = 0
    dataQueue = queue.Queue()
    guiQueue = queue.Queue()
    diagnosisQueue = queue.Queue()
    synchronizationQueue = queue.Queue()

    threads = list()

    if ACQUISITION:

        if SAVE_TO_JSON:
            # Clearing and creating new database inside .js
            # file at DATABASE_PATH

            with open(DATABASE_PATH + ".js", "w+") as f:
                f.write("const data = [")

        threads.append(
            threading.Thread(target=acquiringData,
                             args=(dataQueue, diagnosisQueue))
        )
        threads.append(
            threading.Thread(target=savingThread, args=(dataQueue, guiQueue))
        )

    if READING:
        pathToData = PATH_TO_TDMS
        threads.append(
            threading.Thread(
                target=readTDMS, args=(pathToData, guiQueue, diagnosisQueue)
            )
        )

        threads.append(
            threading.Thread(target=synchronizationThread,
                             args=(synchronizationQueue, ))
        )

    threads.append(threading.Thread(
        target=diagnosisThread, args=(diagnosisQueue, reference)))

    logging.info(f"Threads in buffer: {threads}")
    for thread in threads:
        thread.start()

    animationGui(guiQueue, synchronizationQueue)

    for thread in threads:
        thread.join()

    # Closing database
    if SAVE_TO_JSON and ACQUISITION:
        with open(DATABASE_PATH + ".js", "a") as f:
            f.write("]")

    logging.info("MEASURMENT PHASE ENDED!")

import main
from queue import Queue
import threading
import logging
import keyboard
import matplotlib.pyplot as plt
import time
import scipy.signal as sg
import numpy as np
import sys
import bearing
import json

logging.getLogger().setLevel(logging.INFO)

frequencyA = 2500
frequencyB = 1000

object = bearing.Bearing6306Frequencies()
logging.info(
    f"frequency: 2500Hz : {json.dumps(object.getCharacteristicFrequencies(frequencyA), indent=4)}")
logging.info(
    f"frequency: 1000Hz : {json.dumps(object.getCharacteristicFrequencies(frequencyB), indent=4)}")

PATH_TO_REFERENCE_TDMS = "./source/lab8Data/01sprawne.tdms"
PATH_TO_TDMS = "./source/lab8Data/06uszkPW.tdms"

ReferenceDataQueue = Queue()
DamagedDataQueue = Queue()

ReferenceDataQueueLock = threading.Lock()
DamagedDataQueueLock = threading.Lock()


def gui(ReferenceDataQueue, DamagedDataQueue):
    logging.info("gui thread started!")

    plt.ion()
    fig = plt.figure(figsize=(16, 12), tight_layout=True)

    axList = list()
    ax1 = fig.add_subplot(421)
    ax1.set_title("REFERENCE DATA")
    ax1.set_xlabel("Buffer Sample")
    ax1.set_ylabel("Amplitude $\dfrac{m}{s^2}$")
    axList.append(ax1)

    ax2 = fig.add_subplot(422)
    ax2.grid(True, which="both")
    ax2.set_title("Frequency Domain")
    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_ylabel("Amplitude $\dfrac{m}{s^2}$")
    axList.append(ax2)

    ax3 = fig.add_subplot(423)
    ax3.set_title("DAMAGED DATA")
    ax3.set_xlabel("Buffer Sample")
    ax3.set_ylabel("Amplitude $\dfrac{m}{s^2}$")
    axList.append(ax3)

    ax4 = fig.add_subplot(424)
    ax4.grid(True, which="both")
    ax4.set_title("Frequency Domain")
    ax4.set_xlabel("Frequency [Hz]")
    ax4.set_ylabel("Amplitude $\dfrac{m}{s^2}$")
    axList.append(ax4)

    ax5 = fig.add_subplot(425)
    ax5.grid(True, which="both")
    ax5.set_title("Reference Envelope")
    ax5.set_xlabel("BUffer sample")
    ax5.set_ylabel("Amplitude g$\dfrac{m}{s^2}$")
    axList.append(ax5)

    ax6 = fig.add_subplot(426)
    ax6.grid(True, which="both")
    ax6.set_title("reference Envelope spectrum")
    ax6.set_xlabel("Frequency [Hz]")
    ax6.set_ylabel("Amplitude $\dfrac{m}{s^2}$")
    axList.append(ax6)

    ax7 = fig.add_subplot(427)
    ax7.grid(True, which="both")
    ax7.set_title("Damaged Buffer Sample")
    ax7.set_xlabel("Buffer Sample")
    ax7.set_ylabel("Amplitude $\dfrac{m}{s^2}$")
    axList.append(ax7)

    ax8 = fig.add_subplot(428)
    ax8.grid(True, which="both")
    ax8.set_title("Damaged Envelope spectrum")
    ax8.set_xlabel("Frequency [Hz]")
    ax8.set_ylabel("Amplitude $\dfrac{m}{s^2}$")
    axList.append(ax8)

    with ReferenceDataQueueLock:
        referenceData = ReferenceDataQueue.get()
    referenceSpectrum = main.Spectrum(referenceData)
    referenceEnvelope = np.real(sg.hilbert(referenceData))
    referenceEnvelopeSpectrum = main.Spectrum(referenceEnvelope)

    with DamagedDataQueueLock:
        damagedData = DamagedDataQueue.get()
    damagedSpectrum = main.Spectrum(damagedData)
    damagedEnvelope = np.real(sg.hilbert(damagedData))
    damagedEnvelopeSpectrum = main.Spectrum(damagedEnvelope)

    (referenceTimeGraph, ) = ax1.plot(
        main.Time(referenceData), referenceData * main.GRAVITY)
    (referenceSpectrumGraph, ) = ax2.plot(
        main.Freq(referenceData), referenceSpectrum)

    (damagedTimeGraph, ) = ax3.plot(
        main.Time(damagedData), damagedData * main.GRAVITY)
    (damagedSpectrumGraph, ) = ax4.plot(
        main.Freq(damagedData), damagedSpectrum)

    (referenceEnvelopeGraph, ) = ax5.plot(
        main.Time(referenceEnvelope), referenceEnvelope * main.GRAVITY)
    (referenceEnvelopeSpectrumGraph, ) = ax6.plot(
        main.Freq(referenceEnvelope), referenceEnvelopeSpectrum)

    (damagedEnvelopeGraph, ) = ax7.plot(
        main.Time(damagedEnvelope), damagedEnvelope * main.GRAVITY)
    (damagedEnvelopeSpectrumGraph, ) = ax8.plot(
        main.Freq(damagedEnvelope), damagedEnvelopeSpectrum)

    plt.show()

    while not keyboard.is_pressed("q"):
        if not ReferenceDataQueue.empty() and not DamagedDataQueue.empty():

            with ReferenceDataQueueLock:
                referenceData = ReferenceDataQueue.get()
            referenceSpectrum = main.Spectrum(referenceData)

            referenceEnvelope = np.real(sg.hilbert(referenceData))
            referenceEnvelopeSpectrum = main.Spectrum(referenceEnvelope)

            with DamagedDataQueueLock:
                damagedData = DamagedDataQueue.get()
            damagedSpectrum = main.Spectrum(damagedData)

            damagedEnvelope = np.real(sg.hilbert(damagedData))
            damagedEnvelopeSpectrum = main.Spectrum(damagedEnvelope)

            referenceTimeGraph.set_ydata(referenceData)
            referenceSpectrumGraph.set_ydata(referenceSpectrum)
            damagedTimeGraph.set_ydata(damagedData)
            damagedSpectrumGraph.set_ydata(damagedSpectrum)

            referenceEnvelopeGraph.set_ydata(np.abs(referenceEnvelope))
            referenceEnvelopeSpectrumGraph.set_ydata(referenceEnvelopeSpectrum)
            damagedEnvelopeGraph.set_ydata(np.abs(damagedEnvelope))
            damagedEnvelopeSpectrumGraph.set_ydata(
                damagedEnvelopeSpectrum)

            for ax in axList:
                ax.relim()
                ax.autoscale_view()

            fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.05)


referenceThread = threading.Thread(
    target=main.readTDMS, args=(PATH_TO_REFERENCE_TDMS, ReferenceDataQueue, None))

damagedThread = threading.Thread(
    target=main.readTDMS, args=(PATH_TO_TDMS, DamagedDataQueue, None))

threads = [referenceThread, damagedThread]

for thread in threads:
    thread.start()

gui(ReferenceDataQueue=ReferenceDataQueue, DamagedDataQueue=DamagedDataQueue)

for thread in threads:
    thread.join()

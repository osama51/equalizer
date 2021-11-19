import sys
import wave
import pyaudio
import threading
import numpy as np
from os import path
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets
import matplotlib.pyplot as plt
import time
from PyQt5.uic import loadUiType
from PyQt5 import QtCore, QtGui
from scipy.fft import fft, fftfreq
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.uic import loadUiType
from PyQt5.QtCore import QTimer
from os import path
import pyqtgraph

import soundfile as sf
import sounddevice as sd
import logging
import threading
import _thread


FORM_CLASS,_ = loadUiType(path.join(path.dirname(__file__), "gui.ui"))
class MainApp(QMainWindow, FORM_CLASS):
    def __init__(self , parent=None):
        pyqtgraph.setConfigOption('background', (0,0,0))
        super(MainApp,self).__init__(parent)
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.iterator = 0
        self.timor= 0.0 
        self.p=pyaudio.PyAudio()
        # self.chunk=4096 # gets replaced automatically
        # self.updatesPerSecond=10
        # self.chunksRead=0
        # self.rate=None
        self.timer = QTimer(self)
        self.grMain.plotItem.showGrid(True, True, 0.7)
        self.grSpec.plotItem.showGrid(True, True, 0.7)
        # self.handle_buttons()
        
        self.actionOpen.triggered.connect(self.browse)
        self.actionPlay.triggered.connect(self.toggle)
        self.actionStop.triggered.connect(self.stop)
        
    
    def browse(self):
        self.file_path_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', " ", "(*.txt *.csv *.xls *.wav)")
        self.file_name, self.file_extension = os.path.splitext(self.file_path_name)
        
        self.read_data()
        print("I read the data!")
        # self.sample()

    def read_data(self):
        self.iterator = 0
        spf = wave.open(self.file_path_name, "r")
        print(spf)
        # print(type(spf))
        # Extract Raw Audio from Wav File
        self.signal = spf.readframes(-1)
        print(type(self.signal))
        self.signal = np.frombuffer(self.signal, "int32")
        
        print(type(self.signal))
        self.fs = spf.getframerate()
        self.one_frame = int(self.fs/17)
        self.Time = np.linspace(0, len(self.signal) / self.fs, num=len(self.signal))
        
    def play_sound(self):
        sd.play(self.signal[self.iterator:], self.fs)

    def update(self):
        # print(type(self.signal))
        # print("Fs:", fs)
        
        # self.one_frame = int(self.sampling / 12)
        # purple = (102, 0, 204)
        # pen=pyqtgraph.mkPen(color='c')
        # self.grMain.plotItem.setXRange(self.Time[0], self.Time[self.one_frame]/8)
        # self.grMain.plotItem.setYRange(min(self.signal)*1.5,
        #                                 max(self.signal)*1.5)
               
        # print(self.iterator, self.timor)
        # self.timor += 0.059
        # print(type(self.timor))
        self.grMain.plotItem.setXRange(self.Time[self.iterator], self.Time[self.iterator+self.one_frame])
        # self.grMain.plot(self.Time[self.iterator:self.iterator+self.one_frame],
        #                  self.signal[self.iterator:self.iterator+self.one_frame],
        #                  pen=pen) #clear=True
        
        # pen=pyqtgraph.mkPen(color='r')
        # self.grSpec.plot(self.freq,self.spectrum,pen=pen,clear=True)

        self.iterator += self.one_frame
        if self.iterator >= len(self.signal)-self.one_frame: 
            # self.iterator = len(self.signal)-1
            self.iterator = 0
            self.actionPlay.setChecked(False)
            # self.timer.stop()
        # QtCore.QTimer.singleShot(250, self.update)
        if self.actionPlay.isChecked():
            self.timer.singleShot(1, self.update)
        else:
            self.timer.stop()
            sd.stop()
            
    # def spec(self):
    #     spectrum = fft(data)[:len(data)//2]
    #     freq = fftfreq(len(spectrum),1/samplerate)[:len(data)//2
                                                   
    #     threshold = 0.116 * max(abs(spectrum))
    #     mask = abs(spectrum) > threshold
    #     peaks_freqs=abs(freq[mask[:len(data)//2]])
    #     phases_of_peaks=np.angle(spectrum)[mask[:len(data)//2]]
    #     mags=spectrum[mask[:len(data)//2]]
        
    #     basis_fun=mags[i]*np.sin(2*np.pi*peaks_freqs[i]*time+phases_of_peaks[i]),new_track

    def play(self):
        self.fft()
        # self.play_sound()
        _thread.start_new_thread(self.play_sound, ())
        # self.timer.timeout.connect(self.update)
        # self.timer.start()
        # self.current = time.clock_gettime()
        # while self.actionPlay.isChecked():
        pen=pyqtgraph.mkPen(color='c')
        self.grMain.plotItem.setYRange(min(self.signal)*1.5,
                                        max(self.signal)*1.5)
        self.grMain.plot(self.Time, self.signal, pen=pen)
        pen=pyqtgraph.mkPen(color='r')
        self.grSpec.plot(self.freq,self.spectrum,pen=pen,clear=True)
        # x = threading.Thread(target=self.update)
        # x.start()
        # _thread.start_new_thread(self.update)
        self.update()
            
    def pause(self):
        self.timer.stop()
        # self.iterator = 0
        
    def toggle(self):
        if self.actionPlay.isChecked():
            self.play()
        else:
            self.pause()
            
    def stop(self):
        self.timer.stop()
        self.grMain.plotItem.clearPlots()
        self.actionPlay.setChecked(False)
        self.iterator = 0
        
    def fft(self):
        N = int(len(self.signal))
        self.spectrum = fft(self.signal)
        self.spectrum = 2.0/N * np.abs(self.spectrum[0:N//2])
        self.freq = fftfreq(len(self.spectrum), 1/self.fs)
            
    def exporting_to_csv(self):
        data = {'freq': self.freq,'spectrum': self.spectrum } #list(np.arange(0,len(self.freq),1))
        df = pd.DataFrame(data, columns= ['freq', 'spectrum'])
        name = QFileDialog.getSaveFileName(self, 'Save File')
        df.to_csv (str(name[0]), index = False, header=True)
            
        
def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    app.exec_()
if __name__ == '__main__':
    main()

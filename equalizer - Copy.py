import sys
import wave
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
from scipy import signal
import pyqtgraph as pg

import sounddevice as sd
import logging
import threading
import _thread

import shutil

from PyQt5 import QtCore, QtGui, QtWidgets
from pyqtgraph import PlotWidget
import pyqtgraph as pg

# from pop import popWindow
from scipy.io import wavfile
import numpy as np
import sys
import os
from scipy.fftpack import fft
#from funcations import funcation as f
import wave
import struct
from scipy import signal
from playsound import playsound
from collections import OrderedDict
from scipy.io import wavfile



FORM_CLASS,_ = loadUiType(path.join(path.dirname(__file__), "gui.ui"))
class MainApp(QMainWindow, FORM_CLASS):
    def __init__(self , parent=None):
        pyqtgraph.setConfigOption('background', (0,0,0))
        super(MainApp,self).__init__(parent)
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.iterator = 0
        self.timor= 0.0 

        # self.chunk=4096 # gets replaced automatically
        # self.updatesPerSecond=10
        # self.chunksRead=0
        # self.rate=None
        self.timer = QTimer(self)
        self.grMain.plotItem.showGrid(True, True, 0.7)
        # self.grSpec.plotItem.showGrid(True, True, 0.7)
        # self.handle_buttons()
        self.verticalSlider_gain.valueChanged.connect(self.gain)
        self.verticalSlider_60.valueChanged.connect(
            lambda:self.equalizer(0,60,self.verticalSlider_60.value()))
        self.verticalSlider_170.valueChanged.connect(
            lambda:self.equalizer(60,250,self.verticalSlider_170.value()))
        self.verticalSlider_310.valueChanged.connect(
            lambda:self.equalizer(250,400,self.verticalSlider_310.value()))
        self.verticalSlider_600.valueChanged.connect(
            lambda:self.equalizer(400,700,self.verticalSlider_600.value()))
        self.verticalSlider_1k.valueChanged.connect(
            lambda:self.equalizer(700,2000,self.verticalSlider_1k.value()))
        self.verticalSlider_3k.valueChanged.connect(
            lambda:self.equalizer(2000,4000,self.verticalSlider_3k.value()))
        self.verticalSlider_6k.valueChanged.connect(
            lambda:self.equalizer(4000,6000,self.verticalSlider_6k.value()))
        self.verticalSlider_10k.valueChanged.connect(
            lambda:self.equalizer(6000,10000,self.verticalSlider_10k.value()))
        self.verticalSlider_16k.valueChanged.connect(
            lambda:self.equalizer(10000,16000,self.verticalSlider_16k.value()))
        
        self.actionOpen.triggered.connect(self.browse)
        self.actionPlay.triggered.connect(self.toggle)
        self.actionStop.triggered.connect(self.stop)
        
    def handle_slider(self):
        self.label_60.setText(str(self.verticalSlider_60.value()))
        self.label_170.setText(str(self.verticalSlider_170.value()))
        self.label_310.setText(str(self.verticalSlider_310.value()))
        self.label_600.setText(str(self.verticalSlider_600.value()))
        self.label_1k.setText(str(self.verticalSlider_1k.value()))
        self.label_3k.setText(str(self.verticalSlider_3k.value()))
        self.label_6k.setText(str(self.verticalSlider_6k.value()))
        self.label_10k.setText(str(self.verticalSlider_10k.value()))
        self.label_16k.setText(str(self.verticalSlider_16k.value()))

    def browse(self):
        self.file_path_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', " ", "(*.txt *.csv *.xls *.wav)")
        self.file_name, self.file_extension = os.path.splitext(self.file_path_name)
        self.read_data()
        print("I read the data!")

    def read_data(self):
        self.iterator = 0
        spf = wave.open(self.file_path_name, "r")
        print(spf)
        # print(type(spf))
        # Extract Raw Audio from Wav File
        self.original_signal = spf.readframes(-1)
        print(type(self.original_signal))
        self.original_signal = np.frombuffer(self.original_signal, "int32")
        self.signal=self.original_signal
        print(type(self.original_signal))
        self.samplerate = spf.getframerate()
        self.one_frame = int(self.samplerate/17)
        self.Time = np.linspace(0, len(self.original_signal) / self.samplerate, num=len(self.original_signal))
        self.original_spectrum = fft(self.original_signal)[:len(self.original_signal)//2]
        self.spectrum=self.original_spectrum
        self.freq = fftfreq(len(self.original_spectrum),1/self.samplerate)[:len(self.original_signal)//2]

        
    #################################################################
    # add a flag to check if spectrum is modified a specific range ##           <--DON'T
    #################################################################
    def equalizer(self,min_freq,max_freq,slider_value):
        self.handle_slider()
        freq_list=list(self.freq)
        first_index = 0
        second_index = 0
        db=slider_value
        for frequency in freq_list :
            if first_index == 0 and frequency>min_freq:
                first_index = freq_list.index(frequency)

            if second_index == 0 and frequency>max_freq:
                second_index = freq_list.index(frequency)
        #print('ff',first_index,freq_list[first_index],'sec',second_index,freq_list[second_index])
        for index, item in enumerate(self.original_spectrum):
            if index > first_index and index < second_index :
                self.spectrum[index] = (item * 10 **(db/20)).real
                #print(len(self.spectrum))
                
        self.filtered_data=np.fft.ifft(self.spectrum)
        self.signal= self.filtered_data.real/100000
        # self.play_sound()
        # self.fttttt()
        self.play()
        
    # def fttttt(self):
    #     pen=pyqtgraph.mkPen(color='r')
    #     self.grSpec.plot(self.freq,abs(self.spectrum),pen=pen,clear=True)
        
    def gain(self):
        self.label_master_gain.setText(str(self.verticalSlider_gain.value()))
        gain_ratio = float(self.verticalSlider_gain.value()/100)
        #print(gain_ratio)
        print('before',self.signal[15000])
        self.signal = self.original_signal * gain_ratio
        print('after',self.signal[15000])
        # self.play_sound()
        # self.fttttt()
        
    def play_sound(self):
        # self.gain()
        sd.play(self.signal[self.iterator:], self.samplerate)

    def update(self):
        self.grMain.plotItem.setXRange(self.Time[self.iterator], self.Time[self.iterator+self.one_frame])
        self.iterator += self.one_frame
        if self.iterator >= len(self.signal)-self.one_frame: 
            self.iterator = 0
            self.actionPlay.setChecked(False)
        if self.actionPlay.isChecked():
            self.timer.singleShot(1, self.update)
        else:
            self.timer.stop()
            sd.stop()
            
    def play(self):
        # self.draw_spectrogram()
        self.fft()
        # self.fttttt()
        self.draw_spectrogram()
        # self.play_sound()
        self.spec_plot.setLimits(xMin=0, xMax=self.t[-1], yMin=0, yMax=self.f[-1])
        _thread.start_new_thread(self.play_sound, ())

        pen=pyqtgraph.mkPen(color='c')
        self.grMain.plotItem.setYRange(min(self.signal)*1.5,
                                        max(self.signal)*1.5)
        self.grMain.plot(self.Time, self.signal, pen=pen)
        self.update()
        # t = len(self.signal)
        # self.grMain.plotItem.setLimits(xMin=0, xMax=len(self.signal)[0], yMin=0, yMax=self.signal[0])
            
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
        self.freq = fftfreq(len(self.spectrum), 1/self.samplerate)
            
    # def exporting_to_csv(self):
    #     data = {'freq': self.freq,'spectrum': self.spectrum } #list(np.arange(0,len(self.freq),1))
    #     df = pd.DataFrame(data, columns= ['freq', 'spectrum'])
    #     name = QFileDialog.getSaveFileName(self, 'Save File')
    #     df.to_csv (str(name[0]), index = False, header=True)
        
    def draw_spectrogram(self):
        self.grSpec.clear()
        self.f, self.t, Sxx = signal.spectrogram(self.signal, self.samplerate)
        # Interpret image data as row-major instead of col-major
        pg.setConfigOptions(imageAxisOrder='row-major')
        pg.mkQApp()
        self.spec_plot = self.grSpec.addPlot()
        img = pg.ImageItem()
        self.spec_plot.addItem(img)
        hist = pg.HistogramLUTItem() # histogram to control the gradient of the image
        hist.setImageItem(img)
        self.grSpec.addItem(hist)
        # self.grSpec.show()
        hist.setLevels(np.min(Sxx), np.max(Sxx))
        img.setImage(Sxx) # Sxx: amplitude for each pixel
        img.scale(self.t[-1]/np.size(Sxx, axis=1),
                  self.f[-1]/np.size(Sxx, axis=0))
        
        self.spec_plot.setLabel('bottom', "Time", units='s')
        self.spec_plot.setLabel('left', "Frequency", units='Hz')
        hist.gradient.restoreState({'ticks': [(0.0, (0, 0, 0, 255)), (0.0, (32, 0, 129, 255)),
                                            (0.8, (255, 255, 0, 255)), (0.5, (115, 15, 255, 255)),
                                            (1.0, (255, 255, 255, 255))], 'mode': 'rgb'})
        

        
def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    app.exec_()
if __name__ == '__main__':
    main()

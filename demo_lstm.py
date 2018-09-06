# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 19:39:39 2018

@author: BRIGHT
"""

from lstm_model_final import train_lstm, update_lstm, forcast_with_lstm
train_lstm('C:/Users/BRIGHT/Desktop/ZENOTEC_2/Input/TrainingData_Modified.csv')

update_lstm('C:/Users/BRIGHT/Desktop/ZENOTEC_2/Input/TrainingData_Update.csv','C:/Users/BRIGHT/Desktop/ZENOTEC_2/Model' )

forcast_with_lstm('C:/Users/BRIGHT/Desktop/ZENOTEC_2/Input/TestDataNoOutput.csv',
                  'C:/Users/BRIGHT/Desktop/ZENOTEC_2/Model/20180906020843.h5')
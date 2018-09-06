# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 14:25:00 2018

@author: Bright Uchenna Oparaji, Institute for Risk and Uncertainty.
"""
def train_lstm(SpathToInputCsvFile):
    import numpy as np
    import pandas as pd

    dataset_train = pd.read_csv(SpathToInputCsvFile)
    training_set = dataset_train.iloc[:, 0:6].values
    training_set_new = training_set

    for i in range(len(training_set_new)):
        if training_set[i, 5] < 5:
            training_set_new[i, 5] = 1 # first class
        elif training_set[i, 5] < 30:
            training_set_new[i, 5] = 2 # second class
        elif training_set[i, 5] < 60:
            training_set_new[i, 5] = 3 # third class
        elif training_set[i, 5] < 120:
            training_set_new[i, 5] = 4 # fourth class
        elif training_set[i, 5] < 240:
            training_set_new[i, 5] = 5 # fifth class
        elif training_set[i, 5] < 600:
            training_set_new[i, 5] = 6 # sixth class
        elif training_set[i, 5] > 600:
            training_set_new[i, 5] = 7 # seventh class

    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(training_set_new[:,0:5])

    from sklearn.preprocessing import LabelEncoder
    training_set_new_class = training_set_new[:,5]
    training_set_new_class_reshape = training_set_new_class.reshape(training_set_new_class.size, 1)
    training_set_scaled = np.append(training_set_scaled, training_set_new_class_reshape, axis=1)
    labelencoder = LabelEncoder()
    training_set_scaled[:, 5] = labelencoder.fit_transform(training_set_scaled[:, 5])

    training_size = int(np.ceil(training_set_scaled[:,0].size*0.8))
    lag = 2 # number of realizations in the past the model looks back in order to predict the future
    training_set_final = training_set_scaled[0:training_size,:]
    test_set_final = training_set_scaled[training_size+1-lag:-1,:]

    X_train = []
    y_train = []
    for i in range(lag, training_set_final[:,0].size):
        X_train.append(training_set_final[i-lag:i, 0:5])
        y_train.append(training_set_final[i, 5])
    X_train, y_train = np.array(X_train), np.array(y_train)
    # Reshaping training set
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 5))

    X_test = []
    y_test = []
    for j in range(lag, test_set_final[:,0].size):
        X_test.append(test_set_final[j-lag:j, 0:5])
        y_test.append(test_set_final[j, 5])
    X_test, y_test = np.array(X_test), np.array(y_test)
    # Reshaping test set
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 5))

    # Importing the Keras libraries and packages
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers import Dropout

    classifier = Sequential()
    # Adding the first LSTM layer and some Dropout regularisation
    classifier.add(LSTM(units = 3, input_shape = (X_train.shape[1], 5)))
    # Adding some dropout to help reduce overfitting the model
    classifier.add(Dropout(0.2))
    # Adding the output layer with a softmax activation function for multi classification
    classifier.add(Dense(7, activation='softmax'))
    # Compiling the RNN
    classifier.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # Fitting model to data
    classifier.fit(X_train, y_train, epochs = 2, batch_size = 32)
    # Predicting training data
    predicted_train = classifier.predict(X_train)
    # Predicting test data
    predicted_test = classifier.predict(X_test)

    def compute_class_index(pred_softmax):
        output_list_index = []
        for i in range(pred_softmax.shape[0]):
            max_prediction_index = np.argmax(pred_softmax[i, :])
            output_list_index.append(max_prediction_index)
        return output_list_index

    # Predict training data class
    index_train = compute_class_index(predicted_train)

    # Predict test data class 
    index_test = compute_class_index(predicted_test)
    # Function that computes the reliability of the model
    def compute_model_reliability(class_index, target):
        out_list_reliability = []
        for z in range(predicted_test.shape[0]):
            if class_index[z] == int(target[z]):
                indicator = 1
            else:
                indicator = 0
            out_list_reliability.append(indicator)
            model_reliability = np.sum(out_list_reliability)/len(out_list_reliability)
        return model_reliability

    # Compute reliability of model on training data
    model_reliability_train = compute_model_reliability(index_train, y_train)

    # Compute reliability of model on test data
    model_reliability_test = compute_model_reliability(index_test, y_test)
    message_1 = 'The reliability of the model on training data is ' + repr(model_reliability_train) 
    print(message_1)

    message_2 = 'The reliability of the model on test data is ' + repr(model_reliability_test)
    print(message_2)
    
    import os
    # check current working directory
    cwd = os.getcwd()
    model_directory = cwd + '\Model'
    
    try:  
        os.mkdir(model_directory)
    except OSError:  
        print ('Creation of the directory %s failed ' % model_directory)
    else:  
        print ('Successfully created the directory: %s ' % model_directory)
    
    import time
    import datetime
    ts = time.time()
    model_file_name = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d%H%M%S')
    classifier.save(model_directory +'/'+ model_file_name +'.h5')  # creates a HDF5 file     
    return 


# update model when new data comes

def update_lstm(SpathToCsvInputFile, SpathToModelDirectory):
    # prepare the new data
    import numpy as np
    import pandas as pd

    dataset_train = pd.read_csv(SpathToCsvInputFile)
    training_set = dataset_train.iloc[:, 0:6].values
    training_set_new = training_set

    for i in range(len(training_set_new)):
        if training_set[i, 5] < 5:
            training_set_new[i, 5] = 1 # first class
        elif training_set[i, 5] < 30:
            training_set_new[i, 5] = 2 # second class
        elif training_set[i, 5] < 60:
            training_set_new[i, 5] = 3 # third class
        elif training_set[i, 5] < 120:
            training_set_new[i, 5] = 4 # fourth class
        elif training_set[i, 5] < 240:
            training_set_new[i, 5] = 5 # fifth class
        elif training_set[i, 5] < 600:
            training_set_new[i, 5] = 6 # sixth class
        elif training_set[i, 5] > 600:
            training_set_new[i, 5] = 7 # seventh class

    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(training_set_new[:,0:5])

    from sklearn.preprocessing import LabelEncoder
    training_set_new_class = training_set_new[:,5]
    training_set_new_class_reshape = training_set_new_class.reshape(training_set_new_class.size, 1)
    training_set_scaled = np.append(training_set_scaled, training_set_new_class_reshape, axis=1)
    labelencoder = LabelEncoder()
    training_set_scaled[:, 5] = labelencoder.fit_transform(training_set_scaled[:, 5])

    training_size = int(np.ceil(training_set_scaled[:,0].size*1.0))
    lag = 2 # number of realizations in the past the model looks back in order to predict the future
    training_set_final = training_set_scaled[0:training_size,:]
    X_train = []
    y_train = []
    for i in range(lag, training_set_final[:,0].size):
        X_train.append(training_set_final[i-lag:i, 0:5])
        y_train.append(training_set_final[i, 5])
    X_train, y_train = np.array(X_train), np.array(y_train)
    # Reshaping training set
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 5))
    # search model directory to locate latest model
    import os
    os.chdir(SpathToModelDirectory)
    files = filter(os.path.isfile, os.listdir(SpathToModelDirectory))
    files = [os.path.join(SpathToModelDirectory, f) for f in files] # add path to each file
    files.sort(key = lambda x: os.path.getmtime(x))
    latest_model = files[-1]
    
    
    from keras.models import load_model
    classifier = load_model(latest_model)
    #print('Updating the model weights with new data')
    classifier.fit(X_train, y_train, epochs = 2, batch_size = 32)
    print('Model update completed')
    predicted_train = classifier.predict(X_train)
    
    def compute_class_index(pred_softmax):
        output_list_index = []
        for i in range(pred_softmax.shape[0]):
            max_prediction_index = np.argmax(pred_softmax[i, :])
            output_list_index.append(max_prediction_index)
        return output_list_index

    # Predict training data class
    index_train = compute_class_index(predicted_train)
    # Predict test data class 
    # Function that computes the reliability of the model
    def compute_model_reliability(class_index, target):
        out_list_reliability = []
        for z in range(predicted_train.shape[0]):
            if class_index[z] == int(target[z]):
                indicator = 1
            else:
                indicator = 0
            out_list_reliability.append(indicator)
            model_reliability = np.sum(out_list_reliability)/len(out_list_reliability)
        return model_reliability

    # Compute reliability of model on training data
    model_reliability_train = compute_model_reliability(index_train, y_train)

    # Compute reliability of model on test data
    message_1 = 'The reliability of the updated model on training data is ' + repr(model_reliability_train) 
    print(message_1)
    print('Activating version control...')    
    import time
    import datetime
    ts = time.time()
    model_file_name = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d%H%M%S')
    # save the updated model
    classifier.save(SpathToModelDirectory +'/'+ model_file_name +'.h5')  # creates a HDF5 file
    print('Updated model has been saved in the output directory')
    return

def forecast_with_lstm(SpathToInputCsvFileToForcast, SpathToTrainedModel):
    # prepare the new data
    import numpy as np
    import pandas as pd
    dataset_train = pd.read_csv(SpathToInputCsvFileToForcast)
    test_set = dataset_train.iloc[:, 0:5].values   
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range = (0, 1))
    test_set_scaled = sc.fit_transform(test_set[:,0:5])
    lag = 2
    X_test = []
    for i in range(lag, test_set_scaled[:,0].size):
        X_test.append(test_set_scaled[i-lag:i, 0:5])
    X_test = np.array(X_test)
    # Reshaping training set
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 5))
    # load the model
    from keras.models import load_model
    classifier = load_model(SpathToTrainedModel)
    # make predictions
    predicted = classifier.predict(X_test)
    
    def compute_class_index(pred_softmax):
        output_list_index = []
        for i in range(pred_softmax.shape[0]):
            max_prediction_index = np.argmax(pred_softmax[i, :])
            output_list_index.append(max_prediction_index)
        return output_list_index
    
    index = compute_class_index(predicted)
    
    import time
    import datetime
    ts = time.time()
    output_csv_file = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d%H%M%S')
    
    import os
    # check current working directory
    cwd = os.getcwd()
    output_directory = cwd + '\Output'
    
    try:  
        os.mkdir(output_directory)
    except OSError:  
        print ('Creation of the directory %s failed ' % output_directory)
    else:  
        print ('Successfully created the directory and saved results: %s ' % output_directory)
    
    np.savetxt(output_directory +'/' + output_csv_file + '.csv', index, fmt='%.2f', delimiter=',',
               header='Predicted Class')
    return
    
    

    
    

    
    
    







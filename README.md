# Predicting Queue Time with LSTM Classifier

The aim of this project is to build a model and forcast the expected queue time on the Epic cluster at Zenotech adopting supervised machine learning method. Given the fact that the training data provided to train the model is strongly time dependent (i.e. time series data), a Long Short Term Memory (LSTM) model has been adopted. LSTM are units of a recurrent neural network (RNN). A RNN composed of LSTM units is often called an LSTM network. A common LSTM unit is composed of a cell, an input gate, an output gate and a forget gate. The cell remembers values over arbitrary time intervals and the three gates regulate the flow of information into and out of the cell. LSTM networks are well-suited to perfrom classification, regression and make predictions based on time series data, since there can be lags of unknown duration between important events in a time series. 
## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

1) Keras 
2) Numpy
3) Pandas
4) Sklearn
### Installing Keras

Install Keras from PyPI (recommended):
```
sudo pip install keras
```
If you are using a virtualenv, you may want to avoid using sudo:
```
pip install keras
```
Alternatively: install Keras from the GitHub source:
First, clone Keras using git:
```
git clone https://github.com/keras-team/keras.git
```
Then, cd to the Keras folder and run the install command:
```
cd keras
sudo python setup.py install
```
Similarly, Numpy, Pandas and Sklearn can be installed via pip: 
```
pip install numpy

pip install pandas

pip install sklearn
```

## Running the code in IPython console

The main engine of the code is located in cloud_crystal_lstm.py. There are two functions (i.e. train_lstm and update_lstm) inside cloud_crystal_lstm.py. Here, we give a brief deomstration on how to use the functions:
```python
from cloud_crystal_lstm import train_lstm, update_lstm
```

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc



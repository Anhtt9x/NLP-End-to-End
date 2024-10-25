from keras._tf_keras.keras.layers import LSTM , Activation, Dense, Dropout,  Embedding, Input,SpatialDropout1D
from keras._tf_keras.keras.models import Sequential
from src.entity.config_entity import ModelTrainerConfig
from keras._tf_keras.keras.optimizers import RMSprop
from keras._tf_keras.keras.callbacks import ModelCheckpoint, EarlyStopping
from src.constants import *

class ModelArchitecture:
    def __init__(self):
        pass

    def create_model(self):
        model = Sequential()
        model.add(Embedding(input_dim=MAX_WORDS, output_dim=100, input_length=MAX_LEN))
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(100,dropout=0.2,recurrent_dropout=0.2))
        model.add(Dense(1, activation=ACTIVATION))
        model.compile(loss=LOSS, optimizer=RMSprop(epsilon=1e-8), metrics=METRICS)
        model.summary()

        return model
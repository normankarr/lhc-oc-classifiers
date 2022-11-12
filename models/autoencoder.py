import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization

def autoencoder():
    ae = Sequential()
    # Encoder Portion
    ae.add(Dense(512, input_dim=5, activation='relu'))
    ae.add(BatchNormalization())
    ae.add(Dense(512, activation='relu'))
    ae.add(Dense(2, activation='linear'))
    
    # Decoder Portion
    ae.add(Dense(512, input_dim=2, activation='relu'))
    ae.add(Dense(512, activation='relu'))
    ae.add(Dense(5, activation='linear'))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    ae.compile(loss='mse', optimizer=optimizer)
    return ae

def reconstruction_error(ae_model, X):
    X_pred = ae_model(X)
    return np.linalg.norm(X-X_pred, axis=1)
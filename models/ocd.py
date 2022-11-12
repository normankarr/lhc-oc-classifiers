import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

def OCD(gauss=False, reg=0.0):
    model = Sequential()
    model.add(Dense(128, input_dim=5, activation='relu', activity_regularizer=tf.keras.regularizers.L2(reg))) 
    model.add(Dense(128, activation='relu', activity_regularizer=tf.keras.regularizers.L2(reg)))
    model.add(Dense(128, activation='relu', activity_regularizer=tf.keras.regularizers.L2(reg)))
    if gauss:
        model.add(Dense(1, use_bias=True, activation='linear'))
        model.add(tf.keras.layers.Lambda(gaussian))
    else:
        model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
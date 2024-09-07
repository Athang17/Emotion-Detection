from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def create_model():
    model = Sequential();

    #First layer
    model.add(Conv2D(32,(3,3),activation='relu',input_shape=(48,48,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))

    #Second layer
    model.add(Conv2D(64,(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    #Third layer
    model.add(Conv2D(128,(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    #Fourth layer
    model.add(Conv2D(256,(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    #Flattening layer
    model.add(Flatten())

    #Fully Connected Dense Layer
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.5))

    #Output
    model.add(Dense(7, activation='softmax'))

    #Compiling the model
    model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])

    return model
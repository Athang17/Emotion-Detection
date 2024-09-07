from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

def create_model():
    model = Sequential();

    #First layer
    model.add(Conv2D(32,(3,3),activation='relu',input_shape=(48,48,1),kernel_regularizer=l2(0.05), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32,(3,3),activation='relu',padding = 'same'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    #Second layer
    model.add(Conv2D(64,(3,3),activation='relu',kernel_regularizer=l2(0.05),padding = 'same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64,(3,3),activation='relu',padding = 'same'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    #Third layer
    model.add(Conv2D(128,(3,3),activation='relu',kernel_regularizer=l2(0.05),padding = 'same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128,(3,3),activation='relu',padding = 'same'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    #Fourth layer
    model.add(Conv2D(256,(3,3),activation='relu',kernel_regularizer=l2(0.05),padding = 'same'))
    model.add(BatchNormalization())
    model.add(Conv2D(256,(3,3),activation='relu',padding = 'same'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    #Flattening layer
    model.add(Flatten())

    #Fully Connected Dense Layer
    model.add(Dense(512,activation='relu',kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))

    #Output
    model.add(Dense(7, activation='softmax'))

    #Compiling the model
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,loss='categorical_crossentropy', metrics=['accuracy'])

    return model
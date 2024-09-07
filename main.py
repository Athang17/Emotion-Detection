import deeplake #dataset
import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

#Loading the databases
train_ds = deeplake.load('hub://activeloop/fer2013-train')
public_test_ds = deeplake.load('hub://activeloop/fer2013-public-test')
private_test_ds = deeplake.load('hub://activeloop/fer2013-private-test')

def preprocess(ds):
    #Extracting images and label and converting them to numpy arrays
    images = ds['images'][:].numpy()
    labels = ds['labels'][:].numpy()

    #Normalize the image value
    images = images/255.0

    labels = tf.keras.utils.to_categorical(labels, num_classes=7)

    return images, labels

#Preprocess the training and testing datasets
X_train, y_train = preprocess(train_ds)
X_public_test, y_public_test = preprocess(public_test_ds)
X_private_test, y_private_test = preprocess(private_test_ds) 

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)

def model():
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

def train_model(model, X_train, y_train, X_valid, y_valid):
    model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=100, batch_size=32)

#Build and train the model
model = model()
train_model = (model, X_train, y_train, X_valid, y_valid)

def evaluate(model, X_test, y_test):
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {test_acc:.4f}')

#Evaluate on private test dataset
evaluate(model, X_private_test, y_private_test)

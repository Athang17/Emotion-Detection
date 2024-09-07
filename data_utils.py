import deeplake #dataset
import tensorflow as tf 

def load_data():
    #Loading the databases
    train_ds = deeplake.load('hub://activeloop/fer2013-train')
    public_test_ds = deeplake.load('hub://activeloop/fer2013-public-test')
    private_test_ds = deeplake.load('hub://activeloop/fer2013-private-test')

    return train_ds,public_test_ds,private_test_ds

def preprocess(ds):
    #Extracting images and label and converting them to numpy arrays
    images = ds['images'][:].numpy()
    labels = ds['labels'][:].numpy()

    #Normalize the image value
    images = images/255.0

    #One hot encoding
    labels = tf.keras.utils.to_categorical(labels, num_classes=7)

    return images, labels
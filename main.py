from data_utils import load_data,preprocess,augment_data
from models import create_model
from train import train_model
from evaluate import evaluate_model
from sklearn.model_selection import train_test_split

def main():
    #Loading the data
    train_ds, public_test_ds, private_test_ds = load_data()

    #Preprocess the training and testing datasets
    X_train, y_train = preprocess(train_ds)
    X_public_test, y_public_test = preprocess(public_test_ds)
    X_private_test, y_private_test = preprocess(private_test_ds) 

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)


    #Augmenting the training data
    datagen = augment_data(X_train, y_train)

    #Build and train the model
    model = create_model()
    train_model(model, X_train, y_train, X_valid, y_valid, datagen) 

    #Evaluate on private test dataset
    evaluate_model(model, X_private_test, y_private_test)

if __name__ == "__main__":
    main()
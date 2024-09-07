def train_model(model, X_train, y_train, X_valid, y_valid):
    model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=100, batch_size=32)
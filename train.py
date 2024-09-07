def train_model(model, X_train, y_train, X_valid, y_valid,datagen):
    model.fit(
        datagen.flow(X_train,y_train,batch_size=32),
        validation_data=(X_valid,y_valid),
        epochs=100
    )
    print("Saving the model...")
    model.save(r'D:\NMIMS\Sem 5\IVP\Emotion-Detection\emotion_detection_model.h5')
    print("Model saved")
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Flatten, Input, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

MODEL_PATH = "mnist_model.keras"
NUM_CLASSES = 10
IMAGE_SHAPE = (28,28)
EPOCHS = 5
BATCH_SIZE = 32


def load_data():
    (x_train,y_train),(x_test,y_test) = mnist.load_data()
    x_train = x_train.astype("float32")/255
    x_test = x_test.astype("float32")/255
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return (x_train,y_train),(x_test,y_test)


def build_model():
    model = Sequential([
        Input(shape=IMAGE_SHAPE),
        Flatten(),
        Dense(64,activation="relu"),
        Dropout(0.2),
        Dense(NUM_CLASSES,activation="softmax"),

    ],name="mnist_classifier")
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics = ["accuracy"],   
    )
    return model

def train(model,x_train,y_train,epochs=EPOCHS,batch_size= BATCH_SIZE):
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split = 0.1,
        verbose = 1,
    )
    return history

def evaluate(model , x_test,y_test):
    loss , acc = model.evaluate(
        x_test,
        y_test,
        verbose=0
    )
    print(f"Test accuracy : {acc:.4f}")
    return loss,acc

def predict_one(model,image):
    img = np.array(image,dtype="float32")

    if img.ndim == 2:
        img = img.reshape(1,28,28)
    if img.max() > 1.0:
        img = img / 255.0

    probs = model.predict(img,verbose=0)
    pred = int(np.argmax(probs))

    return pred,probs

def save_model(model,path=MODEL_PATH):
    model.save(path)
    print(f"model saved to: {path}")

def load_trained_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"no model at {path} please the model first the check!")
    return load_model()
    

if __name__ == "__main__":
    (x_train,y_train),(x_test,y_test) = load_data()
    model = build_model()
    model.summary()
    train(model, x_train, y_train)
    evaluate(model, x_test, y_test)
    save_model(model)

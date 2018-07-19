# It is easier to classify into two categories, rather than 10, as there are certain attributes of cars and animals that are common across all the varieties
# of cars and animals. For example, finding an eye would automatically classify the image as an animal, and there is no need to further classify it as a
# particular animal, or finding a wheel/headlight for cars. Hence, binary classification is easier.
import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from keras import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras import optimizers


def load_cifar10():
    train, test = cifar10.load_data()
    xtrain, ytrain = train
    xtest, ytest = test

    ytrain_1hot = np.zeros((50000, 10))
    ytest_1hot = np.zeros((10000, 10))
    for i in range(0, 50000):
        ytrain_1hot[i][ytrain[i][0]] = 1
    for i in range(0, 10000):
        ytest_1hot[i][ytest[i][0]] = 1

    xtrain = xtrain/255
    xtest = xtest/255

    return xtrain, ytrain_1hot, xtest, ytest_1hot

# output from the evaluate function at the end of Part 2: [1.4240363317489624, 0.4975]
def build_multilayer_nn():
    nn = Sequential()
    nn.add(Flatten(input_shape=(32, 32, 3)))

    hidden = Dense(units=100, activation="relu")
    nn.add(hidden)

    output = Dense(units=10, activation="softmax")
    nn.add(output)

    return nn


def train_multilayer_nn(model, xtrain, ytrain_1hot):
    sgd = optimizers.SGD(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy']) 
    model.fit(xtrain, ytrain_1hot, epochs=20, batch_size=32)
 
# Original evaluate function output: [0.83004944944381709, 0.70699999999999996]
# Without the second pooling layer, and 30 epochs, the evaluate function output: [0.78973214600086217, 0.76049999999999995]
def build_convolution_nn():
    nn = Sequential()

    nn.add(Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=(32, 32, 3)))
    nn.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
    nn.add(MaxPooling2D(pool_size=(2, 2)))
    nn.add(Dropout(0.25))
    nn.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
    nn.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
    # nn.add(MaxPooling2D(pool_size=(2, 2)))
    nn.add(Dropout(0.5))

    nn.add(Flatten())
    hidden_1 = Dense(units=250, activation='relu')
    nn.add(hidden_1)
    hidden_2 = Dense(units=100, activation='relu')
    nn.add(hidden_2)
    output = Dense(units=10, activation='softmax')
    nn.add(output)

    return nn
    

def train_convolution_nn(model, xtrain, ytrain_1hot):
    sgd = optimizers.SGD(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(xtrain, ytrain_1hot, epochs=30, batch_size=32)
    

def get_binary_cifar10():
    train, test = cifar10.load_data()
    xtrain, ytrain_initial = train
    xtest, ytest_initial = test

    ytrain = np.zeros(50000)
    for i in range(0, 50000):
        ytrain[i] = 1 if (ytrain_initial[i][0] > 1 and ytrain_initial[i][0] < 8) else 0
    ytest = np.zeros(10000)
    for i in range(0, 10000):
        ytest[i] = 1 if (ytest_initial[i][0] > 1 and ytest_initial[i][0] < 8) else 0

    xtrain = xtrain / 255
    xtest = xtest / 255

    return xtrain, ytrain, xtest, ytest

# Original evaluate function output (using same network as part 3): [0.15681286436617375, 0.93869999999999998]
def build_binary_classifier():
    nn = Sequential()

    nn.add(Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=(32, 32, 3)))
    nn.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
    nn.add(MaxPooling2D(pool_size=(2, 2)))
    nn.add(Dropout(0.25))
    nn.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
    nn.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
    nn.add(MaxPooling2D(pool_size=(2, 2)))
    nn.add(Dropout(0.50))

    nn.add(Flatten())
    hidden_1 = Dense(units=250, activation='relu')
    nn.add(hidden_1)
    hidden_2 = Dense(units=100, activation='relu')
    nn.add(hidden_2)
    output = Dense(units=1, activation='sigmoid')
    nn.add(output)

    return nn


def train_binary_classifier(model, xtrain, ytrain):
    sgd = optimizers.SGD(lr=0.01)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(xtrain, ytrain, epochs=20, batch_size=32)


if __name__ == "__main__":

    # Write any code for testing and evaluation in this main section.
    xtrain, ytrain_1hot, xtest, ytest_1hot = get_binary_cifar10()
    nn = build_binary_classifier()
    # nn.summary()
    train_binary_classifier(nn, xtrain, ytrain_1hot)
    output = nn.evaluate(xtest, ytest_1hot)
    print(output)
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models, layers
import numpy as np

# 1. Przygotowanie danych
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizacja i spłaszczenie
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

# Zamiana cyfr na format "one-hot" (np. 5 -> [0,0,0,0,0,1,0,0,0,0])
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 2. Budowa architektury (Warstwy Gęste)
model = models.Sequential([
    # Warstwa wejściowa + pierwsza ukryta (512 neuronów)
    layers.Dense(512, activation='relu', input_shape=(784,)),
    # Druga warstwa ukryta (wyłapuje bardziej złożone cechy)
    layers.Dense(256, activation='relu'),
    # Warstwa wyjściowa (10 neuronów - po jednym dla każdej cyfry)
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=128)

# Sprawdzenie skuteczności na nowych danych
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Skuteczność sieci: {test_acc * 100:.2f}%")

# --------------------------------------------------------------------
# Zamiana sieci na CNN
# --------------------------------------------------------------------

(x_train_cnn, y_train_cnn), (x_test_cnn, y_test_cnn) = mnist.load_data()

x_train_cnn = np.reshape(
    x_train_cnn, (len(x_train_cnn), 28, 28, 1)).astype('float32') / 255
x_test_cnn = np.reshape(
    x_test_cnn, (len(x_test_cnn), 28, 28, 1)).astype('float32') / 255

y_train_cnn = to_categorical(y_train_cnn, 10)
y_test_cnn = to_categorical(y_test_cnn, 10)

model_CNN = models.Sequential([
    # 1. Pierwsza warstwa konwolucyjna (szuka krawędzi)
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),

    # 2. Druga warstwa konwolucyjna (szuka bardziej złożonych kształtów)
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # 3. Trzecia warstwa konwolucyjna
    layers.Conv2D(64, (3, 3), activation='relu'),

    # --- PRZEJŚCIE DO KLASYFIKACJI ---
    layers.Flatten(),  # Zamiana map cech na długi wektor przed warstwami gęstymi
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 wyjść (cyfry 0-9)
])

model_CNN.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

model_CNN.fit(x_train_cnn, y_train_cnn, epochs=10, batch_size=64)

test_loss_CNN, test_acc_CNN = model_CNN.evaluate(x_test_cnn, y_test_cnn)
print(f"Skuteczność sieci: {test_acc_CNN * 100:.2f}%")

model_CNN.summary()

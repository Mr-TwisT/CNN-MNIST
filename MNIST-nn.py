from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models, layers

# 1. Przygotowanie danych
# Ładowanie danych
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

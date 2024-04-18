import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

x = np.array([[-3], [1], [3]])
y = np.array([-2, -2, 0])

neuron_counts = [1, 2, 4]
models = []
loss_histories = []

for count in neuron_counts:
    model = keras.models.Sequential()
    model.add(
        keras.layers.Dense(
            count,
            use_bias=True,
            kernel_initializer=keras.initializers.constant(0.0),
            bias_initializer=keras.initializers.constant(0.0),
        )
    )
    if count > 1:
        model.add(keras.layers.Dense(1))
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=0.1, momentum=0.0),
        loss='mse',
        metrics=['mse'],
    )
    history = model.fit(x, y, epochs=5)
    models.append(model)
    loss_histories.append(history.history['loss'])

plt.figure(figsize=(10, 6))
for i, history in enumerate(loss_histories):
    plt.plot(history, label=f'{neuron_counts[i]} кол-во нейронов')
plt.title('Изменения функции потерь в процессе обучения;')
plt.xlabel('epochs')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='red', label='Набор данных')

weights = models[0].get_weights()
a = weights[0][0][0]
b = weights[1][0]
x_values = np.linspace(-4, 4, 100)
y_values = a * x_values + b
plt.plot(x_values, y_values, label=f'Функция: y = {a:.2f}x + {b:.2f}')

plt.title('Набор данных и функция')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()

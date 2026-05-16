import os
import time

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

class_names = ["Samolot", "Samochód", "Ptaszek", "Kot", "Jelonek", "Pies",
               "Żaba", "Koń", "Statek", "Ciężarówka"]

plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_train[i])
    plt.title(class_names[int(y_train[i])])
    plt.axis("off")
plt.show()


def create_model(activation='relu', dropout=False, bigger=False):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation=activation, input_shape=(32, 32, 3)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation=activation))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation=activation))

    if bigger:
        model.add(tf.keras.layers.Conv2D(128, (3, 3), activation=activation))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(64, activation=activation))

    if dropout:
        model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


models = {
    "Bazowy_relu": create_model(activation='relu', dropout=False, bigger=False),
    "Relu_Dropout": create_model(activation='relu', dropout=True, bigger=False),
    "Tanh_Dropout": create_model(activation='tanh', dropout=True, bigger=False),
    "Wiekszy_relu_Dropout": create_model(activation='relu', dropout=True, bigger=True)
}

results = {}

for name, model in models.items():
    print("\nTrenowanie modelu:", name)
    model.summary()

    start = time.time()

    history = model.fit(
        x_train,
        y_train,
        epochs=10,
        validation_data=(x_test, y_test)
    )

    end = time.time()

    test_loss, test_accuracy = model.evaluate(x_test, y_test)

    results[name] = {
        "accuracy": test_accuracy,
        "loss": test_loss,
        "time": end - start,
        "history": history
    }

    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoka')
    plt.ylabel('Dokładność')
    plt.legend(loc='lower right')
    plt.title('Dokładność modelu: ' + name)
    plt.show()


print("\nPorównanie modeli:")
for name, result in results.items():
    print(name)
    print("Test accuracy:", result["accuracy"])
    print("Test loss:", result["loss"])
    print("Czas trenowania:", round(result["time"], 2), "s")
    print()


best_model_name = max(results, key=lambda x: results[x]["accuracy"])
print("Najlepszy model:", best_model_name)
print("Najlepsza dokładność:", results[best_model_name]["accuracy"])


best_model = models[best_model_name]
predictions = best_model.predict(x_test)

plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_test[i])
    plt.title(
        f"True: {class_names[int(y_test[i])]}, " +
        f"Predicted: {class_names[tf.argmax(predictions[i])]}"
    )
    plt.axis("off")
plt.show()
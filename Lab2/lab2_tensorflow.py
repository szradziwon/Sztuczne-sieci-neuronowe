import os
import time

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

plt.imshow(x_train[0], cmap='gray')
plt.title("Label: " + str(y_train[0]))
plt.show()

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

start_time = time.time()
model.fit(x_train, y_train, epochs=5)
test_loss, test_accuracy = model.evaluate(x_test, y_test)
end_time = time.time()
elapsed_time = end_time - start_time

print("Basic model training took {} seconds".format(elapsed_time))
print("Test accuracy:", test_accuracy)
print("Test loss:", test_loss)
predictions = model.predict(x_test)
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"True: {y_test[i]}, Predicted: {tf.argmax(predictions[i])}")
    plt.axis('off')
    plt.show()

#1. changing epochs
number_of_epochs = [1,2,5,10,15,30,100]
epoch_change_result = dict()

for epoch in number_of_epochs:
    start_time = time.time()
    model.fit(x_train, y_train, epochs=epoch)
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    end_time = time.time()
    elapsed_time = end_time - start_time
    epoch_change_result[epoch] = (test_accuracy, elapsed_time)
    print("Test accuracy:", test_accuracy)
    predictions = model.predict(x_test)

for result in epoch_change_result:
    print(f"epoch number {result}: accuracy- {epoch_change_result[result][0]}, elapsed time- {epoch_change_result[result][1]}")

#2. diffrent layers count
model_for_1_additional_layer = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model_for_1_additional_layer.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
start_time = time.time()
model_for_1_additional_layer.fit(x_train, y_train, epochs=5)
test_loss, test_accuracy = model_for_1_additional_layer.evaluate(x_test, y_test)
end_time = time.time()
elapsed_time = end_time - start_time
print("Test accuracy:", test_accuracy)
print("Test loss:", test_loss)
print("elapsed time:", elapsed_time)
predictions = model_for_1_additional_layer.predict(x_test)

plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"True: {y_test[i]}, Predicted: {tf.argmax(predictions[i])}")
    plt.axis('off')
    plt.show()

model_for_2_additional_layers = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model_for_2_additional_layers.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

start_time = time.time()
model_for_2_additional_layers.fit(x_train, y_train, epochs=5)
test_loss, test_accuracy = model_for_2_additional_layers.evaluate(x_test, y_test)
end_time = time.time()
elapsed_time = end_time - start_time
print("Test accuracy:", test_accuracy)
print("Test loss:", test_loss)
print("elapsed time:", elapsed_time)
predictions = model_for_2_additional_layers.predict(x_test)

plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"True: {y_test[i]}, Predicted: {tf.argmax(predictions[i])}")
    plt.axis('off')
    plt.show()

#3 changed neurons count
neurons_count = [64,128,256]
neuron_change_result = dict()

for neurons in neurons_count:

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(neurons, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    start_time = time.time()
    model.fit(x_train, y_train, epochs=5)
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    end_time = time.time()
    elapsed_time = end_time - start_time
    neuron_change_result[neurons] = (test_accuracy, elapsed_time)
    print("Test accuracy:", test_accuracy)
    predictions = model.predict(x_test)

    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(x_test[i], cmap='gray')
        plt.title(f"True: {y_test[i]}, Predicted: {tf.argmax(predictions[i])}")
        plt.axis('off')
        plt.show()

for result in neuron_change_result:
    print(
        f"neuron number {result}: accuracy- {neuron_change_result[result][0]}, "
        f"elapsed time- {neuron_change_result[result][1]}")

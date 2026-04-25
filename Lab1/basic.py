import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

x_reg, y_reg = make_regression(n_samples=1000, n_features=20, noise=0.1)
x_clf, y_clf = make_classification(n_samples=1000,
    n_features=20,
    n_informative=12,
    n_redundant=4,
    n_classes=3,
    random_state=42
)

x_reg_train, x_reg_test, y_reg_train, y_reg_test = train_test_split(
    x_reg, y_reg, test_size=0.2)

x_clf_train, x_clf_test, y_clf_train, y_clf_test = train_test_split(
    x_clf, y_clf, test_size=0.2)

model_reg = MLPRegressor(hidden_layer_sizes=(10,), activation='relu', max_iter=1000)
model_reg.fit(x_reg_train, y_reg_train)
y_reg_pred = model_reg.predict(x_reg_test)
mse_basic = mean_squared_error(y_reg_test, y_reg_pred)

print("regresja")
print(f"MSE: {mse_basic:.4f}")
print()

model_clf = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', max_iter=1000)
model_clf.fit(x_clf_train, y_clf_train)
y_clf_pred = model_clf.predict(x_clf_test)
acc_basic = accuracy_score(y_clf_test, y_clf_pred)
cm_basic = confusion_matrix(y_clf_test, y_clf_pred)

print("klasyfikacja")
print(f"Accuracy: {acc_basic:.4f}")
print("Confusion matrix:")
print(cm_basic)
print()

#analiza parametrów
neurons_list = [5, 10, 20, 50, 100]
layers_list = [(10,), (50,), (50, 50), (100, 50), (100, 100, 50)]
activations = ['relu', 'tanh', 'logistic']
learning_rates = [0.0001, 0.001, 0.01, 0.1]

results_reg_neurons = []
results_reg_layers = []
results_reg_activations = []
results_reg_lr = []

#zmiany parametrów dla regresji
#Liczba neuronów
for n in neurons_list:
    model = MLPRegressor(
        hidden_layer_sizes=(n,),
        activation='relu',
        learning_rate_init=0.001,
        max_iter=1000,
        random_state=42
    )
    model.fit(x_reg_train, y_reg_train)
    pred = model.predict(x_reg_test)
    mse = mean_squared_error(y_reg_test, pred)
    results_reg_neurons.append(mse)

#Liczba warstw
for layers in layers_list:
    model = MLPRegressor(
        hidden_layer_sizes=layers,
        activation='relu',
        learning_rate_init=0.001,
        max_iter=1000,
        random_state=42
    )
    model.fit(x_reg_train, y_reg_train)
    pred = model.predict(x_reg_test)
    mse = mean_squared_error(y_reg_test, pred)
    results_reg_layers.append(mse)

#Funkcja aktywacji
for act in activations:
    model = MLPRegressor(
        hidden_layer_sizes=(50,),
        activation=act,
        learning_rate_init=0.001,
        max_iter=1000,
        random_state=42
    )
    model.fit(x_reg_train, y_reg_train)
    pred = model.predict(x_reg_test)
    mse = mean_squared_error(y_reg_test, pred)
    results_reg_activations.append(mse)

#Learning rate
for lr in learning_rates:
    model = MLPRegressor(
        hidden_layer_sizes=(50,),
        activation='relu',
        learning_rate_init=lr,
        max_iter=1000,
        random_state=42
    )
    model.fit(x_reg_train, y_reg_train)
    pred = model.predict(x_reg_test)
    mse = mean_squared_error(y_reg_test, pred)
    results_reg_lr.append(mse)

#zmiany parametrów dla klasyfikacja
results_clf_neurons = []
results_clf_layers = []
results_clf_activations = []
results_clf_lr = []
conf_matrices = {}

#Liczba neuronów
for n in neurons_list:
    model = MLPClassifier(
        hidden_layer_sizes=(n,),
        activation='relu',
        learning_rate_init=0.001,
        max_iter=1000,
        random_state=42
    )
    model.fit(x_clf_train, y_clf_train)
    pred = model.predict(x_clf_test)
    acc = accuracy_score(y_clf_test, pred)
    results_clf_neurons.append(acc)

#Liczba warstw
for layers in layers_list:
    model = MLPClassifier(
        hidden_layer_sizes=layers,
        activation='relu',
        learning_rate_init=0.001,
        max_iter=1000,
        random_state=42
    )
    model.fit(x_clf_train, y_clf_train)
    pred = model.predict(x_clf_test)
    acc = accuracy_score(y_clf_test, pred)
    results_clf_layers.append(acc)

#Funkcja aktywacji
for act in activations:
    model = MLPClassifier(
        hidden_layer_sizes=(50,),
        activation=act,
        learning_rate_init=0.001,
        max_iter=1000,
        random_state=42
    )
    model.fit(x_clf_train, y_clf_train)
    pred = model.predict(x_clf_test)
    acc = accuracy_score(y_clf_test, pred)
    results_clf_activations.append(acc)
    conf_matrices[act] = confusion_matrix(y_clf_test, pred)

#Learning rate
for lr in learning_rates:
    model = MLPClassifier(
        hidden_layer_sizes=(50,),
        activation='relu',
        learning_rate_init=lr,
        max_iter=1000,
        random_state=42
    )
    model.fit(x_clf_train, y_clf_train)
    pred = model.predict(x_clf_test)
    acc = accuracy_score(y_clf_test, pred)
    results_clf_lr.append(acc)

#wyniki
print("REGRESJA - wpływ liczby neuronów")
for n, mse in zip(neurons_list, results_reg_neurons):
    print(f"Neurony: {n:3d} | MSE: {mse:.4f}")
print()

print("REGRESJA - wpływ liczby warstw")
for layers, mse in zip(layers_list, results_reg_layers):
    print(f"Warstwy: {layers} | MSE: {mse:.4f}")
print()

print("REGRESJA - wpływ funkcji aktywacji")
for act, mse in zip(activations, results_reg_activations):
    print(f"Aktywacja: {act:8s} | MSE: {mse:.4f}")
print()

print("REGRESJA - wpływ learning rate")
for lr, mse in zip(learning_rates, results_reg_lr):
    print(f"Learning rate: {lr:7f} | MSE: {mse:.4f}")
print()

print("KLASYFIKACJA - wpływ liczby neuronów")
for n, acc in zip(neurons_list, results_clf_neurons):
    print(f"Neurony: {n:3d} | Accuracy: {acc:.4f}")
print()

print("KLASYFIKACJA - wpływ liczby warstw")
for layers, acc in zip(layers_list, results_clf_layers):
    print(f"Warstwy: {layers} | Accuracy: {acc:.4f}")
print()

print("KLASYFIKACJA - wpływ funkcji aktywacji")
for act, acc in zip(activations, results_clf_activations):
    print(f"Aktywacja: {act:8s} | Accuracy: {acc:.4f}")
print()

print("KLASYFIKACJA - wpływ learning rate")
for lr, acc in zip(learning_rates, results_clf_lr):
    print(f"Learning rate: {lr:7f} | Accuracy: {acc:.4f}")
print()

print("Macierze pomyłek dla klasyfikacji")
for act, cm in conf_matrices.items():
    print(f"\nAktywacja: {act}")
    print(cm)

#wykresy
# Regresja - neurony
plt.figure(figsize=(8, 5))
plt.plot(neurons_list, results_reg_neurons, marker='o')
plt.title("Regresja - wpływ liczby neuronów na MSE")
plt.xlabel("Liczba neuronów")
plt.ylabel("MSE")
plt.grid(True)
plt.show()

# Regresja - warstwy
plt.figure(figsize=(8, 5))
plt.plot([str(x) for x in layers_list], results_reg_layers, marker='o')
plt.title("Regresja - wpływ liczby warstw na MSE")
plt.xlabel("Układ warstw")
plt.ylabel("MSE")
plt.grid(True)
plt.show()

# Regresja - aktywacja
plt.figure(figsize=(8, 5))
plt.plot(activations, results_reg_activations, marker='o')
plt.title("Regresja - wpływ funkcji aktywacji na MSE")
plt.xlabel("Funkcja aktywacji")
plt.ylabel("MSE")
plt.grid(True)
plt.show()

# Regresja - learning rate
plt.figure(figsize=(8, 5))
plt.plot([str(x) for x in learning_rates], results_reg_lr, marker='o')
plt.title("Regresja - wpływ learning rate na MSE")
plt.xlabel("Learning rate")
plt.ylabel("MSE")
plt.grid(True)
plt.show()

# Klasyfikacja - neurony
plt.figure(figsize=(8, 5))
plt.plot(neurons_list, results_clf_neurons, marker='o')
plt.title("Klasyfikacja - wpływ liczby neuronów na accuracy")
plt.xlabel("Liczba neuronów")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

# Klasyfikacja - warstwy
plt.figure(figsize=(8, 5))
plt.plot([str(x) for x in layers_list], results_clf_layers, marker='o')
plt.title("Klasyfikacja - wpływ liczby warstw na accuracy")
plt.xlabel("Układ warstw")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

# Klasyfikacja - aktywacja
plt.figure(figsize=(8, 5))
plt.plot(activations, results_clf_activations, marker='o')
plt.title("Klasyfikacja - wpływ funkcji aktywacji na accuracy")
plt.xlabel("Funkcja aktywacji")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

# Klasyfikacja - learning rate
plt.figure(figsize=(8, 5))
plt.plot([str(x) for x in learning_rates], results_clf_lr, marker='o')
plt.title("Klasyfikacja - wpływ learning rate na accuracy")
plt.xlabel("Learning rate")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

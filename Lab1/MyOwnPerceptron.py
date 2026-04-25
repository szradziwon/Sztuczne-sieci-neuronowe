import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

np.random.seed(42)


# Własna implementacja perceptronu wielowarstwowego
class MyPerceptron:
    def __init__(self, hidden_layer_sizes=(10,), activation='relu',
                 learning_rate=0.001, max_iter=1000, mode='regression'):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.mode = mode
        self.weights = []
        self.biases = []
        self.loss_history = []

    #funkcje aktywacji
    @staticmethod
    def _relu(z):
        return np.maximum(0, z)

    @staticmethod
    def _relu_deriv(z):
        return (z > 0).astype(float)

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def _sigmoid_deriv(z):
        s = MyPerceptron._sigmoid(z)
        return s * (1 - s)

    @staticmethod
    def _softmax(z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _get_activation(self, name):
        if name == 'relu':
            return self._relu, self._relu_deriv
        elif name == 'sigmoid':
            return self._sigmoid, self._sigmoid_deriv
        else:
            raise ValueError(f"Nieznana funkcja aktywacji: {name}")

    def _init_weights(self, layer_sizes):
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            w = np.random.randn(fan_in, layer_sizes[i + 1]) * np.sqrt(2.0 / fan_in)
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)


    def _forward(self, X):
        act_fn, _ = self._get_activation(self.activation)
        a = X
        pre_activations = []  
        activations = [a]     
        for i in range(len(self.weights) - 1):
            z = a @ self.weights[i] + self.biases[i]
            pre_activations.append(z)
            a = act_fn(z)
            activations.append(a)


        z_out = a @ self.weights[-1] + self.biases[-1]
        pre_activations.append(z_out)

        if self.mode == 'classification':
            a_out = self._softmax(z_out)
        else:
            a_out = z_out

        activations.append(a_out)
        return pre_activations, activations

    def _compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        if self.mode == 'classification':
            eps = 1e-15
            y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
            loss = -np.sum(y_true * np.log(y_pred_clipped)) / m
        else:
            loss = np.mean((y_true - y_pred) ** 2)
        return loss

    def _backward(self, X, y, pre_activations, activations):
        _, act_deriv = self._get_activation(self.activation)
        m = X.shape[0]
        n_layers = len(self.weights)
        grads_w = [None] * n_layers
        grads_b = [None] * n_layers

        if self.mode == 'classification':
            delta = activations[-1] - y
        else:
            delta = 2 * (activations[-1] - y) / m

        grads_w[-1] = activations[-2].T @ delta / m
        grads_b[-1] = np.sum(delta, axis=0, keepdims=True) / m
        for i in range(n_layers - 2, -1, -1):
            delta = (delta @ self.weights[i + 1].T) * act_deriv(pre_activations[i])
            grads_w[i] = activations[i].T @ delta / m
            grads_b[i] = np.sum(delta, axis=0, keepdims=True) / m

        return grads_w, grads_b

    def fit(self, X, y):
        if self.mode == 'classification':
            self.classes_ = np.unique(y)
            n_classes = len(self.classes_)
            y_onehot = np.zeros((y.shape[0], n_classes))
            for idx, cls in enumerate(self.classes_):
                y_onehot[y == cls, idx] = 1
            y_train = y_onehot
            output_size = n_classes
        else:
            y_train = y.reshape(-1, 1) if y.ndim == 1 else y
            output_size = y_train.shape[1]

        layer_sizes = [X.shape[1]] + list(self.hidden_layer_sizes) + [output_size]
        self._init_weights(layer_sizes)

        self.loss_history = []
        for epoch in range(self.max_iter):
            pre_act, acts = self._forward(X)
            loss = self._compute_loss(y_train, acts[-1])
            self.loss_history.append(loss)

            grads_w, grads_b = self._backward(X, y_train, pre_act, acts)

            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * grads_w[i]
                self.biases[i] -= self.learning_rate * grads_b[i]

        return self

    def predict(self, X):
        _, acts = self._forward(X)
        output = acts[-1]

        if self.mode == 'classification':
            return self.classes_[np.argmax(output, axis=1)]
        else:
            return output.ravel()

    def predict_proba(self, X):
        _, acts = self._forward(X)
        return acts[-1]


# generowanie danych

# regresja
x_reg, y_reg = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
x_reg_train, x_reg_test, y_reg_train, y_reg_test = train_test_split(
    x_reg, y_reg, test_size=0.2, random_state=42)

scaler_reg = StandardScaler()
x_reg_train_s = scaler_reg.fit_transform(x_reg_train)
x_reg_test_s = scaler_reg.transform(x_reg_test)

scaler_y = StandardScaler()
y_reg_train_s = scaler_y.fit_transform(y_reg_train.reshape(-1, 1)).ravel()
y_reg_test_s = scaler_y.transform(y_reg_test.reshape(-1, 1)).ravel()

x_clf, y_clf = make_classification(
    n_samples=1000, n_features=20, n_informative=12,
    n_redundant=4, n_classes=3, random_state=42
)
x_clf_train, x_clf_test, y_clf_train, y_clf_test = train_test_split(
    x_clf, y_clf, test_size=0.2, random_state=42)

scaler_clf = StandardScaler()
x_clf_train_s = scaler_clf.fit_transform(x_clf_train)
x_clf_test_s = scaler_clf.transform(x_clf_test)


# Regresja własny perceptron (1 warstwa ukryta, 10 neuronów)
print("Własna implementacja perceptronu - regresja")

t0 = time.time()
my_reg = MyPerceptron(
    hidden_layer_sizes=(10,),
    activation='relu',
    learning_rate=0.001,
    max_iter=1000,
    mode='regression'
)
my_reg.fit(x_reg_train_s, y_reg_train_s)
t_my_reg = time.time() - t0

y_pred_my_s = my_reg.predict(x_reg_test_s)
y_pred_my = scaler_y.inverse_transform(y_pred_my_s.reshape(-1, 1)).ravel()
mse_my = mean_squared_error(y_reg_test, y_pred_my)

print(f"Własny perceptron - MSE:  {mse_my:.4f}  (czas: {t_my_reg:.3f}s)")


plt.figure(figsize=(8, 5))
plt.plot(my_reg.loss_history)
plt.title("Krzywa uczenia - regresja (własny perceptron)")
plt.xlabel("Epoka")
plt.ylabel("MSE (znormalizowane)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Porównanie z MLPRegressor (scikit-learn)
print()
print("Porównanie z MLPRegressor (scikit-learn)")

t0 = time.time()
sk_reg = MLPRegressor(
    hidden_layer_sizes=(10,),
    activation='relu',
    learning_rate_init=0.001,
    max_iter=1000,
    random_state=42
)
sk_reg.fit(x_reg_train_s, y_reg_train_s)
t_sk_reg = time.time() - t0

y_pred_sk_s = sk_reg.predict(x_reg_test_s)
y_pred_sk = scaler_y.inverse_transform(y_pred_sk_s.reshape(-1, 1)).ravel()
mse_sk = mean_squared_error(y_reg_test, y_pred_sk)

print(f"Własny perceptron  - MSE: {mse_my:.4f}  | czas: {t_my_reg:.3f}s")
print(f"MLPRegressor (sklearn) - MSE: {mse_sk:.4f} | czas: {t_sk_reg:.3f}s")

# wykres porównawczy
plt.figure(figsize=(8, 5))
labels = ['Własny perceptron', 'MLPRegressor']
mses = [mse_my, mse_sk]
colors = ['#4e79a7', '#e15759']
plt.bar(labels, mses, color=colors)
plt.title("4.2 Porównanie MSE — regresja")
plt.ylabel("MSE")
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# 5.1  Rozszerzenie implementacji - dowolna liczba warstw, parametryzacja

print()
print("Rozszerzenie implementacji")

configs_reg = [
    {"name": "(10,) relu",       "layers": (10,),       "act": "relu"},
    {"name": "(50,) relu",       "layers": (50,),       "act": "relu"},
    {"name": "(50,50) relu",     "layers": (50, 50),    "act": "relu"},
    {"name": "(100,50) relu",    "layers": (100, 50),   "act": "relu"},
    {"name": "(100,100,50) relu","layers": (100, 100, 50),"act": "relu"},
    {"name": "(50,) sigmoid",    "layers": (50,),       "act": "sigmoid"},
    {"name": "(50,50) sigmoid",  "layers": (50, 50),    "act": "sigmoid"},
]

results_51 = []
for cfg in configs_reg:
    t0 = time.time()
    mp = MyPerceptron(
        hidden_layer_sizes=cfg["layers"],
        activation=cfg["act"],
        learning_rate=0.001,
        max_iter=1000,
        mode='regression'
    )
    mp.fit(x_reg_train_s, y_reg_train_s)
    elapsed = time.time() - t0
    pred_s = mp.predict(x_reg_test_s)
    pred = scaler_y.inverse_transform(pred_s.reshape(-1, 1)).ravel()
    mse = mean_squared_error(y_reg_test, pred)
    results_51.append({"name": cfg["name"], "mse": mse, "time": elapsed})
    print(f"  {cfg['name']:25s}  MSE: {mse:.4f}  czas: {elapsed:.3f}s")

# wykres
plt.figure(figsize=(10, 5))
names = [r["name"] for r in results_51]
mses_51 = [r["mse"] for r in results_51]
plt.barh(names, mses_51, color='#59a14f')
plt.title("Regresja - wpływ architektury (własny perceptron)")
plt.xlabel("MSE")
plt.grid(axis='x')
plt.tight_layout()
plt.show()


# Klasyfikacja wieloklasowa - softmax
print()
print("Klasyfikacja wieloklasowa")

t0 = time.time()
my_clf = MyPerceptron(
    hidden_layer_sizes=(50,),
    activation='relu',
    learning_rate=0.01,
    max_iter=1000,
    mode='classification'
)
my_clf.fit(x_clf_train_s, y_clf_train)
t_my_clf = time.time() - t0

y_pred_my_clf = my_clf.predict(x_clf_test_s)
acc_my = accuracy_score(y_clf_test, y_pred_my_clf)
cm_my = confusion_matrix(y_clf_test, y_pred_my_clf)

print(f"Własny perceptron - Accuracy: {acc_my:.4f}  (czas: {t_my_clf:.3f}s)")
print("Macierz pomyłek:")
print(cm_my)

# krzywa uczenia
plt.figure(figsize=(8, 5))
plt.plot(my_clf.loss_history)
plt.title("Krzywa uczenia - klasyfikacja (własny perceptron)")
plt.xlabel("Epoka")
plt.ylabel("Cross-entropy loss")
plt.grid(True)
plt.tight_layout()
plt.show()


#   Porównanie z MLPClassifier (scikit-learn)
print()
print("Porównanie z MLPClassifier (scikit-learn)")

t0 = time.time()
sk_clf = MLPClassifier(
    hidden_layer_sizes=(50,),
    activation='relu',
    learning_rate_init=0.01,
    max_iter=1000,
    random_state=42
)
sk_clf.fit(x_clf_train_s, y_clf_train)
t_sk_clf = time.time() - t0

y_pred_sk_clf = sk_clf.predict(x_clf_test_s)
acc_sk = accuracy_score(y_clf_test, y_pred_sk_clf)
cm_sk = confusion_matrix(y_clf_test, y_pred_sk_clf)

print(f"Własny perceptron - Accuracy: {acc_my:.4f} | czas: {t_my_clf:.3f}s")
print(f"MLPClassifier (sklearn) - Accuracy: {acc_sk:.4f} | czas: {t_sk_clf:.3f}s")
print()
print("Macierz pomyłek (sklearn):")
print(cm_sk)

# wykres porównawczy
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# accuracy
axes[0].bar(['Własny', 'sklearn'], [acc_my, acc_sk], color=['#4e79a7', '#e15759'])
axes[0].set_title("Accuracy - klasyfikacja")
axes[0].set_ylabel("Accuracy")
axes[0].set_ylim(0, 1)
axes[0].grid(axis='y')
# czas
axes[1].bar(['Własny', 'sklearn'], [t_my_clf, t_sk_clf], color=['#4e79a7', '#e15759'])
axes[1].set_title("Czas trenowania - klasyfikacja")
axes[1].set_ylabel("Czas [s]")
axes[1].grid(axis='y')
plt.tight_layout()
plt.show()

# Wpływ parametrów na klasyfikację
print()
print("Wpływ parametrów na klasyfikację")

configs_clf = [
    {"name": "(10,) relu lr=0.01",       "layers": (10,),       "act": "relu",    "lr": 0.01},
    {"name": "(50,) relu lr=0.01",       "layers": (50,),       "act": "relu",    "lr": 0.01},
    {"name": "(50,50) relu lr=0.01",     "layers": (50, 50),    "act": "relu",    "lr": 0.01},
    {"name": "(100,50) relu lr=0.01",    "layers": (100, 50),   "act": "relu",    "lr": 0.01},
    {"name": "(50,) sigmoid lr=0.01",    "layers": (50,),       "act": "sigmoid", "lr": 0.01},
    {"name": "(50,) relu lr=0.001",      "layers": (50,),       "act": "relu",    "lr": 0.001},
    {"name": "(50,) relu lr=0.1",        "layers": (50,),       "act": "relu",    "lr": 0.1},
]

results_clf_params = []
for cfg in configs_clf:
    t0 = time.time()
    mp = MyPerceptron(
        hidden_layer_sizes=cfg["layers"],
        activation=cfg["act"],
        learning_rate=cfg["lr"],
        max_iter=1000,
        mode='classification'
    )
    mp.fit(x_clf_train_s, y_clf_train)
    elapsed = time.time() - t0
    pred = mp.predict(x_clf_test_s)
    acc = accuracy_score(y_clf_test, pred)
    results_clf_params.append({"name": cfg["name"], "acc": acc, "time": elapsed})
    print(f"  {cfg['name']:30s}  Accuracy: {acc:.4f}  czas: {elapsed:.3f}s")

# porównanie z sklearn dla tych samych konfiguracji
print()
print(" Te same konfiguracje w MLPClassifier (scikit-learn)")

act_map = {"relu": "relu", "sigmoid": "logistic"}
results_sk_params = []
for cfg in configs_clf:
    sk_act = act_map.get(cfg["act"], cfg["act"])
    t0 = time.time()
    model = MLPClassifier(
        hidden_layer_sizes=cfg["layers"],
        activation=sk_act,
        learning_rate_init=cfg["lr"],
        max_iter=1000,
        random_state=42
    )
    model.fit(x_clf_train_s, y_clf_train)
    elapsed = time.time() - t0
    pred = model.predict(x_clf_test_s)
    acc = accuracy_score(y_clf_test, pred)
    results_sk_params.append({"name": cfg["name"], "acc": acc, "time": elapsed})
    print(f"  {cfg['name']:30s}  Accuracy: {acc:.4f}  czas: {elapsed:.3f}s")

# wykres porównawczy parametrów
fig, ax = plt.subplots(figsize=(12, 6))
x_pos = np.arange(len(configs_clf))
width = 0.35
accs_my = [r["acc"] for r in results_clf_params]
accs_sk = [r["acc"] for r in results_sk_params]
ax.bar(x_pos - width / 2, accs_my, width, label='Własny', color='#4e79a7')
ax.bar(x_pos + width / 2, accs_sk, width, label='sklearn', color='#e15759')
ax.set_xticks(x_pos)
ax.set_xticklabels([c["name"] for c in configs_clf], rotation=45, ha='right', fontsize=8)
ax.set_ylabel("Accuracy")
ax.set_title("Porównanie accuracy- różne parametry")
ax.set_ylim(0, 1)
ax.legend()
ax.grid(axis='y')
plt.tight_layout()
plt.show()


# Analiza

print()

print("Analiza- wpływ architektury i porównanie czasów")

architectures = [
    (10,),
    (50,),
    (100,),
    (50, 50),
    (100, 50),
    (100, 100, 50),
]

print()
print("Regresja- wpływ architektury:")

times_my_reg_arch = []
times_sk_reg_arch = []
mses_my_arch = []
mses_sk_arch = []

for arch in architectures:
    # własny
    t0 = time.time()
    mp = MyPerceptron(hidden_layer_sizes=arch, activation='relu',
                      learning_rate=0.001, max_iter=1000, mode='regression')
    mp.fit(x_reg_train_s, y_reg_train_s)
    t_my = time.time() - t0
    pred_s = mp.predict(x_reg_test_s)
    pred = scaler_y.inverse_transform(pred_s.reshape(-1, 1)).ravel()
    mse_my_a = mean_squared_error(y_reg_test, pred)

    # sklearn
    t0 = time.time()
    skm = MLPRegressor(hidden_layer_sizes=arch, activation='relu',
                       learning_rate_init=0.001, max_iter=1000, random_state=42)
    skm.fit(x_reg_train_s, y_reg_train_s)
    t_sk = time.time() - t0
    pred_sk_s = skm.predict(x_reg_test_s)
    pred_sk2 = scaler_y.inverse_transform(pred_sk_s.reshape(-1, 1)).ravel()
    mse_sk_a = mean_squared_error(y_reg_test, pred_sk2)

    times_my_reg_arch.append(t_my)
    times_sk_reg_arch.append(t_sk)
    mses_my_arch.append(mse_my_a)
    mses_sk_arch.append(mse_sk_a)

    print(f"  {str(arch):20s}  Własny MSE: {mse_my_a:10.4f} ({t_my:.3f}s)"
          f"  |  sklearn MSE: {mse_sk_a:10.4f} ({t_sk:.3f}s)")

print()
print("Klasyfikacja- wpływ architektury:")
times_my_clf_arch = []
times_sk_clf_arch = []
accs_my_arch = []
accs_sk_arch = []

for arch in architectures:
    # własny
    t0 = time.time()
    mp = MyPerceptron(hidden_layer_sizes=arch, activation='relu',
                      learning_rate=0.01, max_iter=1000, mode='classification')
    mp.fit(x_clf_train_s, y_clf_train)
    t_my = time.time() - t0
    pred = mp.predict(x_clf_test_s)
    acc_my_a = accuracy_score(y_clf_test, pred)

    # sklearn
    t0 = time.time()
    skm = MLPClassifier(hidden_layer_sizes=arch, activation='relu',
                        learning_rate_init=0.01, max_iter=1000, random_state=42)
    skm.fit(x_clf_train_s, y_clf_train)
    t_sk = time.time() - t0
    pred_sk2 = skm.predict(x_clf_test_s)
    acc_sk_a = accuracy_score(y_clf_test, pred_sk2)

    times_my_clf_arch.append(t_my)
    times_sk_clf_arch.append(t_sk)
    accs_my_arch.append(acc_my_a)
    accs_sk_arch.append(acc_sk_a)

    print(f"  {str(arch):20s}  Własny Acc: {acc_my_a:.4f} ({t_my:.3f}s)"
          f"  |  sklearn Acc: {acc_sk_a:.4f} ({t_sk:.3f}s)")

#   wykresy podsumowujące 
arch_labels = [str(a) for a in architectures]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Regresja MSE
axes[0, 0].bar(np.arange(len(architectures)) - 0.2, mses_my_arch, 0.4,
               label='Własny', color='#4e79a7')
axes[0, 0].bar(np.arange(len(architectures)) + 0.2, mses_sk_arch, 0.4,
               label='sklearn', color='#e15759')
axes[0, 0].set_xticks(range(len(architectures)))
axes[0, 0].set_xticklabels(arch_labels, rotation=45, ha='right', fontsize=8)
axes[0, 0].set_title("Regresja - MSE vs architektura")
axes[0, 0].set_ylabel("MSE")
axes[0, 0].legend()
axes[0, 0].grid(axis='y')

# Regresja czas
axes[0, 1].bar(np.arange(len(architectures)) - 0.2, times_my_reg_arch, 0.4,
               label='Własny', color='#4e79a7')
axes[0, 1].bar(np.arange(len(architectures)) + 0.2, times_sk_reg_arch, 0.4,
               label='sklearn', color='#e15759')
axes[0, 1].set_xticks(range(len(architectures)))
axes[0, 1].set_xticklabels(arch_labels, rotation=45, ha='right', fontsize=8)
axes[0, 1].set_title("Regresja - czas trenowania vs architektura")
axes[0, 1].set_ylabel("Czas [s]")
axes[0, 1].legend()
axes[0, 1].grid(axis='y')

# Klasyfikacja accuracy
axes[1, 0].bar(np.arange(len(architectures)) - 0.2, accs_my_arch, 0.4,
               label='Własny', color='#4e79a7')
axes[1, 0].bar(np.arange(len(architectures)) + 0.2, accs_sk_arch, 0.4,
               label='sklearn', color='#e15759')
axes[1, 0].set_xticks(range(len(architectures)))
axes[1, 0].set_xticklabels(arch_labels, rotation=45, ha='right', fontsize=8)
axes[1, 0].set_title("Klasyfikacja - accuracy vs architektura")
axes[1, 0].set_ylabel("Accuracy")
axes[1, 0].set_ylim(0, 1)
axes[1, 0].legend()
axes[1, 0].grid(axis='y')

# Klasyfikacja czas
axes[1, 1].bar(np.arange(len(architectures)) - 0.2, times_my_clf_arch, 0.4,
               label='Własny', color='#4e79a7')
axes[1, 1].bar(np.arange(len(architectures)) + 0.2, times_sk_clf_arch, 0.4,
               label='sklearn', color='#e15759')
axes[1, 1].set_xticks(range(len(architectures)))
axes[1, 1].set_xticklabels(arch_labels, rotation=45, ha='right', fontsize=8)
axes[1, 1].set_title("5.4 Klasyfikacja — czas trenowania vs architektura")
axes[1, 1].set_ylabel("Czas [s]")
axes[1, 1].legend()
axes[1, 1].grid(axis='y')

plt.tight_layout()
plt.show()
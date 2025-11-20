# Deep Learning Assignment Solutions

This README contains all the questions along with their corresponding code implementations.

---

## **1. Using only NumPy, design a simple neural network to classify the Iris flowers.**

### **Code:**

```python
# NumPy-only Iris neural network implementation
# iris_numpy_nn.py
# NumPy-only 2-layer neural network for Iris classification (3 classes)
# Usage: python iris_numpy_nn.py

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

np.random.seed(42)

# Load and preprocess
iris = datasets.load_iris()
X = iris.data.astype(np.float32)  # (150,4)
y = iris.target.reshape(-1,1)     # (150,1)

# One-hot encode labels
enc = OneHotEncoder(sparse=False)
Y = enc.fit_transform(y)  # (150,3)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Network architecture
D = X_train.shape[1]
H = 16
C = Y_train.shape[1]

# Initialize weights
def glorot_init(in_dim, out_dim):
    limit = np.sqrt(6.0 / (in_dim + out_dim))
    return np.random.uniform(-limit, limit, (in_dim, out_dim)).astype(np.float32)

W1 = glorot_init(D, H)
b1 = np.zeros((1, H), dtype=np.float32)
W2 = glorot_init(H, C)
b2 = np.zeros((1, C), dtype=np.float32)

# Activations and helper functions
def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)

def relu(z):
    return np.maximum(0, z)

def relu_grad(z):
    return (z > 0).astype(np.float32)

def cross_entropy_loss(probs, targets):
    # targets: one-hot
    N = targets.shape[0]
    loss = -np.sum(targets * np.log(probs + 1e-12)) / N
    return loss

# Training loop (mini-batch SGD)
lr = 0.01
epochs = 1000
batch_size = 16
N = X_train.shape[0]

for epoch in range(1, epochs+1):
    # Shuffle
    perm = np.random.permutation(N)
    X_sh, Y_sh = X_train[perm], Y_train[perm]
    for i in range(0, N, batch_size):
        xb = X_sh[i:i+batch_size]
        yb = Y_sh[i:i+batch_size]

        # forward
        z1 = xb.dot(W1) + b1         # (b,H)
        a1 = relu(z1)                # (b,H)
        z2 = a1.dot(W2) + b2         # (b,C)
        probs = softmax(z2)          # (b,C)

        # loss
        loss = cross_entropy_loss(probs, yb)

        # backward
        Nbatch = xb.shape[0]
        dz2 = (probs - yb) / Nbatch  # (b,C)
        dW2 = a1.T.dot(dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2.dot(W2.T)          # (b,H)
        dz1 = da1 * relu_grad(z1)
        dW1 = xb.T.dot(dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # update
        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1

    if epoch % 100 == 0 or epoch==1:
        # evaluate train accuracy
        a1_train = relu(X_train.dot(W1) + b1)
        probs_train = softmax(a1_train.dot(W2) + b2)
        preds_train = np.argmax(probs_train, axis=1)
        acc_train = np.mean(preds_train == np.argmax(Y_train, axis=1))
        print(f"Epoch {epoch:4d}  Loss {loss:.4f}  Train acc {acc_train:.4f}")

# Final test accuracy
a1_test = relu(X_test.dot(W1) + b1)
probs_test = softmax(a1_test.dot(W2) + b2)
preds_test = np.argmax(probs_test, axis=1)
acc_test = np.mean(preds_test == np.argmax(Y_test, axis=1))
print("Test accuracy:", acc_test)
```

---

## **2. Develop a CNN to classify images from the CIFAR‑10 dataset.**

### **Code:**

```python
# CNN for CIFAR-10
# cifar10_cnn.py
# Train a reasonably strong CNN on CIFAR-10 using Keras.
# Usage: python cifar10_cnn.py

import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_datasets as tfds
import time

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 128
EPOCHS = 50
IMG_SIZE = (32,32,3)

# Load CIFAR-10 via Keras
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train = y_train.flatten()
y_test = y_test.flatten()

# Preprocessing + augmentation
def preprocess(x, y, training=False):
    x = tf.cast(x, tf.float32) / 255.0
    if training:
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_brightness(x, 0.1)
        x = tf.image.random_crop(x, size=[32,32,3])
    return x, y

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(50000).map(lambda x,y:preprocess(x,y,True), num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)
test_ds  = tf.data.Dataset.from_tensor_slices((x_test, y_test)).map(lambda x,y:preprocess(x,y,False), num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)

# Example architecture (Conv -> BN -> ReLU blocks)
def make_model():
    inp = layers.Input(shape=IMG_SIZE)
    x = layers.Conv2D(64,3,padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64,3,padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D()(x)

    x = layers.Conv2D(128,3,padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(128,3,padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D()(x)

    x = layers.Conv2D(256,3,padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(10, activation='softmax')(x)

    model = models.Model(inp, out)
    return model

model = make_model()
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

start = time.time()
history = model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS)
print("Training time (s):", time.time()-start)
model.save("cifar10_cnn.h5")
```

---

## **3. Train a neural network with various combinations of learning rates, batch sizes, and optimizers.**

### **Code:**

```python
# Hyperparameter sweep script
# hp_sweep_example.py
# Experiment script: loops over combinations of hyperparams, logs val accuracy & training time.
# Usage: python hp_sweep_example.py

import tensorflow as tf
from tensorflow.keras import layers, models
import itertools, time, json, os
os.makedirs("hp_results", exist_ok=True)

# Dataset: Fashion MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train[..., None] / 255.0
x_test  = x_test[..., None]  / 255.0

def make_model():
    inp = layers.Input(shape=(28,28,1))
    x = layers.Conv2D(32,3,activation='relu')(inp)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64,3,activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(10, activation='softmax')(x)
    return models.Model(inp,out)

# Hyperparameters
learning_rates = [1e-3, 1e-4]
batch_sizes = [64, 128]
optimizers = ['adam', 'sgd']

results = []
for lr, bs, opt_name in itertools.product(learning_rates, batch_sizes, optimizers):
    print(f"\n=== LR={lr} BS={bs} OPT={opt_name} ===")
    model = make_model()
    if opt_name == 'adam':
        opt = tf.keras.optimizers.Adam(lr)
    elif opt_name == 'sgd':
        opt = tf.keras.optimizers.SGD(lr)
    else:
        raise ValueError(opt_name)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    start = time.time()
    history = model.fit(x_train, y_train, epochs=5, batch_size=bs, validation_split=0.1, verbose=1)
    elapsed = time.time() - start
    val_acc = history.history['val_accuracy'][-1]
    final_train_acc = history.history['accuracy'][-1]
    print(f"Time {elapsed:.1f}s ValAcc {val_acc:.4f} TrainAcc {final_train_acc:.4f}")
    results.append({'lr':lr,'bs':bs,'opt':opt_name,'time_s':elapsed,'val_acc':float(val_acc),'train_acc':float(final_train_acc)})

# Save results as JSON
with open("hp_results/results.json","w") as f:
    json.dump(results, f, indent=2)
print("Saved to hp_results/results.json")
```

---

## **4. Evaluate and compare CNN, RNN, and MLP on a standardized dataset.**

### **Code:**

```python
# Model comparison script
# compare_architectures.py
# Compare MLP, CNN, simple RNN on MNIST. Measure train time & test accuracy.
# Usage: python compare_architectures.py

import tensorflow as tf, time, json, os
from tensorflow.keras import layers, models
os.makedirs("compare_results", exist_ok=True)

# Data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32')/255.0
x_test  = x_test.astype('float32')/255.0

# MLP: flatten input
def build_mlp():
    inp = layers.Input(shape=(28,28))
    x = layers.Flatten()(inp)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(10, activation='softmax')(x)
    return models.Model(inp,out)

# CNN
def build_cnn():
    inp = layers.Input(shape=(28,28,1))
    x = layers.Reshape((28,28,1))(inp)
    x = layers.Conv2D(32,3,activation='relu')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64,3,activation='relu')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128,activation='relu')(x)
    out = layers.Dense(10,activation='softmax')(x)
    return models.Model(inp,out)

# Simple RNN (LSTM)
def build_rnn():
    # treat each row as a timestep vector of length 28
    inp = layers.Input(shape=(28,28))
    x = layers.LSTM(128)(inp)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(10, activation='softmax')(x)
    return models.Model(inp,out)

architectures = {
    'MLP': build_mlp,
    'CNN': build_cnn,
    'RNN': build_rnn
}

results = {}
for name, builder in architectures.items():
    print(f"\n== Training {name} ==")
    model = builder()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # reshape or adapt data per model
    if name == 'CNN':
        X_train = x_train[..., None]
        X_test  = x_test[..., None]
    else:
        X_train = x_train
        X_test  = x_test
    start = time.time()
    history = model.fit(X_train, y_train, epochs=5, batch_size=128, validation_split=0.1, verbose=1)
    elapsed = time.time() - start
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"{name} - Test Acc: {test_acc:.4f}, Time: {elapsed:.1f}s")
    results[name] = {'test_acc': float(test_acc), 'train_time_s': elapsed, 'history': history.history}

# Save summary
with open("compare_results/summary.json","w") as f:
    json.dump({k:{'test_acc':v['test_acc'],'train_time_s':v['train_time_s']} for k,v in results.items()}, f, indent=2)
print("Saved compare_results/summary.json")

```

---

## **5. Design a TensorFlow/Keras neural network for a specific application with dropout and batch normalization.**

### **Code:**

```python
# Sentiment analysis model with CNN + BN + Dropout
# imdb_sentiment_model.py
# Sentiment analysis on IMDB using Keras. Demonstrates Dropout and BatchNormalization.
# Usage: python imdb_sentiment_model.py

import tensorflow as tf
from tensorflow.keras import layers, models
import time

vocab_size = 20000
maxlen = 200
embed_dim = 128

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=vocab_size)
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test  = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

inp = layers.Input(shape=(maxlen,))
x = layers.Embedding(vocab_size, embed_dim, input_length=maxlen)(inp)
x = layers.Conv1D(128, 5, padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.MaxPool1D(2)(x)
x = layers.Conv1D(128, 5, padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = layers.GlobalMaxPool1D()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.5)(x)
out = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inp,out)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
start = time.time()
history = model.fit(x_train, y_train, epochs=5, batch_size=256, validation_split=0.1)
print("Time:", time.time()-start)
print("Test eval:", model.evaluate(x_test, y_test))
model.save("imdb_sentiment.h5")
```

---

## **6. Apply model quantization & pruning and deploy to Raspberry Pi for real‑time object detection.**

### **Code:**

```python
# Pruning MobileNetV2
# prune_mobile_net.py
# Applies pruning to MobileNetV2 on a small dataset subset (here use CIFAR-10) and saves a pruned h5 model.
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras import layers, models
import time

# Load something small for quick fine-tune (CIFAR-10)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32')/255.0
x_test  = x_test.astype('float32')/255.0

# Use MobileNetV2 as base
base = tf.keras.applications.MobileNetV2(input_shape=(32,32,3), include_top=False, weights=None, pooling='avg')
x = base.output
out = layers.Dense(10, activation='softmax')(x)
model = models.Model(base.input, out)

# Apply pruning wrapper
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0, final_sparsity=0.5,
        begin_step=0, end_step=1000)
}
model_for_pruning = prune_low_magnitude(model, **pruning_params)

model_for_pruning.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]

model_for_pruning.fit(x_train, y_train, epochs=3, batch_size=128, callbacks=callbacks, validation_split=0.1)
# strip pruning wrappers for export
model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
model_for_export.save("mobilenet_pruned.h5")
print("Saved mobilenet_pruned.h5")
```

```python
# Convert pruned model to TFLite
# convert_to_tflite.py
# Convert saved TF model to a quantized TFLite model (dynamic range or full integer quant).
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("mobilenet_pruned.h5")
# TFLite dynamic range quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()
open("mobilenet_pruned_quant.tflite", "wb").write(tflite_quant_model)
print("Saved mobilenet_pruned_quant.tflite")
```

```python
# Raspberry Pi TFLite runtime script
# run_tflite_camera_pi.py
# Run TFLite model on Raspberry Pi camera or USB webcam. Requires tflite_runtime or full TF on Pi.
# Usage on Pi: python3 run_tflite_camera_pi.py --model mobilenet_pruned_quant.tflite

import argparse, time
import numpy as np
import cv2

try:
    # Try using tflite_runtime (lighter) if installed
    from tflite_runtime.interpreter import Interpreter
except Exception:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--labels', default=None)
args = ap.parse_args()

# Load tflite model
interpreter = Interpreter(model_path=args.model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# adjust preprocessing depending on model input size:
in_shape = input_details[0]['shape']
H, W = in_shape[1], in_shape[2]

while True:
    ret, frame = cap.read()
    if not ret:
        break
    img = cv2.resize(frame, (W, H))
    inp = img.astype('float32') / 255.0
    inp = np.expand_dims(inp, axis=0)

    interpreter.set_tensor(input_details[0]['index'], inp)
    start = time.time()
    interpreter.invoke()
    infer_time = (time.time() - start)*1000
    out = interpreter.get_tensor(output_details[0]['index'])
    topk = np.argsort(out[0])[-3:][::-1]
    cv2.putText(frame, f"Infer: {infer_time:.1f}ms", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    # show top-1 class (just numbers here)
    cv2.putText(frame, f"Top1: {topk[0]}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.imshow("TFLite", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## **7. Optimize a deep learning model for edge deployment and test in a simulated edge environment.**

### **Code:**

```python
# QAT example
# qat_example.py (sketch)
# This demonstrates QAT with TF MOT (you need TF >=2.4)
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras import layers, models

# Build small model (example on CIFAR-10)
def small_cnn():
    inp = layers.Input(shape=(32,32,3))
    x = layers.Conv2D(16,3,activation='relu')(inp)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(32,3,activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Dense(10, activation='softmax')(x)
    return models.Model(inp,out)

model = small_cnn()
# Apply quantize_annotate -> quantize_apply for QAT
quantize_annotate = tfmot.quantization.keras.quantize_annotate_model
quantize_apply = tfmot.quantization.keras.quantize_apply

annotated_model = quantize_annotate(model)
qat_model = quantize_apply(annotated_model)

qat_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Fine-tune on real data for a few epochs, then convert to TFLite with quantization
```

```python
# Edge device benchmark
# edge_benchmark.py
# Load TFLite model and measure average per-inference time to simulate edge device perf.
import time, numpy as np, argparse
try:
    from tflite_runtime.interpreter import Interpreter
except:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--iters', type=int, default=200)
args = ap.parse_args()

interpreter = Interpreter(model_path=args.model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
out_details = interpreter.get_output_details()

# Create dummy input
shape = input_details[0]['shape']
dummy = np.random.random_sample(shape).astype(input_details[0]['dtype'])

# Warmup
for _ in range(10):
    interpreter.set_tensor(input_details[0]['index'], dummy)
    interpreter.invoke()

# Benchmark
t0 = time.time()
for _ in range(args.iters):
    interpreter.set_tensor(input_details[0]['index'], dummy)
    interpreter.invoke()
t1 = time.time()
avg_ms = ((t1 - t0) / args.iters) * 1000.0
print(f"Avg inference time: {avg_ms:.2f} ms per run over {args.iters} iterations")
```

---

## **8. Comprehensive deep learning project with preprocessing, model building, training, evaluation, and deployment.**

### **Code:**

```python
# End-to-end DL project template
# project_template.py
# Comprehensive template for a DL project:
# - Data load & preprocessing
# - Model build
# - Train/evaluate
# - Save model + convert to TFLite
# - Basic Flask deployment stub
#
# Usage: python project_template.py train
#        python project_template.py convert
#        python project_template.py serve

import argparse, os, time, numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from flask import Flask, request, jsonify
from PIL import Image
import io

MODEL_PATH = "project_model.h5"
TFLITE_PATH = "project_model.tflite"

def load_data():
    # Replace here with your dataset (example: CIFAR10)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32')/255.0
    x_test = x_test.astype('float32')/255.0
    return x_train, y_train.flatten(), x_test, y_test.flatten()

def build_model(input_shape=(32,32,3), num_classes=10):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(32,3,activation='relu')(inp)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64,3,activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inp, out)

def train():
    x_train, y_train, x_test, y_test = load_data()
    model = build_model(input_shape=x_train.shape[1:], num_classes=len(np.unique(y_train)))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    start = time.time()
    model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.1)
    print("Train time", time.time()-start)
    print("Eval:", model.evaluate(x_test, y_test))
    model.save(MODEL_PATH)
    print("Saved", MODEL_PATH)

def convert_tflite():
    model = tf.keras.models.load_model(MODEL_PATH)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Example: full-int requires representative dataset
    def representative_gen():
        x_train, _, _, _ = load_data()
        for i in range(100):
            yield [np.expand_dims(x_train[i], axis=0).astype(np.float32)]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()
    open(TFLITE_PATH,"wb").write(tflite_model)
    print("Saved", TFLITE_PATH)

# --- Simple Flask app to serve predictions with saved Keras model
def serve():
    model = tf.keras.models.load_model(MODEL_PATH)
    app = Flask(__name__)

    @app.route('/predict', methods=['POST'])
    def predict():
        if 'file' not in request.files:
            return jsonify({'error':'no file'}), 400
        file = request.files['file'].read()
        img = Image.open(io.BytesIO(file)).convert('RGB').resize((32,32))
        arr = np.array(img).astype('float32')/255.0
        arr = np.expand_dims(arr,0)
        preds = model.predict(arr)
        top = int(np.argmax(preds[0]))
        return jsonify({'class': top, 'scores': preds[0].tolist()})

    app.run(host='0.0.0.0', port=5000)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['train','convert','serve'])
    args = ap.parse_args()
    if args.action == 'train':
        train()
    elif args.action == 'convert':
        convert_tflite()
    elif args.action == 'serve':
        serve()
```

---

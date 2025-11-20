import numpy as np
import os
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import tensorflow_datasets as tfds
import tensorflow as tf

# ---------- 1) Model Paths ----------
MNIST_MODEL_PATH = "mnist_cnn.h5"
EMNIST_MODEL_PATH = "emnist_cnn.h5"

# ---------- 2) Model Builder ----------
def build_model(input_shape=(28,28,1), num_classes=10):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax'),
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ---------- 3) MNIST Training ----------
def train_mnist_model():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test  = np.expand_dims(x_test, -1)

    model = build_model(num_classes=10)
    model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1)
    model.save(MNIST_MODEL_PATH)
    return model

# ---------- 4) EMNIST Training ----------
def train_emnist_model():
    ds_train, ds_test = tfds.load('emnist/letters', split=['train', 'test'], as_supervised=True)

    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.expand_dims(image, -1)
        label = label - 1  # EMNIST labels: 1–26 → 0–25
        return image, label

    ds_train = ds_train.map(preprocess).batch(128).prefetch(1)
    ds_test = ds_test.map(preprocess).batch(128).prefetch(1)

    model = build_model(num_classes=26)
    model.fit(ds_train, epochs=5, validation_data=ds_test)
    model.save(EMNIST_MODEL_PATH)
    return model

# ---------- 5) Load Models ----------
mnist_model = load_model(MNIST_MODEL_PATH) if os.path.exists(MNIST_MODEL_PATH) else train_mnist_model()
emnist_model = load_model(EMNIST_MODEL_PATH) if os.path.exists(EMNIST_MODEL_PATH) else train_emnist_model()

# ---------- 6) Tkinter UI ----------
class PaintApp:
    def __init__(self, mnist_model, emnist_model):
        self.mnist_model = mnist_model
        self.emnist_model = emnist_model

        self.root = tk.Tk()
        self.root.title("Digit & Letter Recognizer")

        self.canvas_width = 280
        self.canvas_height = 280
        self.canvas = tk.Canvas(self.root, width=self.canvas_width, height=self.canvas_height, bg='black')
        self.canvas.pack()

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill='x')
        tk.Button(btn_frame, text="Predict", command=self.predict).pack(side='left')
        tk.Button(btn_frame, text="Clear", command=self.clear).pack(side='left')
        tk.Button(btn_frame, text="Save Image", command=self.save_image).pack(side='left')

        mode_frame = tk.Frame(self.root)
        mode_frame.pack()
        self.mode = tk.StringVar(value="digit")
        tk.Radiobutton(mode_frame, text="Digit", variable=self.mode, value="digit").pack(side='left')
        tk.Radiobutton(mode_frame, text="Letter", variable=self.mode, value="letter").pack(side='left')

        self.label = tk.Label(self.root, text="Draw a digit or letter and click Predict", font=("Helvetica", 14))
        self.label.pack()

        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self.last_x, self.last_y = None, None

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

    def paint(self, event):
        x, y = event.x, event.y
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, x, y, fill='white', width=20, capstyle=tk.ROUND, smooth=True)
            self.draw.line([self.last_x, self.last_y, x, y], fill=255, width=20)
        self.last_x, self.last_y = x, y

    def reset(self, event):
        self.last_x, self.last_y = None, None

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, self.canvas_width, self.canvas_height], fill=0)
        self.label.config(text="Draw a digit or letter and click Predict")

    def save_image(self):
        self.image.save("last_draw.png")
        self.label.config(text="Saved to last_draw.png")

    def preprocess_image(self, pil_img):
        img = pil_img.resize((28,28), Image.Resampling.LANCZOS)
        img = ImageOps.invert(img)
        arr = np.array(img).astype("float32") / 255.0
        arr = np.expand_dims(arr, axis=(0,-1))
        return arr

    def predict(self):
        img_copy = self.image.copy()
        processed = self.preprocess_image(img_copy)

        if self.mode.get() == "digit":
            preds = self.mnist_model.predict(processed)
            class_idx = np.argmax(preds, axis=1)[0]
            confidence = float(np.max(preds))
            self.label.config(text=f"Digit: {class_idx} (confidence {confidence:.2f})")
        else:
            preds = self.emnist_model.predict(processed)
            class_idx = np.argmax(preds, axis=1)[0]
            letter = chr(ord('A') + class_idx)
            confidence = float(np.max(preds))
            self.label.config(text=f"Letter: {letter} (confidence {confidence:.2f})")

    def run(self):
        self.root.mainloop()

# ---------- 7) Run App ----------
if __name__ == "__main__":
    app = PaintApp(mnist_model, emnist_model)
    app.run()

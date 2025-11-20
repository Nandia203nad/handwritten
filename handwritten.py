# save as digit_recognizer.py
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import os

# ---------- 1) Load and train model (MNIST) ----------
MODEL_PATH = "mnist_cnn.h5"

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

def train_and_save_model():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)  # (N,28,28,1)
    x_test  = np.expand_dims(x_test, -1)

    model = build_model()
    model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1)
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")
    model.save(MODEL_PATH)
    return model

# If model file exists, load it; otherwise train (takes time)
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    print("Loaded model from", MODEL_PATH)
else:
    print("Training model (this may take a few minutes)...")
    model = train_and_save_model()

# ---------- 2) Simple Tkinter drawing UI ----------
class PaintApp:
    def __init__(self, model):
        self.model = model
        self.root = tk.Tk()
        self.root.title("Digit Recognizer - draw a digit (0-9)")
        self.canvas_width = 280
        self.canvas_height = 280

        self.canvas = tk.Canvas(self.root, width=self.canvas_width, height=self.canvas_height, bg='black')
        self.canvas.pack()

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill='x')
        predict_btn = tk.Button(btn_frame, text="Predict", command=self.predict)
        predict_btn.pack(side='left')
        clear_btn = tk.Button(btn_frame, text="Clear", command=self.clear)
        clear_btn.pack(side='left')
        save_btn = tk.Button(btn_frame, text="Save Image", command=self.save_image)
        save_btn.pack(side='left')

        # For drawing with PIL too
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self.last_x, self.last_y = None, None

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

        self.label = tk.Label(self.root, text="Draw a digit and click Predict", font=("Helvetica", 14))
        self.label.pack()

    def paint(self, event):
        x, y = event.x, event.y
        if self.last_x and self.last_y:
            # draw thicker line
            self.canvas.create_line(self.last_x, self.last_y, x, y, fill='white', width=20, capstyle=tk.ROUND, smooth=True)
            self.draw.line([self.last_x, self.last_y, x, y], fill=255, width=20)
        self.last_x, self.last_y = x, y

    def reset(self, event):
        self.last_x, self.last_y = None, None

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, self.canvas_width, self.canvas_height], fill=0)
        self.label.config(text="Draw a digit and click Predict")

    def save_image(self):
        self.image.save("last_draw.png")
        self.label.config(text="Saved to last_draw.png")

    def preprocess_image(self, pil_img):
        # Convert to 28x28, invert colors (MNIST digits are black-on-white), normalize
        img = pil_img.resize((28,28), Image.Resampling.LANCZOS
)
        img = ImageOps.invert(img)  # because we used black background & white strokemag
        arr = np.array(img).astype("float32") / 255.0
        arr = np.expand_dims(arr, axis=(0,-1))  # shape (1,28,28,1)
        return arr

    def predict(self):
        # Use a copy so we don't modify original
        img_copy = self.image.copy()
        processed = self.preprocess_image(img_copy)
        preds = self.model.predict(processed)
        class_idx = np.argmax(preds, axis=1)[0]
        confidence = float(np.max(preds))
        self.label.config(text=f"Prediction: {class_idx} (confidence {confidence:.2f})")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = PaintApp(model)
    app.run()

import numpy as np
import cv2
import os
from tensorflow import keras
from keras import layers
from keras.models import load_model
import tensorflow_datasets as tfds
import tensorflow as tf

# ---------- Загварын замууд ----------
MNIST_MODEL_PATH = "mnist_cnn.h5"
EMNIST_MODEL_PATH = "emnist_cnn.h5"

# ---------- Загвар үүсгэх ----------
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

# ---------- MNIST сургах ----------
def train_mnist_model():
    print("MNIST загварыг сургаж байна...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test  = np.expand_dims(x_test, -1)

    model = build_model(num_classes=10)
    model.fit(x_train, y_train, epochs=10, batch_size=256, validation_split=0.1)
    model.save(MNIST_MODEL_PATH)
    print("MNIST загвар хадгалагдлаа:", MNIST_MODEL_PATH)
    return model

# ---------- EMNIST сургах ----------
def preprocess_emnist(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, -1)
    image = tf.image.rot90(image, k=1)
    image = tf.image.flip_left_right(image)
    label = label - 1
    return image, label

def train_emnist_model():
    print("EMNIST загварыг сургаж байна...")
    ds_train, ds_test = tfds.load("emnist/letters", split=["train", "test"], as_supervised=True)
    ds_train = ds_train.map(preprocess_emnist).batch(256).shuffle(10000)
    ds_test  = ds_test.map(preprocess_emnist).batch(256)

    model = build_model(num_classes=26)
    model.fit(ds_train, epochs=12, validation_data=ds_test)
    model.save(EMNIST_MODEL_PATH)
    print("EMNIST загвар хадгалагдлаа:", EMNIST_MODEL_PATH)
    return model

# ---------- Зураг боловсруулах ----------
def preprocess_for_prediction(roi):
    """Таасан хэсгийг 28x28 болгох"""
    # Дилэйшн
    kernel = np.ones((2,2), np.uint8)
    roi = cv2.dilate(roi, kernel, iterations=1)
    
    # Gaussian blur + threshold
    roi = cv2.GaussianBlur(roi, (5,5), 0)
    _, roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # Хэмжээ өөрчлөх (харьцаа хадгалах)
    h, w = roi.shape
    if h == 0 or w == 0:
        return None
        
    scale = 20 / max(w, h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 28x28 төв рүү
    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = roi

    canvas = canvas.astype("float32") / 255.0
    canvas = np.expand_dims(canvas, axis=-1)
    canvas = np.expand_dims(canvas, axis=0)
    return canvas

# ---------- Таах ----------
def predict_digit(roi, model):
    """Тоо таних"""
    x = preprocess_for_prediction(roi)
    if x is None:
        return None
    pred = model.predict(x, verbose=0)[0]
    return str(np.argmax(pred))

def predict_letter(roi, model):
    """Үсэг таних"""
    x = preprocess_for_prediction(roi)
    if x is None:
        return None
    pred = model.predict(x, verbose=0)[0]
    idx = np.argmax(pred)
    return chr(idx + ord('A'))

# ---------- Камера танигч ----------
class CameraRecognizer:
    def __init__(self):
        self.digit_mode = True  # True = тоо, False = үсэг
        
        # Загварууд ачаалах
        if os.path.exists(MNIST_MODEL_PATH):
            self.mnist_model = load_model(MNIST_MODEL_PATH)
            print("MNIST загвар ачаалагдлаа")
        else:
            self.mnist_model = train_mnist_model()
            
        if os.path.exists(EMNIST_MODEL_PATH):
            self.emnist_model = load_model(EMNIST_MODEL_PATH)
            print("EMNIST загвар ачаалагдлаа")
        else:
            self.emnist_model = train_emnist_model()

    def process_frame(self, frame):
        """Камерын фрэймийг боловсруулах"""
        # Саарал болгох
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Инверт + threshold
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Контуруудыг олох
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Контуруудыг эрэмблэх (зүүнээс баруун)
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
        
        results = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Хэт жижиг контуруудыг алгасах
            if w < 20 or h < 20 or w > 200 or h > 200:
                continue
                
            # Талбайн шалгалт
            area = cv2.contourArea(cnt)
            if area < 400:
                continue
            
            # ROI таслах
            roi = thresh[y:y+h, x:x+w]
            
            # Таах
            if self.digit_mode:
                pred = predict_digit(roi, self.mnist_model)
            else:
                pred = predict_letter(roi, self.emnist_model)
            
            if pred:
                results.append((x, y, w, h, pred))
                
                # Зурах
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, pred, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 255, 0), 2)
        
        return frame, results

    def run(self):
        """Камер ажиллуулах"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Камер нээх боломжгүй!")
            return
        
        print("\n=== Камераар үсэг тоо таних ===")
        print("Д/d - Тоо танах горим")
        print("Ү/l - Үсэг танах горим")
        print("Space - Зураг дарах")
        print("Q/q - Гарах\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Зураг боловсруулах
            processed, results = self.process_frame(frame)
            
            # Горимыг харуулах
            mode_text = "Горим: ТОО" if self.digit_mode else "Горим: ҮСЭГ"
            cv2.putText(processed, mode_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # Илэрсэн текстийг харуулах
            if results:
                text = " ".join([r[4] for r in results])
                cv2.putText(processed, f"Үр дүн: {text}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            cv2.imshow('Камераар таних', processed)
            
            # Товчлуур
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('d') or key == ord('D'):
                self.digit_mode = True
                print("→ Тоо танах горим")
            elif key == ord('l') or key == ord('L') or key == ord('ү') or key == ord('Ү'):
                self.digit_mode = False
                print("→ Үсэг танах горим")
            elif key == ord(' '):
                # Зураг хадгалах
                if results:
                    text = "".join([r[4] for r in results])
                    filename = f"captured_{text}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Зураг хадгалагдлаа: {filename}")
        
        cap.release()
        cv2.destroyAllWindows()

# ---------- Ажиллуулах ----------
if __name__ == "__main__":
    recognizer = CameraRecognizer()
    recognizer.run()
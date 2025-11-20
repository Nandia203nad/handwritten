import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import os

# Алдаа гарвал шалгах
print("Програм эхэлж байна...")

# MNIST өгөгдөл ачаалах
print("MNIST өгөгдөл ачаалж байна...")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Өгөгдлийг бэлтгэх
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)  # (60000, 28, 28, 1)
x_test = np.expand_dims(x_test, -1)    # (10000, 28, 28, 1)

print(f"Сургалтын өгөгдөл: {x_train.shape}")
print(f"Тест өгөгдөл: {x_test.shape}")

# Энгийн боловч үр дүнтэй загвар
def create_model():
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(100, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ])
    return model  # *** ЭНЭ МӨРИЙГ НЭМСЭН ***

# Загварыг үүсгэх
print("Загвар үүсгэж байна...")
model = create_model()

# Загварыг эмхлэх
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Загварын бүтэц:")
model.summary()

# Загварыг сургах
print("\nЗагвар сургаж байна...")
history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=5,  # Хурдан сургахын тулд цөөхөн epoch
    validation_split=0.1,
    verbose=1
)

# Үнэлгээ хийх
print("\nЗагварыг тестлэж байна...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Тестийн нарийвчлал: {test_acc:.4f}")

# Загварыг хадгалах
model.save('mnist_model.h5')
print("\nЗагвар 'mnist_model.h5' файлд хадгалагдлаа")

# Зураг таних GUI
class DigitRecognizerApp:
    def __init__(self, root, model):
        self.root = root
        self.model = model
        self.root.title("Тоо Таних Програм")
        
        # Canvas үүсгэх
        self.canvas = tk.Canvas(root, width=280, height=280, bg='white')
        self.canvas.pack()
        
        # Товчнууд
        btn_frame = tk.Frame(root)
        btn_frame.pack()
        
        predict_btn = tk.Button(btn_frame, text="Таах", command=self.predict_digit, 
                                bg='green', fg='white', font=('Arial', 12))
        predict_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        clear_btn = tk.Button(btn_frame, text="Арилгах", command=self.clear_canvas,
                              bg='red', fg='white', font=('Arial', 12))
        clear_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Үр дүн харуулах
        self.result_label = tk.Label(root, text="Тоо зурж, 'Таах' дарна уу", 
                                     font=('Arial', 16))
        self.result_label.pack(pady=10)
        
        # Зурах үйлдэл
        self.canvas.bind('<B1-Motion>', self.paint)
        
        # PIL зураг үүсгэх
        self.image = Image.new('L', (280, 280), 'white')
        self.draw = ImageDraw.Draw(self.image)
        
    def paint(self, event):
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', width=0)
        self.draw.ellipse([x1, y1, x2, y2], fill='black')
    
    def clear_canvas(self):
        self.canvas.delete('all')
        self.image = Image.new('L', (280, 280), 'white')
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="Тоо зурж, 'Таах' дарна уу")
    
    def predict_digit(self):
        # Зургийг боловсруулах
        img = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        img = ImageOps.invert(img)
        img_array = np.array(img).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=(0, -1))
        
        # Таамаг хийх
        prediction = self.model.predict(img_array, verbose=0)
        digit = np.argmax(prediction)
        confidence = prediction[0][digit] * 100
        
        self.result_label.config(
            text=f"Таасан тоо: {digit} ({confidence:.1f}% итгэлтэй)"
        )

# GUI эхлүүлэх
print("\nGUI нээж байна...")
root = tk.Tk()
app = DigitRecognizerApp(root, model)
root.mainloop()






# from ultralytics import YOLO
# import cv2
# import time

# def main_optimized():
#     model = YOLO('yolov8n.pt')  
    
#     # Use smaller model for better performance
#     # model = YOLO('yolov8n.pt')  # nano (fastest)
#     # model = YOLO('yolov8s.pt')  # small
#     # model = YOLO('yolov8m.pt')  # medium
    
#     cap = cv2.VideoCapture(0)
    
#     if not cap.isOpened():
#         print("Error: Could not open webcam")
#         return
    
#     # Performance tracking
#     frame_count = 0
#     start_time = time.time()
#     fps = 0
    
#     print("Optimized detection running. Press 'q' to quit")
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
            
#         # Calculate FPS
#         frame_count += 1
#         if frame_count % 30 == 0:
#             end_time = time.time()
#             fps = 30 / (end_time - start_time)
#             start_time = end_time
        
#         # Perform inference with confidence threshold
#         results = model(frame, conf=0.5)  # Only show detections with >50% confidence
        
#         # Get annotated frame
#         annotated_frame = results[0].plot()
        
#         # Display performance info
#         cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#         cv2.putText(annotated_frame, f"Objects: {len(results[0])}", (10, 60), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
#         cv2.imshow("YOLOv8 Object Detection", annotated_frame)
        
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main_optimized()






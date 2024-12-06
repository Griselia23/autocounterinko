import mysql.connector
import numpy as np
from PIL import Image, ImageTk
import tflite_runtime.interpreter as tflite
from datetime import datetime
import os
import tkinter as tk
from tkinter import filedialog, Label, Button, Text
import shutil

db_host = '192.168.35.100'
db_user = 'root'
db_password = 'kiwicanggih40'
db_name = 'db_ac'

tflite_model_path = "/home/kiwi/dataset/coba/main/model1.tflite"
classes = ["ht12","unknown","z2m"]
CONFIDENCE_THRESHOLD = 0.7

interpreter = tflite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def create_uploads_directory():
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

create_uploads_directory()

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB').resize((224, 224))
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict_image(image):
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    confidence = np.max(predictions)
    label_index = np.argmax(predictions)
    label = classes[label_index]
    return label, confidence

def insert_to_db(label, confidence, status="Processed"):
    status = 1 if confidence >= CONFIDENCE_THRESHOLD else 0
    conn = mysql.connector.connect(
        host=db_host,
        user=db_user,
        password=db_password,
        database=db_name
    )
    cursor = conn.cursor()
    creation_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute(
        "INSERT INTO label (creation_date, label, confidence, status) VALUES (%s, %s, %s, %s)",
        (creation_date, label, round(float(confidence), 2), status)
    )
    conn.commit()
    cursor.close()
    conn.close()

def upload_image(file_path, result_text, image_label):

    create_uploads_directory()

    # Use the actual filename (no hash, just the original name)
    filename = os.path.basename(file_path)
    upload_path = os.path.join('uploads', filename)

    # Copy the file to the uploads directory
    shutil.copy(file_path, upload_path)

    image = preprocess_image(upload_path)
    label, confidence = predict_image(image)
    insert_to_db(label, confidence)

    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, f"Prediction: {label}\nConfidence: {round(confidence, 2):.2f}\nStatus: {'1' if confidence >= CONFIDENCE_THRESHOLD else '0'}")

    img = Image.open(upload_path)
    img.thumbnail((150, 150))
    img = ImageTk.PhotoImage(img)

    image_label.config(image=img)
    image_label.image = img

def select_file(result_text, image_label):
    file_path = filedialog.askopenfilename(
        title="Select an Image File", 
        filetypes=[("Image Files", "*.jpg"), ("JPEG Images", "*.jpeg"), ("PNG Images", "*.png")]
    )
    
    if file_path:
        upload_image(file_path, result_text, image_label)

def create_gui():
    root = tk.Tk()
    root.title("Image Upload and Prediction")

    result_text = Text(root, height=5, width=50)
    result_text.pack(pady=10)

    image_label = Label(root)
    image_label.pack(pady=10)

    upload_button = Button(root, text="Upload Image", command=lambda: select_file(result_text, image_label))
    upload_button.pack(pady=10)

    root.mainloop()

create_gui()


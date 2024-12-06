import time
import tflite_runtime.interpreter as tflite
import cv2
import numpy as np
from PIL import Image
import mysql.connector
from datetime import datetime
db_host = '192.168.35.100'
db_user = 'root'
db_password = 'kiwicanggih40'
db_name = 'db_ac'

tflite_model_path = "/home/kiwi/dataset/coba/main/model1.tflite"
classes = ["ht12", "unknown", "z2m"]
CONFIDENCE_THRESHOLD = 0.7

interpreter = tflite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image).resize((224, 224))
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
    try:
        conn = mysql.connector.connect(
            host=db_host,
            user=db_user,
            password=db_password,
            database=db_name
        )
        cursor = conn.cursor()
        creation_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        formatted_confidence = "{:.2f}".format(confidence)
        cursor.execute(
            "INSERT INTO label (creation_date, label, confidence, status) VALUES (%s, %s, %s, %s)",
            (creation_date, label, formatted_confidence, status)
        )
        conn.commit()
        cursor.close()
        conn.close()
    except mysql.connector.Error as err:
        print(f"Error: {err}")

def display_camera():
    cap = cv2.VideoCapture(0)
    
    last_capture_time = time.time()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        current_time = time.time()

        elapsed_time = current_time - last_capture_time
        if elapsed_time >= 2    :  
            last_capture_time = current_time  

            image = preprocess_image(frame)
            label, confidence = predict_image(image)
            insert_to_db(label, confidence)

            result_box = np.zeros((100, frame.shape[1], 3), dtype=np.uint8)
            cv2.putText(result_box, f"Label: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(result_box, f"Confidence: {confidence:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            combined_frame = np.vstack((frame, result_box))
            cv2.imshow('Camera Feed', combined_frame)

        # Wait for the next key press, exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    display_camera()
 

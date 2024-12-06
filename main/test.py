import cv2

# Open the camera
cap = cv2.VideoCapture(0)

# Get the FPS of the camera
fps = cap.get(cv2.CAP_PROP_FPS)

# Print the FPS value
print(f"Camera FPS: {fps}")

# Close the camera
cap.release()


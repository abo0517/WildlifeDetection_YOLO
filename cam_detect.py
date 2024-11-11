import cv2
import numpy as np
import sounddevice as sd
import random
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("")

# Use webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera.")
    exit()

# Mapping class IDs to frequency ranges (in Hz)
frequency_range_map = {
    'magpie': (950, 1050),      # 950-1050 Hz sound when magpie is detected
    'waterdeer': (1450, 1550),  # 1450-1550 Hz sound when waterdeer is detected
    'wildboar': (1950, 2050)    # 1950-2050 Hz sound when wildboar is detected
}

# Function to play a random tone within a frequency range
def play_random_tone(frequency_range, duration=0.5, sample_rate=44100):
    frequency = random.uniform(*frequency_range)  # Select a random frequency within the range
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = 0.5 * np.sin(2 * np.pi * frequency * t)  # Generate a sine wave
    sd.play(wave, sample_rate)  # Play the sound
    sd.wait()  # Wait until the sound playback is complete

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Cannot read frame.")
        break

    # Object detection using YOLO model
    results = model(frame)

    # Create a new image and draw only objects with confidence >= 0.6
    filtered_frame = frame.copy()  # Copy the original frame
    for result in results:
        for obj in result.boxes:
            confidence = obj.conf.item()  # Convert object confidence to float
            if confidence >= 0.6:
                class_id = obj.cls
                class_name = model.names[int(class_id)]  # Get class name
                print(f"Detected: {class_name} with confidence: {confidence:.2f}")

                # Play random tone within the frequency range for detected object class
                frequency_range = frequency_range_map.get(class_name)
                if frequency_range:
                    play_random_tone(frequency_range)

                # Draw bounding box and label only for objects with confidence >= 0.6
                box = obj.xyxy[0].cpu().numpy().astype(int)  # Bounding box coordinates
                cv2.rectangle(filtered_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(filtered_frame, f"{class_name} ({confidence:.2f})", 
                            (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (0, 255, 0), 2)

    # Display only objects with confidence >= 0.6
    cv2.imshow("YOLO Real-Time Detection", filtered_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
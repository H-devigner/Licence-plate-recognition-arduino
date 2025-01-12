import cv2
import requests
import serial
import time
import numpy as np

# Plate Recognizer API settings
API_URL = "https://app.platerecognizer.com/v1/plate-reader/"
API_TOKEN = "c4eede3c7f1b20de5e068122e46410fa8b42e7bd"  # Replace with your Plate Recognizer API token

# Configure serial communication with Arduino
arduino = serial.Serial('COM6', 9600, timeout=1)  # Replace 'COM6' with your Arduino's port
time.sleep(2)  # Wait for Arduino to initialize

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Predefined license plates
plates_base = ['tey31491', 'x014hb777', 'ym63777', 'c536ox174', 't837ot163']


def recognize_license_plate_with_api(frame, min_confidence=0.9, retries=3):
    """
    Sends the captured frame to the Plate Recognizer API for license plate recognition
    and retries on failure.
    """
    _, img_encoded = cv2.imencode('.jpg', frame)
    for attempt in range(retries):
        try:
            response = requests.post(
                API_URL,
                headers={'Authorization': f'Token {API_TOKEN}'},
                files={'upload': img_encoded.tobytes()},
                timeout=10  # Set a timeout for the request
            )
            if response.status_code == 201:
                data = response.json()
                if data.get("results"):
                    best_result = max(data["results"], key=lambda x: x["score"])
                    # Get the highest score candidate
                    best_candidate = max(best_result["candidates"], key=lambda x: x["score"])
                    if best_candidate["score"] >= min_confidence:
                        return best_candidate["plate"], data
                    else:
                        print(f"No predictions met the confidence threshold of {min_confidence}.")
                        return "non recognized", None
                else:
                    print("No license plate detected.")
                    return "non recognized", None
            else:
                print(f"Error: {response.status_code}, {response.text}")
                return "non recognized", None
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            break
    print("Max retries reached. Unable to process the request.")
    return "non recognized", None

while True:
    # Check for a request from Arduino
    if arduino.in_waiting > 0:
        command = arduino.readline().decode('utf-8').strip()
        if command == "START_CAMERA":
            print("Arduino requested to start the camera.")
            
            # Open the camera
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
                arduino.write(b'CLOSE')  # Send CLOSE if no frame is captured
                continue

            # Recognize the license plate using the Plate Recognizer API
            license_plate, api_response = recognize_license_plate_with_api(frame)

            print("License plate detected:", license_plate)

            # Send command to Arduino
            if license_plate != "non recognized":
                if license_plate.lower() in plates_base:
                    print("License plate recognized. Sending 'OPEN' command to Arduino.")
                    arduino.write(b'OPEN')  # Send 'OPEN' command
                    time.sleep(5)  # Wait for 5 seconds
                    command = arduino.readline().decode('utf-8').strip()
                    # Wait for the 'CONTINUE_CAMERA' command from Arduino indefinitely
                    if command != "CONTINUE_CAMERA":
                        if arduino.in_waiting > 0:
                            command = arduino.readline().decode('utf-8').strip()
                    else: continue
                    print("Arduino requested to continue the camera.")
                else:
                    print("License plate not authorized. Sending 'CLOSE' command to Arduino.")
                    arduino.write(b'CLOSE')  # Send 'CLOSE' command
            else:
                print("License plate not recognized. Sending 'CLOSE' command to Arduino.")
                arduino.write(b'CLOSE')  # Send 'CLOSE' command

            # Show the frame with detected plates (optional)
            cv2.imshow('License Plate Detection', frame)

            # Close the camera if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

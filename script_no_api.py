import cv2
import numpy as np
import pytesseract
from difflib import SequenceMatcher
import serial
import time

# Configure tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Predefined license plates
plates_base = ['tey31491', 'x014hb777', 'ym63777', 'c536ox174', '00000000', '77143dal6']

# Load Haar Cascade classifier for license plates
license_plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")

# Configure serial communication with Arduino
arduino = serial.Serial('COM6', 9600, timeout=1)  # Replace 'COM6' with your Arduino's port

def recognize_license_plate(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    license_plate = pytesseract.image_to_string(gray, config='--psm 8')
    return license_plate.strip()

def find_most_similar_plate(license_plate, plates_base, tolerance=2):
    most_similar_plate = "non recognized"
    highest_similarity = 0

    for plate in plates_base:
        similarity = SequenceMatcher(None, license_plate.lower(), plate).ratio()
        if similarity > highest_similarity:
            highest_similarity = similarity
            most_similar_plate = plate

    # Check if the similarity is within the tolerance
    if len(license_plate) > 0 and highest_similarity >= (1 - tolerance / len(license_plate)):
        return most_similar_plate
    else:
        return "non recognized"

while True:
    # Check for a request from Arduino
    if arduino.in_waiting > 0:
        command = arduino.readline().decode('utf-8').strip()
        if command == "START_CAMERA":
            print("Arduino requested to start the camera.")
            
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
                arduino.write(b'CLOSE')  # Send CLOSE if no frame is captured
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect plates using Haar Cascade
            plates = license_plate_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in plates:
                # Draw a rectangle around the detected plate
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Cut only the rectangle with the plate
                only_plate = frame[y:y+h, x:x+w]
                
                # Recognize the license plate
                license_plate = recognize_license_plate(only_plate)
                
                # Find the most similar license plate with a tolerance of 3 characters
                most_similar_plate = find_most_similar_plate(license_plate, plates_base, tolerance=3)
                
                print("Most similar license plate:", most_similar_plate)

                # Send command to Arduino
                if most_similar_plate != "non recognized":
                    print("License plate recognized. Sending 'OPEN' command to Arduino.")
                    arduino.write(b'OPEN')  # Send 'OPEN' command
                    time.sleep(5)  # Wait for 5 seconds
                else:
                    print("License plate not recognized. Sending 'CLOSE' command to Arduino.")
                    arduino.write(b'CLOSE')  # Send 'CLOSE' command

            # Show the frame with detected plates
            cv2.imshow('License Plate Detection', frame)

            # Close the camera if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()





import cv2
import requests
import serial
import time
import json

# Plate Recognizer API settings
API_URL = "https://app.platerecognizer.com/v1/plate-reader/"
API_TOKEN = "c4eede3c7f1b20de5e068122e46410fa8b42e7bd"  # Replace with your Plate Recognizer API token

# Configure serial communication with Arduino
arduino = serial.Serial('COM6', 9600, timeout=1)  # Replace 'COM6' with your Arduino's port
time.sleep(2)  # Wait for Arduino to initialize

# Initialize the video capture as None
cap = None

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
            cap = cv2.VideoCapture(0)
            frame_buffer = []  # Buffer to store frames

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame.")
                    arduino.write(b'CLOSE')  # Send CLOSE if no frame is captured
                    break

                license_plate, api_response = recognize_license_plate_with_api(frame)

                print("License plate detected:", license_plate)

                # Send command to Arduino
                if license_plate != "non recognized":
                    print("License plate recognized. Sending 'OPEN' command to Arduino.")
                    arduino.write(b'OPEN')  # Send 'OPEN' command
                    time.sleep(6)  # Wait for 6 seconds
                    break
                else:
                    print("License plate not recognized. Sending 'CLOSE' command to Arduino.")
                    arduino.write(b'CLOSE')  # Send 'CLOSE' command

                # Respect the API rate limit
                time.sleep(1)  # Wait for 1 second before the next API call

                # Show the frame with detected plates (optional)
                if cap is not None:
                    cv2.imshow('License Plate Detection', frame)

                # Close the camera if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break

                # Exit the inner loop if the camera is released
                if cap is None:
                    break

        # Exit the program if the "STOP" command is received
        elif command == "STOP":
            print("Arduino requested to stop the program.")
            break

# Clean up
cv2.destroyAllWindows()

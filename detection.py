import cv2
import numpy as np
import requests
from difflib import SequenceMatcher
import time
import serial
import pytesseract
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# Change this flag to switch between API and local recognition
USE_API_RECOGNITION = True  # Set to True to use API, False to use local recognition


@dataclass
class PlateInfo:
    id: int
    plate_number: str
    owner_name: str
    cin: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    notes: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict) -> 'PlateInfo':
        """Create PlateInfo instance from dictionary, handling datetime strings"""
        data_copy = data.copy()
        if 'created_at' in data_copy:
            data_copy['created_at'] = datetime.fromisoformat(data_copy['created_at'].replace('Z', '+00:00'))
        if 'updated_at' in data_copy:
            data_copy['updated_at'] = datetime.fromisoformat(data_copy['updated_at'].replace('Z', '+00:00'))
        return cls(**data_copy)

class LicensePlateRecognizer:
    def __init__(self):
        # Configuration
        self.TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        self.API_URL = "https://app.platerecognizer.com/v1/plate-reader/"
        self.API_TOKEN = "c4eede3c7f1b20de5e068122e46410fa8b42e7bd"
        self.BACKEND_API_URL = "http://localhost:8000"
        self.ARDUINO_PORT = 'COM6'
        self.ARDUINO_BAUDRATE = 9600
        
        # Recognition settings
        self.MIN_CONFIDENCE = 0.8  # Minimum confidence for API recognition
        self.API_RETRIES = 3      # Number of retries for API calls
        
        # Camera settings
        self.CAMERA_WIDTH = 1280
        self.CAMERA_HEIGHT = 720
        self.CAMERA_INDEX = 0
        
        # Initialize components
        self._init_tesseract()
        self._init_arduino()
        self._init_camera()
        
    def _init_tesseract(self) -> None:
        """Initialize Tesseract OCR"""
        try:
            pytesseract.pytesseract.tesseract_cmd = self.TESSERACT_PATH
            pytesseract.get_tesseract_version()
            print("Tesseract initialized successfully")
        except Exception as e:
            print(f"Failed to initialize Tesseract: {e}")
            raise
            
    def _init_arduino(self) -> None:
        """Initialize Arduino communication"""
        try:
            self.arduino = serial.Serial(
                self.ARDUINO_PORT,
                self.ARDUINO_BAUDRATE,
                timeout=1
            )
            time.sleep(2)  # Wait for Arduino initialization
            print("Arduino connection established")
        except serial.SerialException as e:
            print(f"Failed to initialize Arduino: {e}")
            raise
            
    def _init_camera(self) -> None:
        """Initialize camera capture"""
        try:
            self.cap = cv2.VideoCapture(self.CAMERA_INDEX)
            if not self.cap.isOpened():
                raise RuntimeError("Cannot access camera")
                
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CAMERA_HEIGHT)
            print("Camera initialized successfully")
        except Exception as e:
            print(f"Camera initialization failed: {e}")
            raise

    def draw_info_box(self, frame: np.ndarray, text_lines: List[str], 
                     start_y: int = 60, color: Tuple[int, int, int] = (255, 255, 255), 
                     bg_color: Tuple[int, int, int] = (0, 0, 0)) -> None:
        """Draw a semi-transparent information box on the frame"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        padding = 10
        line_spacing = 30
        
        # Calculate box dimensions
        max_width = max(cv2.getTextSize(text, font, font_scale, thickness)[0][0] 
                       for text in text_lines)
        box_width = max_width + 2 * padding
        box_height = len(text_lines) * line_spacing + 2 * padding
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, start_y), (10 + box_width, start_y + box_height), 
                     bg_color, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw text
        for i, text in enumerate(text_lines):
            y_position = start_y + padding + i * line_spacing
            cv2.putText(frame, text, (10 + padding, y_position), 
                       font, font_scale, color, thickness)

    def get_plates_from_db(self) -> List[PlateInfo]:
        """Fetch license plates from database"""
        try:
            response = requests.get(
                f"{self.BACKEND_API_URL}/plates/",
                timeout=5
            )
            response.raise_for_status()
            return [PlateInfo.from_dict(plate) for plate in response.json()]
        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch plates from database: {e}")
            return []
        except (KeyError, ValueError) as e:
            print(f"Failed to parse database response: {e}")
            return []

    def find_similar_plate(self, license_plate: str, plates_data: List[PlateInfo], 
                         tolerance: int = 2) -> Optional[PlateInfo]:
        """Find most similar plate from database"""
        if not license_plate or not plates_data:
            return None

        license_plate = license_plate.lower()
        best_match = None
        highest_ratio = 0

        for plate in plates_data:
            ratio = SequenceMatcher(None, license_plate, 
                                  plate.plate_number.lower()).ratio()
            if ratio > highest_ratio:
                highest_ratio = ratio
                best_match = plate

        if best_match and len(license_plate) > 0:
            if highest_ratio >= (1 - tolerance / len(license_plate)):
                print(f"Found similar plate: {best_match.plate_number} "
                      f"(similarity: {highest_ratio:.2f})")
                return best_match
                
        return None

    def recognize_plate_api(self, frame: np.ndarray) -> str:
        """Recognize license plate using Plate Recognizer API"""
        _, img_encoded = cv2.imencode('.jpg', frame)
        
        for attempt in range(self.API_RETRIES):
            try:
                response = requests.post(
                    self.API_URL,
                    headers={'Authorization': f'Token {self.API_TOKEN}'},
                    files={'upload': img_encoded.tobytes()},
                    timeout=10
                )
                response.raise_for_status()
                data = response.json()
                
                if results := data.get("results"):
                    best_result = max(results, key=lambda x: x["score"])
                    best_candidate = max(best_result["candidates"], key=lambda x: x["score"])
                    
                    if best_candidate["score"] >= self.MIN_CONFIDENCE:
                        print(f"API detected plate: {best_candidate['plate']}")
                        return best_candidate["plate"]
                        
                print("No valid license plate detected by API")
                return "non recognized"
                
            except requests.exceptions.RequestException as e:
                print(f"API request failed (attempt {attempt + 1}/{self.API_RETRIES}): {e}")
                if attempt < self.API_RETRIES - 1:
                    time.sleep(1)
                    continue
                break
                
        return "non recognized"

    def recognize_plate_locally(self, frame: np.ndarray) -> str:
        """Recognize license plate using local OpenCV and Tesseract"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            plate_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
            )
            
            # Apply image preprocessing
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            gray = cv2.equalizeHist(gray)
            
            plates = plate_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(100, 50)
            )
            
            if len(plates) > 0:
                for (x, y, w, h) in plates:
                    # Draw rectangle around plate
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Extract and preprocess plate image
                    plate_image = frame[y:y+h, x:x+w]
                    plate_gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
                    _, plate_binary = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    # OCR on preprocessed image
                    extracted_text = pytesseract.image_to_string(
                        plate_binary,
                        config='--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                    ).strip()
                    
                    if extracted_text:
                        print(f"Locally detected plate: {extracted_text}")
                        return extracted_text
                        
            return "non recognized"
            
        except Exception as e:
            print(f"Local recognition failed: {e}")
            return "non recognized"

    def recognize_plate(self, frame: np.ndarray) -> str:
        """Recognize license plate using the selected method"""
        if USE_API_RECOGNITION:
            return self.recognize_plate_api(frame)
        else:
            return self.recognize_plate_locally(frame)

    def process_frame(self, frame: np.ndarray) -> None:
        """Process a single frame"""
        plates_data = self.get_plates_from_db()
        
        # Use the selected recognition method
        license_plate = self.recognize_plate(frame)
        
        method = "API" if USE_API_RECOGNITION else "Local"
        print(f"Using {method} recognition method")
        
        if license_plate != "non recognized":
            plate_info = self.find_similar_plate(license_plate, plates_data)
            
            if plate_info:
                info_lines = [
                    f"Method: {method}",
                    f"Plate: {plate_info.plate_number}",
                    f"Owner: {plate_info.owner_name}",
                    f"CIN: {plate_info.cin}"
                ]
                if plate_info.notes:
                    info_lines.append(f"Notes: {plate_info.notes}")
                
                cv2.putText(frame, "MATCH FOUND", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                self.draw_info_box(frame, info_lines, color=(255, 255, 255), 
                                 bg_color=(0, 100, 0))
                
                print(f"Match found: {plate_info.plate_number}")
                self.control_gate('OPEN')
                time.sleep(5)
            else:
                cv2.putText(frame, f"NO MATCH: {license_plate}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                print(f"Plate not authorized: {license_plate}")
                self.control_gate('CLOSE')
        else:
            print("No plate recognized")
            self.control_gate('CLOSE')


    def control_gate(self, command: str) -> None:
        """Send command to Arduino to control the gate"""
        try:
            self.arduino.write(command.encode())
            print(f"Sent command to gate: {command}")
        except serial.SerialException as e:
            print(f"Failed to send command to gate: {e}")


    def run(self) -> None:
        """Main loop"""
        print("Starting License Plate Recognition System")
        
        try:
            while True:
                if self.arduino.in_waiting > 0:
                    command = self.arduino.readline().decode('utf-8').strip()
                    
                    if command == "START_CAMERA":
                        print("Processing camera frame")
                        ret, frame = self.cap.read()
                        
                        if not ret:
                            print("Failed to capture frame")
                            self.control_gate('CLOSE')
                            continue
                            
                        self.process_frame(frame)
                    else:
                        print(f"Camera stopped. Arduino sent: {command}")
                
                ret, frame = self.cap.read()
                if ret:
                    cv2.imshow("License Plate Recognition", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exiting...")
                    break
                    
        except KeyboardInterrupt:
            print("Shutting down...")
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Clean up resources"""
        try:
            self.cap.release()
            self.arduino.close()
            cv2.destroyAllWindows()
            print("Cleanup completed")
        except Exception as e:
            print(f"Error during cleanup: {e}")


if __name__ == "__main__":
    try:
        print(f"Starting License Plate Recognition System using {'API' if USE_API_RECOGNITION else 'Local'} recognition")
        recognizer = LicensePlateRecognizer()
        recognizer.run()
    except Exception as e:
        print(f"Application failed: {e}")
        raise
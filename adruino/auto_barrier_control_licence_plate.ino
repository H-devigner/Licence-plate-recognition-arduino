#include <Servo.h>
#include <Ultrasonic.h>

Ultrasonic ultrasonic(7, 8); // Trig pin 7, Echo pin 8
Servo barrier;

void setup() {
    Serial.begin(9600);
    barrier.attach(9);  // Attach servo to pin 9
    barrier.write(0);   // Initialize servo to closed position (0 degrees)
}

void loop() {
    long distance = ultrasonic.read();  // Read the distance from the ultrasonic sensor
    
    if (distance > 0 && distance <= 10) {  // If an object is within 10 cm
        Serial.println("START_CAMERA");  // Send command to Python to start the camera
        
        // Wait for a response from Python
        if (Serial.available()) {
            String command = Serial.readString();  // Read the command from Python
            command.trim();  // Remove any trailing newline characters
            
            if (command == "OPEN") {
                Serial.println("Opening barrier...");
                barrier.write(90);  // Open the barrier (90 degrees)
                delay(5000);         // Keep the barrier open for 5 seconds
                barrier.write(0);    // Close the barrier (0 degrees)
                Serial.println("CONTINUE_CAMERA");  // Send command to Python to continue the camera
            } 
            else if (command == "CLOSE") {
                Serial.println("Closing barrier...");
                barrier.write(0);  // Ensure the barrier stays closed
            }
        }
    }

    delay(100);  // Short delay to avoid rapid triggering of the sensor
}

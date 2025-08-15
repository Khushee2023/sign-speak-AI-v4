import cv2
import mediapipe as mp
import numpy as np
import math
import time
import threading
import azure.cognitiveservices.speech as speechsdk

class GestureToSpeechRecognizer:
    def __init__(self, speech_key=None, service_region=None):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Azure TTS Configuration
        self.speech_key = speech_key
        self.service_region = service_region
        self.speech_enabled = True
        
        # Initialize Azure Speech SDK
        if self.speech_key and self.service_region:
            self.setup_speech_config()
        else:
            print("⚠️  No Azure credentials provided. Speech will be disabled.")
            self.speech_enabled = False
        
        # Gesture to text mapping
        self.gesture_to_text = {
            "Hello": "Hello there!",
            "Yes": "Yes",
            "No": "No",
            "Thank You": "Thank you very much",
            "Stop": "Stop",
            "Unknown": "",  # Don't speak unknown gestures
            "No Hand": ""   # Don't speak when no hand
        }
        
        # Speech control variables
        self.last_spoken_text = ""
        self.last_speech_time = 0
        self.speech_cooldown = 2.0  # Wait 2 seconds before speaking same text again
        self.is_speaking = False
        
        # Gesture stability variables
        self.current_gesture = "No Hand"
        self.gesture_start_time = time.time()
        self.gesture_hold_threshold = 1.0  # Hold gesture for 1 second to trigger speech
        self.last_confirmed_gesture = "No Hand"
        
    def setup_speech_config(self):
        """Setup Azure Speech SDK configuration"""
        try:
            self.speech_config = speechsdk.SpeechConfig(
                subscription=self.speech_key,
                region=self.service_region
            )
            # Choose a neural voice (you can change this)
            self.speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"
            
            # Create synthesizer
            self.synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config)
            print("✅ Azure TTS initialized successfully!")
            
        except Exception as e:
            print(f"❌ Failed to initialize Azure TTS: {e}")
            self.speech_enabled = False
    
    def speak_text(self, text):
        """Convert text to speech using Azure TTS"""
        if not self.speech_enabled or not text.strip():
            return
            
        current_time = time.time()
        
        # Avoid speaking the same text repeatedly
        if (text == self.last_spoken_text and 
            current_time - self.last_speech_time < self.speech_cooldown):
            return
            
        # Don't start new speech if already speaking
        if self.is_speaking:
            return
        
        def speak_async():
            try:
                self.is_speaking = True
                print(f"🔊 Speaking: '{text}'")
                
                # Use speak_text_async for non-blocking speech
                result = self.synthesizer.speak_text_async(text).get()
                
                if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                    self.last_spoken_text = text
                    self.last_speech_time = time.time()
                    print(f"✅ Speech completed: '{text}'")
                else:
                    print(f"❌ Speech synthesis failed: {result.reason}")
                    
            except Exception as e:
                print(f"❌ Speech error: {e}")
            finally:
                self.is_speaking = False
        
        # Run speech in separate thread to avoid blocking
        speech_thread = threading.Thread(target=speak_async)
        speech_thread.daemon = True
        speech_thread.start()
    
    def is_finger_extended(self, landmarks, finger_tip, finger_pip, finger_mcp):
        """Check if a finger is extended by comparing tip position to pip and mcp"""
        tip_y = landmarks[finger_tip][1]
        pip_y = landmarks[finger_pip][1]
        mcp_y = landmarks[finger_mcp][1]
        
        # For thumb, check x-coordinate instead of y
        if finger_tip == 4:  # Thumb
            tip_x = landmarks[finger_tip][0]
            pip_x = landmarks[finger_pip][0]
            return abs(tip_x - pip_x) > 0.04
        
        # For other fingers, tip should be above pip and mcp
        return tip_y < pip_y and tip_y < mcp_y
    
    def get_finger_states(self, landmarks):
        """Get the state (extended/folded) of all 5 fingers"""
        fingers = [
            [4, 3, 2],   # Thumb
            [8, 6, 5],   # Index
            [12, 10, 9], # Middle  
            [16, 14, 13], # Ring
            [20, 18, 17]  # Pinky
        ]
        
        finger_states = []
        for finger in fingers:
            is_extended = self.is_finger_extended(landmarks, finger[0], finger[1], finger[2])
            finger_states.append(is_extended)
        
        return finger_states
    
    def recognize_gesture(self, landmarks):
        """Recognize gesture based on hand landmarks"""
        if not landmarks:
            return "No Hand"
        
        # Convert landmarks to list of [x, y] coordinates
        landmark_coords = [[lm.x, lm.y] for lm in landmarks]
        
        # Get finger states (True = extended, False = folded)
        finger_states = self.get_finger_states(landmark_coords)
        thumb, index, middle, ring, pinky = finger_states
        
        # Count extended fingers
        extended_count = sum(finger_states)
        
        # GESTURE RECOGNITION RULES
        if all(finger_states):
            return "Hello"
        elif extended_count == 0:
            return "Yes"
        elif not thumb and index and middle and ring and pinky:
            return "Stop"
        elif thumb and not any([index, middle, ring, pinky]):
            return "Thank You"
        elif not thumb and index and not any([middle, ring, pinky]):
            return "No"
        else:
            return "Unknown"
    
    def update_stable_gesture(self, detected_gesture):
        """Update gesture with stability checking and speech triggering"""
        current_time = time.time()
        
        if detected_gesture == self.current_gesture:
            # Same gesture detected, check if held long enough
            if current_time - self.gesture_start_time >= self.gesture_hold_threshold:
                if detected_gesture != self.last_confirmed_gesture:
                    # New stable gesture confirmed - trigger speech
                    self.last_confirmed_gesture = detected_gesture
                    text_to_speak = self.gesture_to_text.get(detected_gesture, "")
                    if text_to_speak:
                        self.speak_text(text_to_speak)
        else:
            # New gesture detected, reset timer
            self.current_gesture = detected_gesture
            self.gesture_start_time = current_time
        
        return self.last_confirmed_gesture
    
    def draw_enhanced_overlay(self, frame, gesture, text, progress=0):
        """Draw enhanced text overlay with speech status"""
        h, w = frame.shape[:2]
        
        # Main info panel
        cv2.rectangle(frame, (10, 10), (w-10, 160), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (w-10, 160), (255, 255, 255), 2)
        
        # Gesture name
        cv2.putText(frame, f"Gesture: {gesture}", (20, 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        # Text to speak
        cv2.putText(frame, f"Text: {text}", (20, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        # Speech status
        speech_status = "🔊 Speaking..." if self.is_speaking else "🔇 Ready to speak"
        if not self.speech_enabled:
            speech_status = "❌ Speech disabled"
        cv2.putText(frame, speech_status, (20, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 255), 2)
        
        # Progress bar for gesture hold
        if progress > 0:
            bar_width = int((w - 40) * progress)
            cv2.rectangle(frame, (20, 135), (20 + bar_width, 145), (0, 255, 0), -1)
            cv2.rectangle(frame, (20, 135), (w - 20, 145), (255, 255, 255), 1)
        
        # Controls at bottom
        controls = "Hold gesture 1s to speak | 'q' to quit | 's' to toggle speech"
        cv2.putText(frame, controls, (20, h-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    def toggle_speech(self):
        """Toggle speech on/off"""
        if self.speech_key and self.service_region:
            self.speech_enabled = not self.speech_enabled
            status = "enabled" if self.speech_enabled else "disabled"
            print(f"🔊 Speech {status}")
        else:
            print("❌ No Azure credentials - speech unavailable")
    
    def run(self):
        """Main function to run the gesture-to-speech recognition"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("🚀 WEEK 4: Gesture-to-Speech Recognition Started!")
        print("=" * 60)
        print("Available Gestures & Speech:")
        for gesture, text in self.gesture_to_text.items():
            if text and gesture not in ["Unknown", "No Hand"]:
                print(f"  {gesture} → '{text}'")
        print("=" * 60)
        if self.speech_enabled:
            print("🔊 Azure TTS is ACTIVE - gestures will be spoken!")
        else:
            print("🔇 Speech is DISABLED - check Azure credentials")
        print("Controls: 's' = toggle speech, 'q' = quit")
        print("Hold gestures steady for 1 second to trigger speech!\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = self.hands.process(rgb_frame)
            
            detected_gesture = "No Hand"
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Recognize gesture
                    detected_gesture = self.recognize_gesture(hand_landmarks.landmark)
            
            # Update stable gesture and handle speech
            stable_gesture = self.update_stable_gesture(detected_gesture)
            gesture_text = self.gesture_to_text.get(stable_gesture, "")
            
            # Calculate progress for gesture hold
            current_time = time.time()
            if detected_gesture == self.current_gesture:
                progress = min(1.0, (current_time - self.gesture_start_time) / self.gesture_hold_threshold)
            else:
                progress = 0
            
            # Draw enhanced overlay
            self.draw_enhanced_overlay(frame, stable_gesture, gesture_text, progress)
            
            # Show frame
            cv2.imshow('Week 4: Gesture-to-Speech Recognition', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.toggle_speech()
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("\n👋 Gesture-to-Speech recognition stopped!")

# Configuration function
def get_azure_credentials():
    """Get Azure credentials from user input or environment"""
    import os
    
    # Try to get from environment variables first
    speech_key = os.getenv('AZURE_SPEECH_KEY')
    service_region = os.getenv('AZURE_SPEECH_REGION')
    
    if not speech_key or not service_region:
        print("🔑 Azure Speech Service Setup")
        print("=" * 40)
        print("Enter your Azure Speech Service credentials:")
        print("(You can also set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION environment variables)")
        print()
        
        speech_key = input("Enter your Azure Speech Key: ").strip()
        service_region = input("Enter your Azure Region (e.g., eastus): ").strip()
        
        if not speech_key or not service_region:
            print("⚠️  No credentials provided. Running without speech.")
            return None, None
    
    return speech_key, service_region

# Run the gesture-to-speech recognizer
if __name__ == "__main__":
    # Get Azure credentials
    speech_key, service_region = get_azure_credentials()
    
    # Create and run recognizer
    recognizer = GestureToSpeechRecognizer(speech_key, service_region)
    recognizer.run()
from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import json
import threading
import time
import azure.cognitiveservices.speech as speechsdk
import os

app = Flask(__name__)

class WebGestureToSpeechRecognizer:
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
            "Unknown": "",
            "No Hand": ""
        }
        
        # Current state
        self.current_gesture = "No Hand"
        self.current_text = ""
        self.is_speaking = False
        self.last_spoken_text = ""
        self.last_speech_time = 0
        self.speech_cooldown = 2.0
        
        # Thread safety
        self.state_lock = threading.Lock()
        
    def setup_speech_config(self):
        """Setup Azure Speech SDK configuration"""
        try:
            self.speech_config = speechsdk.SpeechConfig(
                subscription=self.speech_key,
                region=self.service_region
            )
            self.speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"
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
            
        if self.is_speaking:
            return
        
        def speak_async():
            try:
                with self.state_lock:
                    self.is_speaking = True
                
                print(f"🔊 Speaking: '{text}'")
                result = self.synthesizer.speak_text_async(text).get()
                
                if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                    with self.state_lock:
                        self.last_spoken_text = text
                        self.last_speech_time = time.time()
                    print(f"✅ Speech completed: '{text}'")
                else:
                    print(f"❌ Speech synthesis failed: {result.reason}")
                    
            except Exception as e:
                print(f"❌ Speech error: {e}")
            finally:
                with self.state_lock:
                    self.is_speaking = False
        
        speech_thread = threading.Thread(target=speak_async)
        speech_thread.daemon = True
        speech_thread.start()
    
    def is_finger_extended(self, landmarks, finger_tip, finger_pip, finger_mcp):
        """Check if a finger is extended"""
        tip_y = landmarks[finger_tip][1]
        pip_y = landmarks[finger_pip][1]
        mcp_y = landmarks[finger_mcp][1]
        
        if finger_tip == 4:  # Thumb
            tip_x = landmarks[finger_tip][0]
            pip_x = landmarks[finger_pip][0]
            return abs(tip_x - pip_x) > 0.04
        
        return tip_y < pip_y and tip_y < mcp_y
    
    def get_finger_states(self, landmarks):
        """Get the state of all 5 fingers"""
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
        
        landmark_coords = [[lm.x, lm.y] for lm in landmarks]
        finger_states = self.get_finger_states(landmark_coords)
        thumb, index, middle, ring, pinky = finger_states
        extended_count = sum(finger_states)
        
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
    
    def generate_frames(self):
        """Generate video frames with gesture detection"""
        cap = cv2.VideoCapture(0)
        last_gesture_time = {}
        gesture_hold_threshold = 1.5  # 1.5 seconds for web version
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            gesture = "No Hand"
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    gesture = self.recognize_gesture(hand_landmarks.landmark)
            
            # Handle gesture timing and speech
            current_time = time.time()
            if gesture not in last_gesture_time:
                last_gesture_time[gesture] = current_time
            
            # Check if gesture held long enough
            if current_time - last_gesture_time[gesture] >= gesture_hold_threshold:
                if gesture != self.current_gesture:
                    with self.state_lock:
                        self.current_gesture = gesture
                        self.current_text = self.gesture_to_text.get(gesture, "")
                    
                    # Trigger speech for new gesture
                    if self.current_text:
                        self.speak_text(self.current_text)
                
                # Reset other gesture timers
                for g in list(last_gesture_time.keys()):
                    if g != gesture:
                        last_gesture_time[g] = current_time
            else:
                # Reset timer for other gestures
                for g in list(last_gesture_time.keys()):
                    if g != gesture:
                        last_gesture_time[g] = current_time
            
            # Add text overlay
            speech_status = "🔊 Speaking..." if self.is_speaking else "🔇 Ready"
            if not self.speech_enabled:
                speech_status = "❌ Speech Off"
                
            cv2.putText(frame, f"Gesture: {gesture}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Text: {self.current_text}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, speech_status, (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 255), 2)
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        cap.release()
    
    def toggle_speech(self):
        """Toggle speech on/off"""
        if self.speech_key and self.service_region:
            with self.state_lock:
                self.speech_enabled = not self.speech_enabled
            return self.speech_enabled
        return False
    
    def get_status(self):
        """Get current status"""
        with self.state_lock:
            return {
                'gesture': self.current_gesture,
                'text': self.current_text,
                'is_speaking': self.is_speaking,
                'speech_enabled': self.speech_enabled,
                'last_spoken': self.last_spoken_text,
                'timestamp': time.time()
            }

# Initialize the recognizer
def get_azure_credentials():
    """Get Azure credentials"""
    speech_key = os.getenv('AZURE_SPEECH_KEY')
    service_region = os.getenv('AZURE_SPEECH_REGION')
    
    if not speech_key or not service_region:
        print("⚠️  Set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION environment variables")
        print("   Or the web app will run without speech functionality")
    
    return speech_key, service_region

speech_key, service_region = get_azure_credentials()
recognizer = WebGestureToSpeechRecognizer(speech_key, service_region)

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(recognizer.generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_status')
def get_status():
    """API endpoint to get current status"""
    return jsonify(recognizer.get_status())

@app.route('/toggle_speech', methods=['POST'])
def toggle_speech():
    """API endpoint to toggle speech"""
    enabled = recognizer.toggle_speech()
    return jsonify({'speech_enabled': enabled})

@app.route('/speak_text', methods=['POST'])
def speak_text():
    """API endpoint to speak custom text"""
    data = request.get_json()
    text = data.get('text', '')
    if text:
        recognizer.speak_text(text)
        return jsonify({'success': True, 'message': f'Speaking: {text}'})
    return jsonify({'success': False, 'message': 'No text provided'})

if __name__ == '__main__':
    print("🌐 Starting Week 4 Flask Web Interface...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("🔑 Make sure to set your Azure credentials as environment variables:")
    print("   export AZURE_SPEECH_KEY='your-key'")
    print("   export AZURE_SPEECH_REGION='your-region'")
    print("🛑 Press Ctrl+C to stop")
    app.run(debug=True, port=5000)
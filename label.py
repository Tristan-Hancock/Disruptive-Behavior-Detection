import cv2
import numpy as np
import time
from collections import deque
import mediapipe as mp
import os
from tensorflow.keras.models import load_model

# Load your custom model.
target_model = load_model('models/model.h5')

def preprocess_face(face, target_size=(224, 224)):
    """
    Preprocess a face region for your custom model:
      - Resize to target_size
      - Convert from BGR to RGB
      - Normalize pixel values between 0 and 1
      - Expand dimensions for batch compatibility
    """
    face = cv2.resize(face, target_size)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=0)
    return face

class StudentBehaviorDetector:
    def __init__(self, attention_threshold=10, disruption_threshold=120):
        """
        Initialize the student behavior detector.
        
        Args:
            attention_threshold: Time (in seconds) after which looking away is considered distraction.
            disruption_threshold: Total distraction time (in seconds) to label as disruptive.
        """
        self.attention_threshold = attention_threshold
        self.disruption_threshold = disruption_threshold
        
        # Initialize MediaPipe solutions.
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=10,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        
        # State tracking variables.
        self.current_state = "attentive"
        self.position_history = deque(maxlen=30)
        self.gaze_history = deque(maxlen=10)
        self.distraction_start_time = None
        self.total_distraction_time = 0
        self.last_frame_time = None
        
        # Face tracking.
        self.target_tracker = cv2.TrackerKCF_create()
        self.tracking_initialized = False
        self.tracking_box = None
        self.face_detection_interval = 30
        self.frame_count = 0

    def identify_target_student(self, frame):
        """
        Identify the target student in the frame using the custom model.
        
        For each detected face, this function:
          - Extracts and preprocesses the candidate face.
          - Uses the custom model (target_model) to get a confidence score.
          - Selects the candidate if its confidence exceeds a threshold.
        
        Args:
            frame: Video frame to process.
        
        Returns:
            (face_box, is_target): The bounding box of the detected target face and a Boolean.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detection_results = self.face_detection.process(frame_rgb)
        
        if not detection_results.detections:
            return None, False
        
        best_confidence = 0.0
        best_match_box = None
        CONFIDENCE_THRESHOLD = 0.7  # Adjust as needed.
        
        for detection in detection_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            bbox = (
                int(bboxC.xmin * iw),
                int(bboxC.ymin * ih),
                int(bboxC.width * iw),
                int(bboxC.height * ih)
            )
            
            x, y, w, h = bbox
            face_region = frame[y:y+h, x:x+w]
            if face_region.size == 0:
                continue
            
            processed_face = preprocess_face(face_region, target_size=(224, 224))
            prediction = target_model.predict(processed_face)
            confidence = prediction[0][0]
            print(f"Candidate confidence: {confidence:.2f}")
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_match_box = bbox
        
        if best_confidence > CONFIDENCE_THRESHOLD:
            print(f"Selected candidate with confidence: {best_confidence:.2f}")
            return best_match_box, True
        
        return None, False

    def update_tracking(self, frame, face_box=None):
        if face_box is not None:
            x, y, w, h = face_box
            self.tracking_box = (x, y, w, h)
            self.target_tracker = cv2.TrackerKCF_create()
            self.target_tracker.init(frame, self.tracking_box)
            self.tracking_initialized = True
        elif self.tracking_initialized:
            success, bbox = self.target_tracker.update(frame)
            if success:
                self.tracking_box = tuple(map(int, bbox))
            else:
                self.tracking_initialized = False
        return self.tracking_box

    def detect_gaze_direction(self, frame, face_box):
        if face_box is None:
            return True
        x, y, w, h = face_box
        face_image = frame[y:y+h, x:x+w]
        if face_image.size == 0:
            return True
        face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(face_image_rgb)
        if not results.multi_face_landmarks:
            return True
        landmarks = results.multi_face_landmarks[0]
        def landmark_to_point(landmark):
            return (int(landmark.x * w), int(landmark.y * h))
        nose_tip = landmark_to_point(landmarks.landmark[4])
        chin = landmark_to_point(landmarks.landmark[152])
        left_eye = landmark_to_point(landmarks.landmark[33])
        right_eye = landmark_to_point(landmarks.landmark[263])
        left_mouth = landmark_to_point(landmarks.landmark[61])
        right_mouth = landmark_to_point(landmarks.landmark[291])
        face_center = (int(w/2), int(h/2))
        nose_displacement_x = (nose_tip[0] - face_center[0]) / (w/2)
        nose_displacement_y = (nose_tip[1] - face_center[1]) / (h/2)
        eye_angle = np.arctan2(right_eye[1]-left_eye[1], right_eye[0]-left_eye[0]) * 180 / np.pi
        is_attentive = (abs(nose_displacement_x) < 0.2 and
                        abs(nose_displacement_y) < 0.2 and
                        abs(eye_angle) < 10)
        self.gaze_history.append(is_attentive)
        return sum(self.gaze_history) > len(self.gaze_history) / 2

    def detect_movement(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        if not results.pose_landmarks:
            return False
        try:
            left_shoulder = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            if left_shoulder.visibility < 0.5 or right_shoulder.visibility < 0.5:
                return False
            current_position = ((left_shoulder.x + right_shoulder.x) / 2,
                                (left_shoulder.y + right_shoulder.y) / 2)
            self.position_history.append(current_position)
            if len(self.position_history) < 10:
                return False
            avg_x = sum(pos[0] for pos in list(self.position_history)[:-1]) / (len(self.position_history) - 1)
            avg_y = sum(pos[1] for pos in list(self.position_history)[:-1]) / (len(self.position_history) - 1)
            dx = current_position[0] - avg_x
            dy = current_position[1] - avg_y
            distance = (dx**2 + dy**2)**0.5
            return distance > 0.05
        except:
            return False

    def update_behavior_state(self, is_attentive, has_moved, current_time):
        if self.last_frame_time is None:
            self.last_frame_time = current_time
        if has_moved:
            self.current_state = "disruptive"
            return self.current_state
        time_delta = current_time - self.last_frame_time
        self.last_frame_time = current_time
        if not is_attentive:
            if self.distraction_start_time is None:
                self.distraction_start_time = current_time
                self.current_state = "attentive"
            else:
                continuous_distraction_time = current_time - self.distraction_start_time
                if continuous_distraction_time >= self.attention_threshold:
                    self.total_distraction_time += time_delta
                    if self.total_distraction_time >= self.disruption_threshold:
                        self.current_state = "disruptive"
                    else:
                        self.current_state = "distracted"
        else:
            self.distraction_start_time = None
            self.current_state = "attentive"
        return self.current_state

    def process_frame(self, frame, timestamp):
        self.frame_count += 1
        # Trigger redetection periodically or if tracking is not initialized.
        if self.frame_count % self.face_detection_interval == 0 or not self.tracking_initialized:
            face_box, is_target = self.identify_target_student(frame)
            if not is_target or face_box is None:
                return frame, "not_found"
        else:
            face_box = None
        tracking_box = self.update_tracking(frame, face_box)
        if tracking_box is None:
            return frame, "not_found"
        is_attentive = self.detect_gaze_direction(frame, tracking_box)
        has_moved = self.detect_movement(frame)
        state = self.update_behavior_state(is_attentive, has_moved, timestamp)
        x, y, w, h = tracking_box
        if state == "attentive":
            color = (0, 255, 0)
        elif state == "distracted":
            color = (0, 165, 255)
        else:
            color = (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, state, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(frame, f"Total distraction: {self.total_distraction_time:.1f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if self.distraction_start_time is not None:
            continuous_time = timestamp - self.distraction_start_time
            cv2.putText(frame, f"Current look-away: {continuous_time:.1f}s", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 165, 255) if continuous_time >= self.attention_threshold else (255, 255, 255),
                        2)
        state_colors = {"attentive": (0, 255, 0),
                        "distracted": (0, 165, 255),
                        "disruptive": (0, 0, 255)}
        cv2.putText(frame, f"State: {state}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    state_colors.get(state, (255, 255, 255)), 2)
        cv2.putText(frame,
                    f"Distraction threshold: {self.attention_threshold}s | Disruption threshold: {self.disruption_threshold}s",
                    (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        return frame, state

def run_detection():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise ValueError("Could not open the camera")
    detector = StudentBehaviorDetector()
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = 0
    start_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        timestamp = time.time()
        frame_count += 1
        processed_frame, state = detector.process_frame(frame, timestamp)
        cv2.imshow('Student Behavior Analysis - Live', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    total_time = time.time() - start_time
    return {
        "total_class_time": total_time,
        "total_distraction_time": detector.total_distraction_time,
        "distraction_percentage": (detector.total_distraction_time / total_time) * 100 if total_time > 0 else 0,
        "final_state": detector.current_state
    }

def main():
    try:
        results = run_detection()
        print("\nBehavior Analysis Results:")
        print(f"Total class time: {results['total_class_time']:.2f} seconds")
        print(f"Total distraction time: {results['total_distraction_time']:.2f} seconds")
        print(f"Distraction percentage: {results['distraction_percentage']:.2f}%")
        print(f"Final behavior state: {results['final_state']}")
    except Exception as e:
        print(f"Error occurred during analysis: {e}")

if __name__ == "__main__":
    main()

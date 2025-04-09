import cv2
import numpy as np
import time
from collections import deque
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import csv  # For CSV logging

class StudentBehaviorDetector:
    def __init__(self, attention_threshold=3, movement_threshold=0.05):
        """
        Initialize the student behavior detector.
        
        Args:
            attention_threshold (float): Time (in seconds) after which Not Paying Attention is flagged.
            movement_threshold (float): Normalized threshold for shoulder movement.
        """
        self.attention_threshold = attention_threshold
        self.movement_threshold = movement_threshold

        # Initialize MediaPipe Face Detection.
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )

        # Initialize MediaPipe Face Mesh for gaze and expression estimation.
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=4,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

        # Initialize MediaPipe Pose for shoulder movement detection.
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False, model_complexity=1,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

        self.mp_drawing = mp.solutions.drawing_utils

        # Load the identity detection model (trained to recognize Tristan).
        self.identity_model = load_model("models/final_model.h5")
        # Set a threshold for recognizing Tristan (adjust as needed).
        self.identity_threshold = 0.9

        # Initialize state tracking variables.
        self.last_shoulder_midpoint = None
        self.current_state = "Student Paying Attention"
        self.position_history = deque(maxlen=30)
        self.gaze_history = deque(maxlen=10)
        self.distraction_start_time = None
        self.total_distraction_time = 0
        self.last_frame_time = None

        # Tracker (for potential future use).
        self.target_tracker = cv2.TrackerKCF_create()
        self.tracking_initialized = False
        self.tracking_box = None
        self.face_detection_interval = 30
        self.frame_count = 0
        self.last_gaze_time = time.time()

    def preprocess_face(self, face, target_size=(224, 224)):
        """
        Preprocess the input face image for identity detection.
        Resizes to the target size, converts to RGB, normalizes pixel values,
        and expands dimensions.
        """
        if face.size == 0:
            raise ValueError("Empty face image provided for preprocessing.")
        face = cv2.resize(face, target_size)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype('float32') / 255.0
        face = np.expand_dims(face, axis=0)
        return face

    def detect_faces(self, frame):
        """
        Detect all faces in the frame using MediaPipe Face Detection.
        Returns a list of detections.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(frame_rgb)
        if results.detections:
            return results.detections
        return []

    def get_face_box(self, frame, detection):
        """
        Converts the MediaPipe detection into a bounding box (x, y, w, h) in pixel coordinates.
        """
        ih, iw, _ = frame.shape
        bbox = detection.location_data.relative_bounding_box
        x = int(bbox.xmin * iw)
        y = int(bbox.ymin * ih)
        w = int(bbox.width * iw)
        h = int(bbox.height * ih)
        return (x, y, w, h)

    def detect_gaze_direction(self, frame, face_box):
        """
        Estimates gaze direction by computing the normalized horizontal 
        displacement of the nose tip relative to the center between the eyes.
        """
        x, y, w, h = face_box
        face_roi = frame[y:y+h, x:x+w]
        face_roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(face_roi_rgb)
        if results.multi_face_landmarks:
            # Use the first set of landmarks for the detected face.
            landmarks = results.multi_face_landmarks[0]
            def point(landmark):
                return int(landmark.x * w), int(landmark.y * h)
            nose_tip = point(landmarks.landmark[4])
            left_eye = point(landmarks.landmark[33])
            right_eye = point(landmarks.landmark[263])
            eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
            dx = nose_tip[0] - eye_center[0]
            norm_dx = dx / float(w)
            return norm_dx
        return 0

    def detect_shoulder_movement(self, frame):
        """
        Uses MediaPipe Pose to compute the midpoint of the shoulders.
        Returns True if movement exceeds the threshold.
        """
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                ih, iw, _ = frame.shape
                midpoint_norm = ((left_shoulder.x + right_shoulder.x) / 2.0,
                                 (left_shoulder.y + right_shoulder.y) / 2.0)
                midpoint = (midpoint_norm[0] * iw, midpoint_norm[1] * ih)
                if self.last_shoulder_midpoint is not None:
                    dx = midpoint[0] - self.last_shoulder_midpoint[0]
                    dy = midpoint[1] - self.last_shoulder_midpoint[1]
                    dist = np.sqrt(dx ** 2 + dy ** 2)
                    self.last_shoulder_midpoint = midpoint
                    if dist > self.movement_threshold * iw:
                        return True
                else:
                    self.last_shoulder_midpoint = midpoint
        except Exception as e:
            print("Error in detect_shoulder_movement:", e)
            return False
        return False

    def detect_expression(self, frame, face_box):
        """
        Detects a facial expression (e.g. laughing or smiling) by computing the
        Mouth Aspect Ratio (MAR) from mouth landmarks.
        """
        x, y, w, h = face_box
        face_roi = frame[y:y+h, x:x+w]
        face_roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(face_roi_rgb)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            def to_point(landmark):
                return np.array([landmark.x * w, landmark.y * h])
            try:
                mouth_left = to_point(landmarks.landmark[61])
                mouth_right = to_point(landmarks.landmark[291])
                upper_lip = to_point(landmarks.landmark[13])
                lower_lip = to_point(landmarks.landmark[14])
            except IndexError:
                return 0.0
            mouth_width = np.linalg.norm(mouth_right - mouth_left)
            mouth_height = np.linalg.norm(lower_lip - upper_lip)
            mar = mouth_height / mouth_width if mouth_width > 0 else 0.0
            return mar
        return 0.0

    def update_behavior_state(self, is_attentive, has_moved, current_time):
        if self.last_frame_time is None:
            self.last_frame_time = current_time
        if has_moved:
            self.current_state = "Bad Behaviour - Being Disruptive"
            return self.current_state
        time_delta = current_time - self.last_frame_time
        self.last_frame_time = current_time
        if not is_attentive:
            if self.distraction_start_time is None:
                self.distraction_start_time = current_time
            else:
                continuous_time = current_time - self.distraction_start_time
                if continuous_time >= self.attention_threshold:
                    self.total_distraction_time += time_delta
                    if self.total_distraction_time >= 10:
                        self.current_state = "Bad Behaviour - Being Disruptive"
                    else:
                        self.current_state = "distracted"
        else:
            self.distraction_start_time = None
            self.current_state = "Student Paying Attention"
        return self.current_state

    def process_frame(self, frame, frame_index=None):
        """
        Processes the frame: detects all faces, performs identity detection to determine
        if a person is Tristan, estimates gaze direction, checks shoulder movement,
        and evaluates facial expression for each detected face.
        Returns the annotated frame, an overall combined text, and a list of per-face results.
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        # Detect all faces in the frame.
        face_detections = self.detect_faces(frame)
        if not face_detections:
            cv2.putText(frame, "No face detected", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame, "No face detected", []
        
        combined_texts = []  # To store annotation information for each detected face.
        face_results = []    # List to accumulate detailed results.
        
        for idx, detection in enumerate(face_detections):
            # Get the face bounding box.
            face_box = self.get_face_box(frame, detection)
            x, y, w, h = face_box

            # Draw a rectangle around the detected face.
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Clamp the bounding box to the frame dimensions.
            frame_h, frame_w = frame.shape[:2]
            x_end = min(x + w, frame_w)
            y_end = min(y + h, frame_h)
            
            # Extract the face ROI.
            face_img = frame[y:y_end, x:x_end]
            if face_img.size == 0:
                identity_text = "Non Target Student"
                id_color = (0, 0, 255)
            else:
                # Preprocess and run identity detection.
                try:
                    face_preprocessed = self.preprocess_face(face_img)
                    prediction = self.identity_model.predict(face_preprocessed)
                    prob = prediction[0][0]
                    if prob > self.identity_threshold:
                        identity_text = "Tristan"
                        id_color = (0, 255, 0)
                    else:
                        identity_text = "Non Target Student"
                        id_color = (0, 0, 255)
                except Exception as e:
                    print("Error in identity detection:", e)
                    identity_text = "Non Target Student"
                    id_color = (0, 0, 255)
            
            # Annotate identity above the face.
            cv2.putText(frame, identity_text, (x, y - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, id_color, 2)
            
            # Gaze detection.
            norm_dx = self.detect_gaze_direction(frame, face_box)
            if abs(norm_dx) < 0.1:
                gaze_text = "Paying Attention"
                self.last_gaze_time = time.time()
            else:
                gaze_text = "Not Paying Attention"
                duration = time.time() - self.last_gaze_time
                if duration >= self.attention_threshold:
                    gaze_text += " too long"
            cv2.putText(frame, gaze_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
            # Shoulder movement detection.
            moving = self.detect_shoulder_movement(frame)
            movement_text = "Shoulders moving" if moving else "Static shoulders"
            cv2.putText(frame, movement_text, (x, y + h + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            # Expression detection.
            mar = self.detect_expression(frame, face_box)
            expression_text = "Laughing" if mar > 0.3 else "Neutral"
            cv2.putText(frame, expression_text, (x, y + h + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
            
            # Update overall behavior state.
            current_time = time.time()
            is_attentive = abs(norm_dx) < 0.1
            has_moved = moving
            self.update_behavior_state(is_attentive, has_moved, current_time)
            
            # Combine annotation information for this face.
            face_text = (f"Identity: {identity_text}, Gaze: {gaze_text}, "
                         f"Movement: {movement_text}, Expr: {expression_text}")
            combined_texts.append(face_text)
            
            # Optionally, draw this combined text near the face.
            cv2.putText(frame, face_text, (30, y + h + 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Record results for this face.
            result = {
                "timestamp": timestamp,
                "frame": frame_index if frame_index is not None else "",
                "face_index": idx,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "identity": identity_text,
                "gaze": gaze_text,
                "movement": movement_text,
                "expression": expression_text,
                "overall_state": self.current_state
            }
            face_results.append(result)
        
        overall_text = " | ".join(combined_texts)
        return frame, overall_text, face_results

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    detector = StudentBehaviorDetector(attention_threshold=3, movement_threshold=0.05)
    all_detections_log = []  # This will accumulate log entries for each detected face.
    frame_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame_counter += 1
        processed_frame, state, frame_results = detector.process_frame(frame, frame_index=frame_counter)
        # Append current frame results to our log.
        all_detections_log.extend(frame_results)
        cv2.imshow("Live Feed", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Write the log entries to a CSV file.
    csv_filename = "detections_log.csv"
    fieldnames = ["timestamp", "frame", "face_index", "x", "y", "w", "h", "identity", "gaze", "movement", "expression", "overall_state"]
    try:
        with open(csv_filename, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for entry in all_detections_log:
                writer.writerow(entry)
        print(f"Detection log saved to '{csv_filename}'.")
    except Exception as e:
        print("Error writing CSV file:", e)

if __name__ == "__main__":
    main()

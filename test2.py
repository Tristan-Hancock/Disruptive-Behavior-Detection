import cv2
import numpy as np
import time
from collections import deque
import mediapipe as mp

class StudentBehaviorDetector:
    def __init__(self, attention_threshold=3, movement_threshold=0.05):
        """
        Initialize the student behavior detector.

        Args:
            attention_threshold (float): Time (in seconds) after which looking away is flagged.
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
            static_image_mode=False, max_num_faces=1,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

        # Initialize MediaPipe Pose for shoulder movement detection.
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False, model_complexity=1,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

        self.mp_drawing = mp.solutions.drawing_utils

        # Initialize state tracking variables.
        self.last_shoulder_midpoint = None
        self.current_state = "attentive"
        self.position_history = deque(maxlen=30)
        self.gaze_history = deque(maxlen=10)
        self.distraction_start_time = None
        self.total_distraction_time = 0
        self.last_frame_time = None
        # We use a tracker if needed (here it's available for future extension).
        self.target_tracker = cv2.TrackerKCF_create()
        self.tracking_initialized = False
        self.tracking_box = None
        self.face_detection_interval = 30
        self.frame_count = 0
        self.last_gaze_time = time.time()

    def detect_face(self, frame):
        """Detect the first face in the frame using MediaPipe Face Detection."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(frame_rgb)
        if results.detections:
            return results.detections[0]
        return None

    def get_face_box(self, frame, detection):
        """Convert the MediaPipe detection to a bounding box (x, y, w, h) in pixel coordinates."""
        ih, iw, _ = frame.shape
        bbox = detection.location_data.relative_bounding_box
        x = int(bbox.xmin * iw)
        y = int(bbox.ymin * ih)
        w = int(bbox.width * iw)
        h = int(bbox.height * ih)
        return (x, y, w, h)

    def detect_gaze_direction(self, frame, face_box):
        """
        Estimate the gaze direction by computing the normalized horizontal 
        displacement of the nose tip relative to the center of the eye line.
        """
        x, y, w, h = face_box
        face_roi = frame[y:y+h, x:x+w]
        face_roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(face_roi_rgb)
        if results.multi_face_landmarks:
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
        Returns True if the movement exceeds a set threshold.
        """
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
        return False

    def detect_expression(self, frame, face_box):
        """
        Detects a facial expression (e.g. laughing or smiling) by computing the
        Mouth Aspect Ratio (MAR) using specific mouth landmarks from the face mesh.
        
        Returns:
            mar (float): The mouth aspect ratio, where a higher value may indicate laughing.
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
            self.current_state = "disruptive"
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
                    if self.total_distraction_time >= 10:  # Example disruption threshold.
                        self.current_state = "disruptive"
                    else:
                        self.current_state = "distracted"
        else:
            self.distraction_start_time = None
            self.current_state = "attentive"
        return self.current_state

    def process_frame(self, frame):
        """
        Processes the frame: detects face, estimates gaze direction,
        checks shoulder movement, and evaluates facial expression.
        Draws the bounding box and annotates the frame with the relevant info.
        """
        face_detection = self.detect_face(frame)
        if face_detection is None:
            cv2.putText(frame, "No face detected", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame, "No face detected"
        
        face_box = self.get_face_box(frame, face_detection)
        cv2.rectangle(frame, (face_box[0], face_box[1]), 
                      (face_box[0] + face_box[2], face_box[1] + face_box[3]), 
                      (0, 255, 0), 2)
        
        # Gaze detection.
        norm_dx = self.detect_gaze_direction(frame, face_box)
        if abs(norm_dx) < 0.1:
            gaze_text = "Looking straight"
            self.last_gaze_time = time.time()
        else:
            gaze_text = "Looking away"
            duration = time.time() - self.last_gaze_time
            if duration >= self.attention_threshold:
                gaze_text += " too long"
        cv2.putText(frame, gaze_text, (face_box[0], face_box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        # Shoulder movement detection.
        moving = self.detect_shoulder_movement(frame)
        movement_text = "Shoulders moving" if moving else "Static shoulders"
        cv2.putText(frame, movement_text, (face_box[0], face_box[1] + face_box[3] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        # Expression detection.
        mar = self.detect_expression(frame, face_box)
        if mar > 0.3:
            expression_text = "Laughing"
        else:
            expression_text = "Neutral"
        cv2.putText(frame, expression_text, (face_box[0], face_box[1] + face_box[3] + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
        
        # Update overall behavior state.
        current_time = time.time()
        is_attentive = abs(norm_dx) < 0.1
        has_moved = moving
        state = self.update_behavior_state(is_attentive, has_moved, current_time)
        combined_text = f"Gaze: {gaze_text}, Movement: {movement_text}, Expr: {expression_text}"
        cv2.putText(frame, combined_text, (30, face_box[1] + face_box[3] + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return frame, combined_text

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    detector = StudentBehaviorDetector(attention_threshold=3, movement_threshold=0.05)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        processed_frame, state = detector.process_frame(frame)
        cv2.imshow("Live Feed", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

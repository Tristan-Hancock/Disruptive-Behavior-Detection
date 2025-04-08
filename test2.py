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
            attention_threshold (float): Time in seconds after which looking away is flagged.
            movement_threshold (float): Normalized distance threshold for shoulder movement.
        """
        self.attention_threshold = attention_threshold
        self.movement_threshold = movement_threshold

        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )

        # Initialize MediaPipe Face Mesh for gaze estimation
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

        # Initialize MediaPipe Pose for movement detection
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False, model_complexity=1,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

        # Variables to track gaze and movement over time
        self.last_gaze_time = time.time()
        self.last_shoulder_midpoint = None

    def detect_face(self, frame):
        """Detects the first face in the frame using MediaPipe Face Detection."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(frame_rgb)
        if results.detections:
            return results.detections[0]
        return None

    def get_face_box(self, frame, detection):
        """Converts MediaPipe detection to a bounding box in pixel coordinates."""
        ih, iw, _ = frame.shape
        bbox = detection.location_data.relative_bounding_box
        x = int(bbox.xmin * iw)
        y = int(bbox.ymin * ih)
        w = int(bbox.width * iw)
        h = int(bbox.height * ih)
        return (x, y, w, h)

    def detect_gaze_direction(self, frame, face_box):
        """
        Estimates gaze direction by using the face mesh landmarks.
        Returns the normalized horizontal displacement (dx/w) of the nose tip
        from the center of the eye line.
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
            norm_dx = dx / float(w)  # normalized horizontal displacement
            return norm_dx
        return 0

    def detect_shoulder_movement(self, frame):
        """
        Uses MediaPipe Pose to compute the midpoint of the shoulders.
        Returns True if the movement exceeds a threshold relative to the previous midpoint.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            ih, iw, _ = frame.shape
            midpoint = ((left_shoulder.x + right_shoulder.x) / 2.0,
                        (left_shoulder.y + right_shoulder.y) / 2.0)
            # Convert from normalized to pixel coordinates.
            midpoint = (midpoint[0] * iw, midpoint[1] * ih)
            if self.last_shoulder_midpoint is not None:
                dx = midpoint[0] - self.last_shoulder_midpoint[0]
                dy = midpoint[1] - self.last_shoulder_midpoint[1]
                dist = np.sqrt(dx*dx + dy*dy)
                self.last_shoulder_midpoint = midpoint
                if dist > self.movement_threshold * iw:
                    return True
            else:
                self.last_shoulder_midpoint = midpoint
        return False

    def process_frame(self, frame):
        """
        Process the frame: detect face, evaluate gaze, and check shoulder movement.
        Draws bounding box and text annotations.
        """
        face_detection = self.detect_face(frame)
        state = ""
        if face_detection is None:
            cv2.putText(frame, "No face detected", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame, "No face detected"
        # Get face bounding box and draw it.
        face_box = self.get_face_box(frame, face_detection)
        cv2.rectangle(frame, (face_box[0], face_box[1]), (face_box[0]+face_box[2], face_box[1]+face_box[3]), (0,255,0), 2)
        
        # Gaze detection
        norm_dx = self.detect_gaze_direction(frame, face_box)
        # If gaze deviation is small, assume attention.
        if abs(norm_dx) < 0.1:
            gaze_text = "Looking straight"
            self.last_gaze_time = time.time()
        else:
            # If deviation exceeds threshold, calculate distraction duration.
            gaze_text = "Looking away"
            duration = time.time() - self.last_gaze_time
            if duration >= self.attention_threshold:
                gaze_text += " for too long"
        cv2.putText(frame, gaze_text, (face_box[0], face_box[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
        
        # Movement detection
        moving = self.detect_shoulder_movement(frame)
        movement_text = "Shoulders moving" if moving else "Static shoulders"
        cv2.putText(frame, movement_text, (face_box[0], face_box[1] + face_box[3] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        
        state = f"{gaze_text}, {movement_text}"
        return frame, state

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

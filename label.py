import cv2
import numpy as np
import time
from collections import deque
import mediapipe as mp
import os

class StudentBehaviorDetector:
    def __init__(self, target_face_descriptor, attention_threshold=10, disruption_threshold=120):
        """
        Initialize the student behavior detector.
        
        Args:
            target_face_descriptor: Face descriptor of the target student (can be an image or features)
            attention_threshold: Time in seconds after which looking away is considered distraction
            disruption_threshold: Total distraction time in seconds to label as disruptive
        """
        self.target_face_descriptor = target_face_descriptor
        self.attention_threshold = attention_threshold
        self.disruption_threshold = disruption_threshold
        
        # Initialize MediaPipe solutions
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 0 for short-range, 1 for full-range detection
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
        
        # Drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        
        # State tracking variables
        self.current_state = "attentive"  # "attentive", "distracted", or "disruptive"
        self.position_history = deque(maxlen=30)  # To track position changes
        self.gaze_history = deque(maxlen=10)      # To smooth gaze direction detection
        
        # Time tracking variables
        self.distraction_start_time = None
        self.total_distraction_time = 0
        self.last_frame_time = None
        
        # Face tracking
        self.target_tracker = cv2.TrackerKCF_create()
        self.tracking_initialized = False
        self.tracking_box = None
        self.face_detection_interval = 30  # Frames between face detection runs
        self.frame_count = 0
    
    def identify_target_student(self, frame):
        """
        Identify if the target student is in the frame using face similarity.
        
        Args:
            frame: Video frame to process
            
        Returns:
            (face_box, is_target): Face bounding box and boolean indicating if target was found
        """
        # Run face detection
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detection_results = self.face_detection.process(frame_rgb)
        
        if not detection_results.detections:
            return None, False
        
        # Find the face most similar to our target
        best_match_score = 0
        best_match_box = None
        
        for detection in detection_results.detections:
            # Get bounding box
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            bbox = (
                int(bboxC.xmin * iw),
                int(bboxC.ymin * ih),
                int(bboxC.width * iw),
                int(bboxC.height * ih)
            )
            
            # Extract face for comparison
            x, y, w, h = bbox
            face_region = frame[y:y+h, x:x+w]
            
            if face_region.size == 0:  # Skip if face region is empty
                continue
                
            # In a real implementation, here we would compare the face with the target
            # For this example, we'll use a simple template matching approach
            
            # Resize target descriptor to match face region size
            resized_target = cv2.resize(self.target_face_descriptor, (w, h))
            
            # Calculate similarity score (normalized correlation coefficient)
            if resized_target.shape == face_region.shape:
                result = cv2.matchTemplate(
                    cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(resized_target, cv2.COLOR_BGR2GRAY),
                    cv2.TM_CCOEFF_NORMED
                )
                similarity_score = np.max(result)
                
                if similarity_score > best_match_score:
                    best_match_score = similarity_score
                    best_match_box = bbox
        
        # Use a threshold to determine if this is the target student
        if best_match_score > 0.5:  # Adjust threshold based on testing
            return best_match_box, True
        
        return None, False
    
    def update_tracking(self, frame, face_box=None):
        """
        Update the face tracking.
        
        Args:
            frame: Current video frame
            face_box: Optional face bounding box from detection
            
        Returns:
            tracking_box: Current tracking bounding box
        """
        if face_box is not None:
            # Initialize or re-initialize tracker with detected face
            x, y, w, h = face_box
            self.tracking_box = (x, y, w, h)
            self.target_tracker = cv2.TrackerKCF_create()
            self.target_tracker.init(frame, self.tracking_box)
            self.tracking_initialized = True
        elif self.tracking_initialized:
            # Update tracker
            success, bbox = self.target_tracker.update(frame)
            if success:
                self.tracking_box = tuple(map(int, bbox))
            else:
                self.tracking_initialized = False
                
        return self.tracking_box
    
    def detect_gaze_direction(self, frame, face_box):
        """
        Detect gaze direction using facial landmarks.
        
        Args:
            frame: Video frame
            face_box: Bounding box of the face
            
        Returns:
            is_attentive: Boolean indicating if the student is looking at the lecturer
        """
        if face_box is None:
            return True  # Default to attentive if no face detected
            
        # Extract face region
        x, y, w, h = face_box
        face_image = frame[y:y+h, x:x+w]
        
        if face_image.size == 0:  # Skip if face region is empty
            return True
            
        # Process with face mesh to get landmarks
        face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(face_image_rgb)
        
        if not results.multi_face_landmarks:
            return True  # Default to attentive if landmarks not detected
        
        landmarks = results.multi_face_landmarks[0]
        
        # Get head pose using facial landmarks
        # We'll focus on key points that indicate head orientation
        
        # Maps normalized coordinates to image coordinates
        def landmark_to_point(landmark):
            return (
                int(landmark.x * w),
                int(landmark.y * h)
            )
        
        # Key landmarks for head pose estimation:
        # Nose tip, chin, left eye corner, right eye corner, left mouth corner, right mouth corner
        nose_tip = landmark_to_point(landmarks.landmark[4])
        chin = landmark_to_point(landmarks.landmark[152])
        left_eye = landmark_to_point(landmarks.landmark[33])
        right_eye = landmark_to_point(landmarks.landmark[263])
        left_mouth = landmark_to_point(landmarks.landmark[61])
        right_mouth = landmark_to_point(landmarks.landmark[291])
        
        # Calculate face center
        face_center = (int(w/2), int(h/2))
        
        # Calculate head pose using the displacement of nose from center
        nose_displacement_x = (nose_tip[0] - face_center[0]) / (w/2)
        nose_displacement_y = (nose_tip[1] - face_center[1]) / (h/2)
        
        # Calculate eye line angle
        eye_angle = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]) * 180 / np.pi
        
        # Determine if attentive based on thresholds
        # These thresholds need calibration based on your specific setup
        is_attentive = (
            abs(nose_displacement_x) < 0.2 and    # Not looking too far left/right
            abs(nose_displacement_y) < 0.2 and    # Not looking too far up/down
            abs(eye_angle) < 10                   # Head not tilted too much
        )
        
        # Add to history for smoothing
        self.gaze_history.append(is_attentive)
        
        # Return smoothed result (majority vote)
        return sum(self.gaze_history) > len(self.gaze_history) / 2
    
    def detect_movement(self, frame):
        """
        Detect if the student has moved from their position.
        
        Args:
            frame: Video frame
            
        Returns:
            has_moved: Boolean indicating if the student has moved significantly
        """
        # Process with pose estimation
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if not results.pose_landmarks:
            return False
        
        # Get shoulder landmarks to track position
        try:
            left_shoulder = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            # Check if landmarks are visible
            if left_shoulder.visibility < 0.5 or right_shoulder.visibility < 0.5:
                return False
                
            # Calculate midpoint of shoulders
            current_position = (
                (left_shoulder.x + right_shoulder.x) / 2,
                (left_shoulder.y + right_shoulder.y) / 2
            )
            
            # Add to position history
            self.position_history.append(current_position)
            
            # Not enough history to determine movement
            if len(self.position_history) < 10:
                return False
            
            # Calculate average position from history (excluding current)
            avg_x = sum(pos[0] for pos in list(self.position_history)[:-1]) / (len(self.position_history) - 1)
            avg_y = sum(pos[1] for pos in list(self.position_history)[:-1]) / (len(self.position_history) - 1)
            
            # Calculate distance from current position to average
            dx = current_position[0] - avg_x
            dy = current_position[1] - avg_y
            distance = (dx**2 + dy**2)**0.5
            
            # If distance exceeds threshold, consider as movement
            return distance > 0.05  # Threshold depends on camera angle and distance
        except:
            return False
    
    def update_behavior_state(self, is_attentive, has_moved, current_time):
        """
        Update the student's behavior state based on observations.
        
        Args:
            is_attentive: Boolean indicating if student is attentive
            has_moved: Boolean indicating if student has moved
            current_time: Current timestamp
            
        Returns:
            current_state: Current behavior state
        """
        # Initialize last_frame_time if this is the first frame
        if self.last_frame_time is None:
            self.last_frame_time = current_time
            
        # Check for movement first (highest priority)
        if has_moved:
            self.current_state = "disruptive"
            return self.current_state
        
        # Calculate time since last frame
        time_delta = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        # Then check attention
        if not is_attentive:
            # Start tracking distraction if not already doing so
            if self.distraction_start_time is None:
                self.distraction_start_time = current_time
                self.current_state = "attentive"  # Initially attentive until threshold is reached
            else:
                # Calculate how long student has been continuously distracted
                continuous_distraction_time = current_time - self.distraction_start_time
                
                # If distracted for longer than the threshold
                if continuous_distraction_time >= self.attention_threshold:
                    # Add the time since last frame to total distraction time
                    self.total_distraction_time += time_delta
                    
                    # Determine state based on total accumulated distraction time
                    if self.total_distraction_time >= self.disruption_threshold:
                        self.current_state = "disruptive"
                    else:
                        self.current_state = "distracted"
        else:
            # Reset distraction timer if student becomes attentive
            self.distraction_start_time = None
            self.current_state = "attentive"
            
        return self.current_state
    
    def process_frame(self, frame, timestamp):
        """
        Process a video frame to detect student behavior.
        
        Args:
            frame: Video frame to process
            timestamp: Timestamp of the frame
            
        Returns:
            (frame, state): Processed frame and current behavior state
        """
        self.frame_count += 1
        face_box = None
        
        # Periodically re-detect face to avoid tracking drift
        if self.frame_count % self.face_detection_interval == 0 or not self.tracking_initialized:
            face_box, is_target = self.identify_target_student(frame)
            if not is_target:
                # If target lost during re-detection, continue with tracking if available
                if not self.tracking_initialized:
                    return frame, "not_found"
        
        # Update tracking
        tracking_box = self.update_tracking(frame, face_box)
        
        if tracking_box is None:
            return frame, "not_found"
        
        # Detect gaze direction
        is_attentive = self.detect_gaze_direction(frame, tracking_box)
        
        # Detect movement
        has_moved = self.detect_movement(frame)
        
        # Update behavior state
        state = self.update_behavior_state(is_attentive, has_moved, timestamp)
        
        # Annotate frame
        x, y, w, h = tracking_box
        if state == "attentive":
            color = (0, 255, 0)  # Green
        elif state == "distracted":
            color = (0, 165, 255)  # Orange
        else:  # disruptive
            color = (0, 0, 255)  # Red
            
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, state, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add distraction timer information
        cv2.putText(
            frame, 
            f"Total distraction: {self.total_distraction_time:.1f}s", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 255), 
            2
        )
        
        # Show continuous distraction if currently distracted
        if self.distraction_start_time is not None:
            continuous_time = timestamp - self.distraction_start_time
            cv2.putText(
                frame,
                f"Current look-away: {continuous_time:.1f}s", 
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255) if continuous_time >= self.attention_threshold else (255, 255, 255),
                2
            )
        
        # Add current state display
        state_colors = {
            "attentive": (0, 255, 0),    # Green
            "distracted": (0, 165, 255), # Orange
            "disruptive": (0, 0, 255)    # Red
        }
        cv2.putText(
            frame, 
            f"State: {state}", 
            (10, 90), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            state_colors.get(state, (255, 255, 255)), 
            2
        )
        
        # Add thresholds for reference
        cv2.putText(
            frame,
            f"Distraction threshold: {self.attention_threshold}s | Disruption threshold: {self.disruption_threshold}s",
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1
        )
        
        return frame, state


def run_detection(video_path, target_face_path):
    """
    Run the student behavior detection on a video.
    
    Args:
        video_path: Path to the video file
        target_face_path: Path to an image of the target student's face
    """
    # Load target face image
    target_face = cv2.imread(target_face_path)
    if target_face is None:
        raise ValueError(f"Could not load target face image from {target_face_path}")
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file {video_path}")
    
    # Initialize detector
    detector = StudentBehaviorDetector(target_face)
    
    # Get video properties for output
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writer for output
    output_path = 'behavior_analysis.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process the video
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Calculate timestamp
        frame_count += 1
        timestamp = frame_count / fps
        
        # Process frame
        processed_frame, state = detector.process_frame(frame, timestamp)
        
        # Write to output video
        out.write(processed_frame)
        
        # Display (optional)
        cv2.imshow('Student Behavior Analysis', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Analysis complete. Output saved to {output_path}")
    
    # Return summary
    return {
        "total_class_time": frame_count / fps,
        "total_distraction_time": detector.total_distraction_time,
        "distraction_percentage": (detector.total_distraction_time / (frame_count / fps)) * 100,
        "final_state": detector.current_state
    }


def main():
    """
    Main function to setup and run the behavior detection system.
    """
    # Hardcoded file paths - replace these with your actual file paths
    video_path = "classroom_video2.mp4"  # Change this to your video path
    target_face_path = "target_student2.jpg"  # Change this to your target face image
    
    # Validate input files
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} not found")
        return
    if not os.path.exists(target_face_path):
        print(f"Error: Target face image {target_face_path} not found")
        return
    
    # Run detection
    try:
        results = run_detection(video_path, target_face_path)
        
        # Print summary
        print("\nBehavior Analysis Results:")
        print(f"Total class time: {results['total_class_time']:.2f} seconds")
        print(f"Total distraction time: {results['total_distraction_time']:.2f} seconds")
        print(f"Distraction percentage: {results['distraction_percentage']:.2f}%")
        print(f"Final behavior state: {results['final_state']}")
    except Exception as e:
        print(f"Error occurred during analysis: {e}")


if __name__ == "__main__":
    main()
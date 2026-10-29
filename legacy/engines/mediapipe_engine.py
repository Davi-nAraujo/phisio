import mediapipe as mp
import cv2

class MediaPipeEngine:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5,
            model_complexity=1 # 0=Lite, 1=Full, 2=Heavy (More accurate, slower)
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def process(self, frame):
        # MP needs RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.pose.process(image_rgb)
        
        keypoints = {}
        h, w, _ = frame.shape

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Get specific landmarks
            hip_lm = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            knee_lm = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
            ankle_lm = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]

            # ROBUSTNESS CHECK: Visibility > 0.5
            # "Visibility" is MP's confidence score.
            if (hip_lm.visibility > 0.5 and 
                knee_lm.visibility > 0.5 and 
                ankle_lm.visibility > 0.5):
                
                keypoints['left_hip'] = [hip_lm.x * w, hip_lm.y * h]
                keypoints['left_knee'] = [knee_lm.x * w, knee_lm.y * h]
                keypoints['left_ankle'] = [ankle_lm.x * w, ankle_lm.y * h]
            
            # Draw skeleton for debug/visuals
            self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        return frame, keypoints
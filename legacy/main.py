import cv2
import time

# Core Modules
from core.geometry import calculate_angle_3d
from core.one_euro_filter import OneEuroFilter

# Engine Modules
from engines.mediapipe_engine import MediaPipeEngine
from engines.yolo_engine import YoloEngine

# UI Module
from ui.interface import UIManager 

def main():
    # --- CONFIGURATION ---
    # Options: 'mediapipe', 'yolo-laptop' (Small), 'yolo-vm' (Large P6)
    SELECTED_ENGINE = 'mediapipe' 

    # --- INITIALIZATION ---
    if SELECTED_ENGINE == 'mediapipe':
        engine = MediaPipeEngine()
    elif 'yolo' in SELECTED_ENGINE:
        # Auto-select model size based on name
        model = 'yolov8s-pose.pt' if 'laptop' in SELECTED_ENGINE else 'yolov8x-pose-p6.pt'
        engine = YoloEngine(model_name=model)

    # Filter Setup (Default clinical values)
    angle_filter = OneEuroFilter(freq=30, mincutoff=1.0, beta=0.005, dcutoff=1.0)
    
    # UI Setup
    ui = UIManager(engine_name=SELECTED_ENGINE)
    
    cap = cv2.VideoCapture(0)
    
    # Attempt to use higher quality camera settings if available
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)

        current_time = time.time()

        # 1. Update Filter Params from UI Sliders
        new_cutoff, new_beta = ui.get_tuner_params()
        angle_filter.setMinCutoff(new_cutoff)
        angle_filter.setBeta(new_beta)

        # 2. Run AI Engine
        # Returns frame (with lines) and keypoints (only if confident)
        frame, keypoints = engine.process(frame)

        # 3. Calculate Logic
        raw_angle = 0
        filtered_angle = 0
        
        # Only calculate if the engine gave us ALL the parts
        if 'left_hip' in keypoints and 'left_knee' in keypoints:
            hip = keypoints['left_hip']
            knee = keypoints['left_knee']
            ankle = keypoints['left_ankle']

            # Math
            raw_angle = calculate_angle_3d(hip, knee, ankle)
            
            # Filter
            filtered_angle = angle_filter(raw_angle, timestamp=current_time)
        
        # 4. Update Dashboard
        ui.update(frame, raw_angle, filtered_angle, keypoints)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    ui.close()

if __name__ == "__main__":
    main()
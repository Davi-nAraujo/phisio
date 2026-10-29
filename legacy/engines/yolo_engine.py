from ultralytics import YOLO
import numpy as np

class YoloEngine:
    def __init__(self, model_name='yolov8s-pose.pt', device=0):
        print(f"Loading YOLO: {model_name}...")
        self.model = YOLO(model_name)
        self.device = device
        # P6 models use 1280px, standard models use 640px
        self.imgsz = 1280 if '-p6' in model_name else 640

    def process(self, frame):
        # Run Inference (verbose=False keeps terminal clean)
        results = self.model(frame, device=self.device, imgsz=self.imgsz, verbose=False)
        annotated_frame = results[0].plot()
        
        keypoints = {}
        
        # Mapping: YOLO -> Standard Names
        # Indices: 11=LeftHip, 13=LeftKnee, 15=LeftAnkle
        try:
            # Get data with confidence scores: [x, y, conf]
            # .data gives us the raw tensor with confidence values
            kps = results[0].keypoints.data[0] 
            
            if len(kps) > 0:
                l_hip = kps[11]
                l_knee = kps[13]
                l_ankle = kps[15]

                # ROBUSTNESS CHECK: Confidence > 0.5 (50%)
                # We only accept data if the AI is sure.
                if (l_hip[2] > 0.5 and l_knee[2] > 0.5 and l_ankle[2] > 0.5):
                    
                    # Convert Tensor to standard Numpy Array [x, y]
                    keypoints['left_hip'] = l_hip[:2].cpu().numpy()
                    keypoints['left_knee'] = l_knee[:2].cpu().numpy()
                    keypoints['left_ankle'] = l_ankle[:2].cpu().numpy()

        except Exception:
            pass # No person detected or empty frame

        return annotated_frame, keypoints
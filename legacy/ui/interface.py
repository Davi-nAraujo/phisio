import cv2
import numpy as np

class UIManager:
    def __init__(self, engine_name="Unknown"):
        self.control_window = "Physio Control Panel"
        self.video_window = "Physio Video Feed"
        self.engine_name = engine_name
        
        # --- MEDICAL THEME (Light Mode) ---
        self.colors = {
            'bg': (245, 245, 247),       
            'card': (255, 255, 255),     
            'text_main': (50, 50, 50),   
            'text_sub': (100, 100, 100), 
            'primary': (200, 100, 0),    # Deep Blue
            'accent': (0, 150, 255),     # Orange/Gold
            'success': (60, 180, 75),    
            'border': (220, 220, 220)    
        }

        # Setup Windows
        cv2.namedWindow(self.control_window)
        cv2.namedWindow(self.video_window)
        
        # --- TRACKBARS (Direct Physics Control) ---
        def nothing(x): pass

        # Min Cutoff: 0.1 to 5.0 (Slider 1-50 -> Divide by 10)
        # Higher = Less lag, more jitter when still
        cv2.createTrackbar('Min Cutoff x10', self.control_window, 10, 50, nothing) 
        
        # Beta: 0.001 to 0.1 (Slider 1-100 -> Divide by 1000)
        # Higher = Less lag, more jitter when moving
        cv2.createTrackbar('Beta x1000', self.control_window, 5, 100, nothing) 

        # Background Canvas
        self.H, self.W = 500, 450
        self.dashboard_bg = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        self.dashboard_bg[:] = self.colors['bg']

    def get_tuner_params(self):
        """Reads sliders and converts directly to filter parameters."""
        # Read raw slider values
        mc_val = cv2.getTrackbarPos('Min Cutoff x10', self.control_window)
        beta_val = cv2.getTrackbarPos('Beta x1000', self.control_window)
        
        # DIRECT MAPPING (No inverse logic)
        # Min Cutoff: Slider 10 -> 1.0 Hz
        min_cutoff = max(0.1, mc_val / 10.0)

        # Beta: Slider 5 -> 0.005
        beta = beta_val / 1000.0

        return min_cutoff, beta

    def draw_medical_card(self, image, x, y, w, h, title):
        cv2.rectangle(image, (x-1, y-1), (x + w + 1, y + h + 1), self.colors['border'], -1)
        cv2.rectangle(image, (x, y), (x + w, y + h), self.colors['card'], -1)
        cv2.line(image, (x + 15, y + 35), (x + w - 15, y + 35), self.colors['border'], 1)
        cv2.putText(image, title.upper(), (x + 15, y + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['primary'], 1, cv2.LINE_AA)

    def update(self, frame, raw_angle, filtered_angle, keypoints):
        dash = self.dashboard_bg.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Get current values to display text
        mc, beta = self.get_tuner_params()

        # --- 1. SYSTEM CARD ---
        self.draw_medical_card(dash, 20, 20, 410, 100, "Filter Physics")
        
        # Draw the exact values being used
        cv2.putText(dash, f"Min Cutoff: {mc:.1f}", (35, 60), font, 0.6, self.colors['text_main'], 1)
        cv2.putText(dash, "(Sensitivity when still)", (220, 60), font, 0.4, self.colors['text_sub'], 1)

        cv2.putText(dash, f"Beta: {beta:.4f}", (35, 90), font, 0.6, self.colors['text_main'], 1)
        cv2.putText(dash, "(Lag reduction on move)", (220, 90), font, 0.4, self.colors['text_sub'], 1)

        # --- 2. MEASUREMENT CARD ---
        self.draw_medical_card(dash, 20, 140, 410, 250, "Goniometry")
        
        cv2.putText(dash, "Raw Sensor Input", (40, 190), font, 0.5, self.colors['text_sub'], 1, cv2.LINE_AA)
        cv2.putText(dash, f"{int(raw_angle)} deg", (40, 220), font, 0.8, (150, 150, 150), 1, cv2.LINE_AA)

        cv2.putText(dash, "Clinical Angle", (220, 190), font, 0.5, self.colors['text_sub'], 1, cv2.LINE_AA)
        cv2.putText(dash, f"{int(filtered_angle)}", (220, 240), font, 2.0, self.colors['primary'], 2, cv2.LINE_AA)
        cv2.putText(dash, "deg", (350, 240), font, 0.6, self.colors['text_sub'], 1, cv2.LINE_AA)

        # Progress Bar
        cv2.rectangle(dash, (40, 280), (370, 290), (230, 230, 230), -1)
        bar_w = int((filtered_angle / 180.0) * 330)
        bar_w = max(0, min(330, bar_w))
        cv2.rectangle(dash, (40, 280), (40 + bar_w, 290), self.colors['accent'], -1)
        
        # Axis labels
        cv2.putText(dash, "0", (40, 310), font, 0.4, self.colors['text_sub'], 1)
        cv2.putText(dash, "90", (195, 310), font, 0.4, self.colors['text_sub'], 1)
        cv2.putText(dash, "180", (350, 310), font, 0.4, self.colors['text_sub'], 1)

        # --- 3. VIDEO WINDOW ---
        has_signal = 'left_knee' in keypoints
        if has_signal:
            knee = keypoints['left_knee']
            hip = keypoints['left_hip']
            ankle = keypoints['left_ankle']
            
            pts = np.array([hip, knee, ankle], np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], False, (255, 255, 255), 2)
            cv2.circle(frame, (int(knee[0]), int(knee[1])), 10, (0, 255, 0), -1)
            cv2.putText(frame, f"{int(filtered_angle)}", (int(knee[0])+15, int(knee[1])), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        cv2.imshow(self.control_window, dash)
        cv2.imshow(self.video_window, frame)

    def close(self):
        cv2.destroyAllWindows()
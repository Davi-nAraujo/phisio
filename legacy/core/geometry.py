import numpy as np

def calculate_angle_3d(a, b, c):
    """
    Calculates the 3D angle at point B formed by points A-B-C.
    Returns angle in degrees (0-180).
    
    Robustness: Returns 0.0 if vectors are invalid (length 0).
    """
    a, b, c = np.array(a), np.array(b), np.array(c)
    
    # Vector Subtraction (Creating the bones)
    ba = a - b 
    bc = c - b 
    
    # Calculate lengths
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    # SAFETY CHECK: If the AI puts points on top of each other, length is 0.
    # We must return 0.0 to prevent a "Divide by Zero" crash.
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0
    
    # Dot Product Formula: (A . B) / (|A| * |B|)
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    
    # Clip to safety range (-1.0 to 1.0) to prevent numerical errors
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)
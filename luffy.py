import cv2
import mediapipe as mp
import math
import os
import urllib.request
import time
import numpy as np

# 1. Ensure models are downloaded
FACE_MODEL_PATH = "face_landmarker.task"
HAND_MODEL_PATH = "hand_landmarker.task"

# 2. Advanced Directional Pull Math
def create_pull_effect(image, cheek_center, target_pos, radius):
    h, w = image.shape[:2]
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    
    dist_x = X - cheek_center[0]
    dist_y = Y - cheek_center[1]
    distance = np.sqrt(dist_x**2 + dist_y**2)
    
    # Calculate how far the hand has pulled away
    pull_x = target_pos[0] - cheek_center[0]
    pull_y = target_pos[1] - cheek_center[1]
    
    mask = distance < radius
    
    # Smooth linear falloff so the edge blends perfectly with the background
    falloff = (radius - distance) / radius
    falloff = np.clip(falloff, 0, 1) ** 1.5  
    
    map_x = np.where(mask, X - pull_x * falloff, X).astype(np.float32)
    map_y = np.where(mask, Y - pull_y * falloff, Y).astype(np.float32)
    
    map_x = np.clip(map_x, 0, w - 1)
    map_y = np.clip(map_y, 0, h - 1)
    
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)

# 3. Setup MediaPipe Engines
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

face_options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO, num_faces=1
)
face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(face_options)

hand_options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO, num_hands=1
)
hand_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(hand_options)

cap = cv2.VideoCapture(0)

# MOVED LANDMARKS: Shifting down from upper cheekbones directly to the lower cheek/mouth corners
LEFT_CHEEK_FLAP = 147   
RIGHT_CHEEK_FLAP = 376  
THUMB_TIP = 4
INDEX_FINGER_TIP = 8  

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int(time.time() * 1000)
    
    face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
    hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

    pinch_px = None
    is_pinching = False

    # Detect if Hand is Pinching
    if hand_result.hand_landmarks:
        hand_landmarks = hand_result.hand_landmarks[0]
        thumb = hand_landmarks[THUMB_TIP]
        index = hand_landmarks[INDEX_FINGER_TIP]
        
        pinch_dist = math.sqrt((thumb.x - index.x)**2 + (thumb.y - index.y)**2)
        if pinch_dist < 0.07: 
            is_pinching = True
            pinch_px = (int((thumb.x + index.x)/2 * w), int((thumb.y + index.y)/2 * h))

    # Process Face and apply stretch
    if face_result.face_landmarks:
        landmarks = face_result.face_landmarks[0]
        left_px = (int(landmarks[LEFT_CHEEK_FLAP].x * w), int(landmarks[LEFT_CHEEK_FLAP].y * h))
        right_px = (int(landmarks[RIGHT_CHEEK_FLAP].x * w), int(landmarks[RIGHT_CHEEK_FLAP].y * h))
        
        stretched = False

        if pinch_px is not None:
            dist_to_left = math.sqrt((pinch_px[0] - left_px[0])**2 + (pinch_px[1] - left_px[1])**2)
            dist_to_right = math.sqrt((pinch_px[0] - right_px[0])**2 + (pinch_px[1] - right_px[1])**2)
            
            # SHRUNK RADIUS to 100 so it doesn't touch your eyes!
            if is_pinching:
                if dist_to_left < dist_to_right and dist_to_left < 250:
                    frame = create_pull_effect(frame, cheek_center=left_px, target_pos=pinch_px, radius=100)
                    stretched = True
                elif dist_to_right < dist_to_left and dist_to_right < 250:
                    frame = create_pull_effect(frame, cheek_center=right_px, target_pos=pinch_px, radius=100)
                    stretched = True

        if stretched:
            cv2.putText(frame, "GOMU GOMU NO...", (30, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
        else:
            # Clean preview dots when not stretching
            cv2.circle(frame, left_px, 4, (0, 255, 0), -1)   
            cv2.circle(frame, right_px, 4, (0, 0, 255), -1)  
            if pinch_px:
                cv2.circle(frame, pinch_px, 5, (255, 255, 0), -1)

    cv2.imshow('Luffy Clean Pull Filter', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
face_landmarker.close()
hand_landmarker.close()
cv2.destroyAllWindows()
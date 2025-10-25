import cv2
import mediapipe as mp
import pyautogui

face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
cam = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()


SENSITIVITY_X = 2.5  
SENSITIVITY_Y = 2.5  
SMOOTHING = 0.3      
DEAD_ZONE = 0.02    


center_x, center_y = None, None
calibration_frames = 0
CALIBRATION_TIME = 30  


prev_screen_x, prev_screen_y = screen_w // 2, screen_h // 2

print("Eye Controller Started!")
print("Look at the center of the screen for 1 second to calibrate...")
print("Press 'q' to quit, 'r' to recalibrate")

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    frame_h, frame_w, _ = frame.shape
    landmark_points = output.multi_face_landmarks
    
    if landmark_points:
        landmarks = landmark_points[0].landmark

        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0))
            
            if id == 1:  
                if calibration_frames < CALIBRATION_TIME:
                    if center_x is None:
                        center_x, center_y = x, y
                    else:
                        center_x = (center_x + x) / 2
                        center_y = (center_y + y) / 2
                    calibration_frames += 1
                    
                    cv2.putText(frame, f"Calibrating... {calibration_frames}/{CALIBRATION_TIME}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.circle(frame, (int(center_x), int(center_y)), 10, (0, 255, 255), 2)
                
                else:
                    rel_x = (x - center_x) / frame_w
                    rel_y = (y - center_y) / frame_h
                    
                    if abs(rel_x) < DEAD_ZONE:
                        rel_x = 0
                    if abs(rel_y) < DEAD_ZONE:
                        rel_y = 0
                    
                    screen_x = screen_w // 2 + (rel_x * SENSITIVITY_X * screen_w)
                    screen_y = screen_h // 2 + (rel_y * SENSITIVITY_Y * screen_h)
                    
                    screen_x = prev_screen_x + SMOOTHING * (screen_x - prev_screen_x)
                    screen_y = prev_screen_y + SMOOTHING * (screen_y - prev_screen_y)
                    
                    screen_x = max(0, min(int(screen_x), screen_w - 1))
                    screen_y = max(0, min(int(screen_y), screen_h - 1))
                    
                    pyautogui.moveTo(screen_x, screen_y)
                    prev_screen_x, prev_screen_y = screen_x, screen_y
                    
                    cv2.putText(frame, f"Cursor: ({screen_x}, {screen_y})", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"Eye: ({x}, {y})", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"Sensitivity: {SENSITIVITY_X}x", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    cv2.circle(frame, (int(center_x), int(center_y)), 5, (255, 0, 0), 1)

        left = [landmarks[145], landmarks[159]]
        for landmark in left:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255))
        
        if (left[0].y - left[1].y) < 0.015:
            if calibration_frames >= CALIBRATION_TIME:  
                pyautogui.click()
                pyautogui.sleep(0.3)
    
    cv2.imshow("Eye Controller", frame) 
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        
        center_x, center_y = None, None
        calibration_frames = 0
        print("Recalibrating... Look at center of screen")

cam.release()
cv2.destroyAllWindows()
print("Eye Controller stopped.")

import cv2
import numpy as np
import mss
from screeninfo import get_monitors
import time

def nothing(x):
    pass

def calibration_tool():
    # 1. Setup Screen Capture
    try:
        monitor = get_monitors()[0]
        width = monitor.width
        height = monitor.height
    except:
        width = 1920
        height = 1080
    
    monitor_area = {"top": 0, "left": 0, "width": width, "height": height}
    sct = mss.mss()

    # 2. Create a Window with Sliders (Trackbars)
    cv2.namedWindow("HSV Calibrator")
    
    # Create sliders for Lower and Upper HSV ranges
    # Standard starting values for "Blue"
    cv2.createTrackbar("L - H", "HSV Calibrator", 100, 179, nothing)
    cv2.createTrackbar("L - S", "HSV Calibrator", 150, 255, nothing)
    cv2.createTrackbar("L - V", "HSV Calibrator", 50, 255, nothing)
    
    cv2.createTrackbar("U - H", "HSV Calibrator", 130, 179, nothing)
    cv2.createTrackbar("U - S", "HSV Calibrator", 255, 255, nothing)
    cv2.createTrackbar("U - V", "HSV Calibrator", 255, 255, nothing)

    print("-------------------------------------------------------")
    print("INSTRUCTIONS:")
    print("1. Open Bopl Battle and stand still with your character.")
    print("2. Adjust the sliders until ONLY your character is white.")
    print("3. Everything else (background) should be black.")
    print("4. Press 'q' to quit and print the values.")
    print("-------------------------------------------------------")

    while True:
        # Capture screen
        img = np.array(sct.grab(monitor_area))
        img = img[:, :, :3] # Remove alpha
        
        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Get current positions of all trackbars
        l_h = cv2.getTrackbarPos("L - H", "HSV Calibrator")
        l_s = cv2.getTrackbarPos("L - S", "HSV Calibrator")
        l_v = cv2.getTrackbarPos("L - V", "HSV Calibrator")
        
        u_h = cv2.getTrackbarPos("U - H", "HSV Calibrator")
        u_s = cv2.getTrackbarPos("U - S", "HSV Calibrator")
        u_v = cv2.getTrackbarPos("U - V", "HSV Calibrator")

        lower_bound = np.array([l_h, l_s, l_v])
        upper_bound = np.array([u_h, u_s, u_v])

        # Create the mask
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Visualize
        # Resize for easier viewing on desktop
        preview = cv2.resize(mask, (640, 360))
        cv2.imshow("HSV Calibrator", preview)

        if cv2.waitKey(1) == ord('q'):
            print("\n\n### COPY THESE VALUES INTO train_bopl.py ###")
            print(f"self.lower_color = np.array([{l_h}, {l_s}, {l_v}])")
            print(f"self.upper_color = np.array([{u_h}, {u_s}, {u_v}])")
            print("##############################################\n")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    calibration_tool()
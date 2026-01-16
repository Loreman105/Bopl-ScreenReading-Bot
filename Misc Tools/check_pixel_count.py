import cv2
import mss
import numpy as np
import time
from screeninfo import get_monitors

# YOUR COLORS
MY_LOWER = np.array([168, 60, 200])
MY_UPPER = np.array([173, 255, 255])

def main():
    sct = mss.mss()
    try:
        monitor = get_monitors()[0]
        monitor_area = {"top": 0, "left": 0, "width": monitor.width, "height": monitor.height}
    except:
        monitor_area = {"top": 0, "left": 0, "width": 1920, "height": 1080}

    print("--- PIXEL COUNTER TOOL ---")
    print("1. Play the game.")
    print("2. Look at the 'My Pixels' number.")
    print("3. NOTICE THE DIFFERENCE between Normal Gameplay and the WIN SCREEN.")
    print("--------------------------")
    time.sleep(3)

    while True:
        # Grab screen
        img = np.array(sct.grab(monitor_area))
        img = img[:, :, :3]

        # Count pixels
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, MY_LOWER, MY_UPPER)
        count = np.count_nonzero(mask)

        # Print with visual bar
        bar = "|" * (count // 1000) # One bar per 1000 pixels
        print(f"Pixels: {count} {bar}")
        
        time.sleep(0.1)

if __name__ == "__main__":
    main()
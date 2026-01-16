import cv2
import numpy as np
import mss
from screeninfo import get_monitors
import time


def copy_to_clipboard(text: str) -> None:
    """Copy text to the system clipboard (Windows/macOS/Linux) using stdlib tkinter."""
    try:
        import tkinter as tk

        r = tk.Tk()
        r.withdraw()
        r.clipboard_clear()
        r.clipboard_append(text)
        r.update()  # ensures it persists after the window is destroyed
        r.destroy()
    except Exception as e:
        print("[WARN] Clipboard copy failed:", e)
        print(text)

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

    # Clickable "Copy" button (drawn onto the preview image)
    BTN_X, BTN_Y, BTN_W, BTN_H = 10, 10, 140, 36
    state = {
        "l_h": 0,
        "l_s": 0,
        "l_v": 0,
        "u_h": 0,
        "u_s": 0,
        "u_v": 0,
        "copied_at": 0.0,
    }

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if (BTN_X <= x <= (BTN_X + BTN_W)) and (BTN_Y <= y <= (BTN_Y + BTN_H)):
            txt = (
                f"(({state['l_h']}, {state['l_s']}, {state['l_v']}), ({state['u_h']}, {state['u_s']}, {state['u_v']}))"
            )
            copy_to_clipboard(txt)
            state["copied_at"] = time.time()

    cv2.setMouseCallback("HSV Calibrator", on_mouse)
    
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
    print("5. Click the 'Copy' button in the preview to copy values.")
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

        state["l_h"], state["l_s"], state["l_v"] = l_h, l_s, l_v
        state["u_h"], state["u_s"], state["u_v"] = u_h, u_s, u_v

        lower_bound = np.array([l_h, l_s, l_v])
        upper_bound = np.array([u_h, u_s, u_v])

        # Create the mask
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Visualize
        # Resize for easier viewing on desktop
        preview = cv2.resize(mask, (640, 360))

        # Convert to color so we can draw UI elements (button/text)
        preview_ui = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)

        # Draw Copy button
        cv2.rectangle(preview_ui, (BTN_X, BTN_Y), (BTN_X + BTN_W, BTN_Y + BTN_H), (60, 60, 60), -1)
        cv2.rectangle(preview_ui, (BTN_X, BTN_Y), (BTN_X + BTN_W, BTN_Y + BTN_H), (255, 255, 255), 1)
        cv2.putText(preview_ui, "Copy", (BTN_X + 40, BTN_Y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show a brief "COPIED!" status after clicking
        if (time.time() - float(state["copied_at"])) < 1.2:
            cv2.putText(preview_ui, "COPIED!", (BTN_X + BTN_W + 12, BTN_Y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("HSV Calibrator", preview_ui)

        if cv2.waitKey(1) == ord('q'):
            print("\n\n### COPY THESE VALUES INTO CONFIG ###")
            print(f"(({l_h}, {l_s}, {l_v}), ({u_h}, {u_s}, {u_v}))")
            print("##############################################\n")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    calibration_tool()
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import mss
import pydirectinput
import time
import os
from screeninfo import get_monitors
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback

# --- CONFIGURATION ---
TOTAL_TIMESTEPS = 500000 
MODEL_DIR = "./models/bopl_pure"
LOG_DIR = "./logs/bopl_pure_logs"
TEMPLATE_FILE = "win_template.png"

# --- MOUSE SETTINGS ---
# Increased from 50 to 300 for much faster/farther aiming
MOUSE_STEP = 300  

# Disable failsafe to prevent crashes
pydirectinput.FAILSAFE = False

class PointTracker:
    def __init__(self, initial_points=0):
        self.points = initial_points
        self.GREEN = '\033[92m'
        self.RESET = '\033[0m'
        self.BOLD = '\033[1m'

    def add_back(self, amount, reason="Refund"):
        self.points += amount
        print(f"\n{self.GREEN}{self.BOLD}[+] {amount} POINTS ADDED ({reason}){self.RESET}")

class BoplPureEnv(gym.Env):
    def __init__(self):
        super(BoplPureEnv, self).__init__()
        
        self.tracker = PointTracker(initial_points=0)

        # --- TEMPLATE & MATCHING ---
        if os.path.exists(TEMPLATE_FILE):
            self.win_template = cv2.imread(TEMPLATE_FILE, 0)
            self.use_template = True
            self.template_h, self.template_w = self.win_template.shape
            print(f"SUCCESS: Loaded '{TEMPLATE_FILE}' for layout matching.")
        else:
            self.win_template = None
            self.use_template = False
            self.template_h, self.template_w = (0, 0)
            print("WARNING: Template not found. Using pixel count only.")

        # Kept this at 0.6 (The code you pasted had 0.8 which was broken)
        self.MATCH_THRESHOLD = 0.60

        # Screen Setup
        try:
            monitor = get_monitors()[0]
            self.width = monitor.width
            self.height = monitor.height
        except:
            self.width = 1920
            self.height = 1080
            
        self.monitor_area = {"top": 0, "left": 0, "width": self.width, "height": self.height}
        self.sct = mss.mss()

        # Spaces (11 Actions)
        self.action_space = spaces.Discrete(11)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

        # Colors (Purple/Slime)
        # Kept the WIDER range (140-180) so it sees the player correctly
        self.my_lower = np.array([140, 50, 50]) 
        self.my_upper = np.array([179, 255, 255])

        self.dashboard = np.zeros((200, 400, 3), dtype=np.uint8) 
        
        # --- INPUT STATE TRACKING ---
        self.held_keys = set()
        self.held_mouse = set()

    def step(self, action):
        # execute logic to hold/release keys
        self._apply_action_state(action)
        
        # Wait a bit (frame time)
        time.sleep(0.04) 
        
        sct_img = self.sct.grab(self.monitor_area)
        raw_frame = np.array(sct_img)[:, :, :3]
        
        # --- DEBUG CAMERA (Optional: Shows what bot sees) ---
        # hsv_debug = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2HSV)
        # mask_debug = cv2.inRange(hsv_debug, self.my_lower, self.my_upper)
        # cv2.imshow("DEBUG VIEW", cv2.resize(mask_debug, (300, 200)))
        # cv2.waitKey(1)
        # ----------------------------------------------------

        gray_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        
        reward = 0
        terminated = False
        
        # --- WIN/LOSS LOGIC ---
        match_found = False
        is_my_win = False
        match_val = 0.0

        if self.use_template:
            res = cv2.matchTemplate(gray_frame, self.win_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            match_val = max_val

            if max_val >= self.MATCH_THRESHOLD:
                match_found = True
                top_left = max_loc
                bottom_right = (top_left[0] + self.template_w, top_left[1] + self.template_h)
                roi = raw_frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                
                purple_pixels = self._count_pixels(roi, self.my_lower, self.my_upper)
                total_pixels_in_box = self.template_w * self.template_h
                
                # 10% threshold
                if purple_pixels > (total_pixels_in_box * 0.1):
                    is_my_win = True

        # --- REWARDS ---
        if match_found:
            if is_my_win:
                reward = 1000.0
                terminated = True
                self.tracker.add_back(1000, "VICTORY")
            else:
                reward = -1000.0
                terminated = True
                print(">>> DEFEAT (Enemy Win)")

        # Using 50 pixel threshold for death detection
        elif self._count_pixels(raw_frame, self.my_lower, self.my_upper) < 50:
            reward = -1000.0
            terminated = True
            print(">>> DEFEAT (Died / Not detected)")

        else:
            reward = -0.01

        self._update_dashboard(reward, action, match_val, is_my_win if match_found else None)

        obs = self._process_obs(raw_frame)
        return obs, reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. Release ALL held keys/buttons from previous episode
        self._release_all_inputs()
        
        # Center Mouse
        pydirectinput.moveTo(self.width // 2, self.height // 2)
        
        # Restart Level Macro
        pydirectinput.press('space')
        time.sleep(0.2) 
        pydirectinput.press('space') 
        time.sleep(1.5) 
        
        sct_img = self.sct.grab(self.monitor_area)
        raw_frame = np.array(sct_img)[:, :, :3]
        return self._process_obs(raw_frame), {}

    def _process_obs(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (84, 84))
        return np.expand_dims(resized, axis=2)

    def _count_pixels(self, img, lower, upper):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        return np.count_nonzero(mask)

    def _update_dashboard(self, reward, action, match_val, is_my_win):
        self.dashboard[:] = (40, 40, 40)
        
        if reward > 0: color = (0, 255, 0); sign = "+"
        elif reward < -0.1: color = (0, 0, 255); sign = ""
        else: color = (200, 200, 200); sign = ""
        
        cv2.putText(self.dashboard, "AI MONITOR", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(self.dashboard, f"Last: {sign}{reward:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        match_color = (0, 255, 0) if match_val > self.MATCH_THRESHOLD else (0, 255, 255)
        cv2.putText(self.dashboard, f"Match: {match_val:.2f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, match_color, 1)

        # Action Info
        actions = ["WAIT", "LEFT (A)", "RIGHT (D)", "JUMP", "L-CLICK", "R-CLICK", "M-CLICK", "AIM UP", "AIM DOWN", "AIM LEFT", "AIM RIGHT"]
        act_name = actions[action] if action < len(actions) else "???"
        
        # Show what is currently HELD
        held_text = "HELD: " + ", ".join(list(self.held_keys) + list(self.held_mouse))
        cv2.putText(self.dashboard, held_text, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(self.dashboard, f"Action: {act_name}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("AI Dashboard", self.dashboard)
        cv2.waitKey(1)

    # --- UPDATED: SMART HOLDING + FARTHER MOUSE ---
    def _apply_action_state(self, action):
        """
        Updates the State of keys and Mouse Position.
        """
        
        # Define what keys/mouse SHOULD be down for each action
        target_keys = set()
        target_mouse = set()

        # Mapping
        if action == 1: target_keys.add('a')           # Left
        elif action == 2: target_keys.add('d')         # Right
        elif action == 3: target_keys.add('space')     # Jump
        
        elif action == 4: target_mouse.add('left')     # Click
        elif action == 5: target_mouse.add('right')    # Click
        elif action == 6: target_mouse.add('middle')   # Click
        
        # Mouse Aiming (Instant, not held)
        # NOW USES 'MOUSE_STEP' (300px)
        elif action == 7: pydirectinput.moveRel(0, -MOUSE_STEP)
        elif action == 8: pydirectinput.moveRel(0, MOUSE_STEP)
        elif action == 9: pydirectinput.moveRel(-MOUSE_STEP, 0)
        elif action == 10: pydirectinput.moveRel(MOUSE_STEP, 0)

        # 1. Release keys that are no longer in target
        for k in list(self.held_keys):
            if k not in target_keys:
                pydirectinput.keyUp(k)
                self.held_keys.remove(k)
        
        for b in list(self.held_mouse):
            if b not in target_mouse:
                pydirectinput.mouseUp(button=b)
                self.held_mouse.remove(b)

        # 2. Press keys that are new
        for k in target_keys:
            if k not in self.held_keys:
                pydirectinput.keyDown(k)
                self.held_keys.add(k)
                
        for b in target_mouse:
            if b not in self.held_mouse:
                pydirectinput.mouseDown(button=b)
                self.held_mouse.add(b)

    def _release_all_inputs(self):
        """Emergency release of everything."""
        for k in list(self.held_keys):
            pydirectinput.keyUp(k)
        self.held_keys.clear()
        
        for b in list(self.held_mouse):
            pydirectinput.mouseUp(button=b)
        self.held_mouse.clear()

def main():
    if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)

    env = BoplPureEnv()
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)

    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.0003, ent_coef=0.01)

    print("--- TRAINING STARTED (FARTHER MOUSE ENABLED) ---")
    time.sleep(5)
    
    checkpoint_callback = CheckpointCallback(save_freq=20000, save_path=MODEL_DIR, name_prefix='bopl_pure')
    
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)
        model.save(f"{MODEL_DIR}/bopl_final")
    except KeyboardInterrupt:
        # Ensure we release keys if user presses Ctrl+C
        pydirectinput.keyUp('a')
        pydirectinput.keyUp('d')
        pydirectinput.keyUp('space')
        pydirectinput.mouseUp(button='left')
        pydirectinput.mouseUp(button='right')
        model.save(f"{MODEL_DIR}/bopl_interrupted")
        
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
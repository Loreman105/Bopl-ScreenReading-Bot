"""Gymnasium environment for Bopl with screen-reading + direct input."""

from __future__ import annotations

import math
import os
import time

import cv2
import gymnasium as gym
import mss
import numpy as np
import pydirectinput
from gymnasium import spaces
from screeninfo import get_monitors

from . import config
from .ui import EnvUI, PointTracker
from .win32 import clip_cursor, key_down, release_cursor


class BoplPureEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        cv2.setNumThreads(0)
        pydirectinput.FAILSAFE = False

        self.tracker = PointTracker(initial_points=0)

        # --- TEMPLATE & MATCHING ---
        if os.path.exists(config.TEMPLATE_FILE):
            self.win_template = cv2.imread(config.TEMPLATE_FILE, 0)
            self.use_template = True
            self.template_h, self.template_w = self.win_template.shape
            print(f"SUCCESS: Loaded '{config.TEMPLATE_FILE}' for layout matching.")
        else:
            self.win_template = None
            self.use_template = False
            self.template_h, self.template_w = (0, 0)
            print("WARNING: Template not found. Using pixel count only.")

        self.MATCH_THRESHOLD = config.MATCH_THRESHOLD

        # Monitor Setup
        monitors = get_monitors()
        game = monitors[config.GAME_MONITOR_INDEX]
        dbg = monitors[config.DEBUG_MONITOR_INDEX]
        self.game_x = game.x
        self.game_y = game.y
        self.width = game.width
        self.height = game.height
        self.monitor_area = {"top": game.y, "left": game.x, "width": game.width, "height": game.height}
        self.debug_origin = (dbg.x, dbg.y)
        self.sct = mss.mss()
        self.ui = EnvUI(debug_origin=self.debug_origin)

        # Actions
        # 0: movement  (0=none, 1=left, 2=right, 3=jump)
        # 1: attack    (0=none, 1=left click, 2=right click, 3=middle click)
        # 2: aim angle (0..359)
        self.action_space = spaces.MultiDiscrete([4, 4, 360])
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

        # Colors (YOUR player)
        self.my_lower = np.array([16, 222, 205])
        self.my_upper = np.array([23, 233, 255])

        # --- INPUT STATE TRACKING ---
        self.held_keys: set[str] = set()
        self.held_mouse: set[str] = set()

        # Player tracking
        self.prev_my_center = None
        self._prev_gray_for_motion = None
        self.missing_player_frames = 0
        self.PLAYER_MIN_AREA = 120
        self.TRAIL_ASPECT_CUTOFF = 6.0
        self.MISSING_FRAMES_FOR_DEATH = 8

        # Motion-based disambiguation (player vs same-colored platforms)
        self.MOTION_DIFF_THRESHOLD = 18
        self.MOTION_DILATE_ITERS = 1

        # Pause/resume state
        self.paused = False
        self._f8_was_down = False
        self._f9_was_down = False

    # ---------- cursor clip ----------
    def _clip_cursor_enable(self): # Enables cursor clipping.
        clip_cursor(
            self.game_x,
            self.game_y,
            self.game_x + self.width,
            self.game_y + self.height,
        )

    def _clip_cursor_disable(self): # Disables cursor clipping.
        release_cursor()

    # ---------- pause handling ----------
    def _update_pause_hotkeys(self): # Checks F8/F9 keys to pause/resume the environment.
        f8 = key_down(config.VK_F8)
        f9 = key_down(config.VK_F9)

        if f8 and not self._f8_was_down:
            self.paused = True
            print("=== PAUSED (F8) ===")
            self._release_all_inputs()

        if f9 and not self._f9_was_down:
            self.paused = False
            print("=== RESUMED (F9) ===")

        self._f8_was_down = f8
        self._f9_was_down = f9

    def _block_while_paused(self): # Blocks execution until unpaused.
        self._clip_cursor_disable()
        self._release_all_inputs()

        while self.paused:
            self._update_pause_hotkeys()
            self._release_all_inputs()
            cv2.waitKey(1)
            time.sleep(0.05)

        self.paused = False
        self._clip_cursor_enable()

    # ---------------------------
    # ENV STEP
    # ---------------------------
    def step(self, action): # Executes one step of action in the environment.
        self._update_pause_hotkeys()
        if self.paused:
            self._block_while_paused()

        self._apply_action_state(action)
        time.sleep(0.04)

        sct_img = self.sct.grab(self.monitor_area)
        raw_frame = np.array(sct_img)[:, :, :3]
        gray_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)

        reward = 0.0
        terminated = False

        match_found = False
        is_my_win = False
        match_val = 0.0

        if self.use_template:
            res = cv2.matchTemplate(gray_frame, self.win_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            match_val = max_val

            if max_val >= self.MATCH_THRESHOLD:
                match_found = True
                top_left = max_loc
                bottom_right = (top_left[0] + self.template_w, top_left[1] + self.template_h)
                roi = raw_frame[top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]]

                player_color_pixels = self._count_pixels(roi, self.my_lower, self.my_upper)
                total_pixels_in_box = self.template_w * self.template_h

                if player_color_pixels > (total_pixels_in_box * 0.1):
                    is_my_win = True

        if match_found:
            if is_my_win:
                reward = 1000.0
                terminated = True
                self.tracker.add_back(1000, "VICTORY")
            else:
                reward = -1000.0
                terminated = True
                print(">>> DEFEAT (Enemy Win)")
        else:
            player_center, _, player_blob_mask = self._detect_player_blob(raw_frame, gray_frame, debug=True)
            self.ui.show_player_blob(player_blob_mask)

            if player_center is None:
                self.missing_player_frames += 1
            else:
                self.missing_player_frames = 0
                self.prev_my_center = player_center

            if self.missing_player_frames >= self.MISSING_FRAMES_FOR_DEATH:
                reward = -1000.0
                terminated = True
                print(">>> DEFEAT (Died / Player blob not found)")
            else:
                reward = -0.01

        self.ui.update_dashboard(
            reward=float(reward),
            action=action,
            match_val=float(match_val),
            match_threshold=float(self.MATCH_THRESHOLD),
            paused=bool(self.paused),
            missing_player_frames=int(self.missing_player_frames),
            held_keys=self.held_keys,
            held_mouse=self.held_mouse,
        )

        obs = self._process_obs(raw_frame)
        self._clamp_cursor_to_monitor()
        self.ui.show_agent_view(obs)
        return obs, float(reward), terminated, False, {}

    # ---------------------------
    # RESET
    # ---------------------------
    def reset(self, seed=None, options=None): # Creates a new blank episode. New episode starts after victory or defeat.
        super().reset(seed=seed)

        self._update_pause_hotkeys()
        if self.paused:
            self._block_while_paused()

        self._release_all_inputs()
        self.prev_my_center = None
        self._prev_gray_for_motion = None
        self.missing_player_frames = 0

        center_x = self.game_x + self.width // 2
        center_y = self.game_y + self.height // 2
        pydirectinput.moveTo(center_x, center_y)

        self._clip_cursor_enable()

        # Restart Level Macro
        pydirectinput.press("space")
        time.sleep(0.2)
        pydirectinput.press("space")
        time.sleep(1.5)

        sct_img = self.sct.grab(self.monitor_area)
        raw_frame = np.array(sct_img)[:, :, :3]
        return self._process_obs(raw_frame), {}

    # ---------------------------
    # OBS / VISION
    # ---------------------------
    def _process_obs(self, img): # Image becomes 84x84 grayscale, added to NN input layer
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (84, 84))
        return np.expand_dims(resized, axis=2)

    def _count_pixels(self, img, lower, upper): # Counts the number of pixels in an image within a color range
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        return int(np.count_nonzero(mask))

    def _detect_player_blob(self, raw_frame, gray_frame=None, *, debug: bool = False): # Detects the player blob in the frame using color thresholding and connected components.
        if gray_frame is None:
            gray_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)

        hsv = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv, self.my_lower, self.my_upper)
        color_mask = cv2.medianBlur(color_mask, 5)

        # If the map has platforms matching player HSV, intersect with motion to bias toward the moving player.
        mask = color_mask
        if self._prev_gray_for_motion is not None:
            diff = cv2.absdiff(gray_frame, self._prev_gray_for_motion)
            _, motion = cv2.threshold(diff, int(self.MOTION_DIFF_THRESHOLD), 255, cv2.THRESH_BINARY)
            motion = cv2.medianBlur(motion, 5)
            motion = cv2.dilate(motion, np.ones((5, 5), np.uint8), iterations=int(self.MOTION_DILATE_ITERS))

            moving_color = cv2.bitwise_and(color_mask, motion)
            if int(np.count_nonzero(moving_color)) >= int(self.PLAYER_MIN_AREA):
                mask = moving_color

        # Update reference for next frame (do this before early returns)
        self._prev_gray_for_motion = gray_frame

        num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        best_center = None
        best_area = None
        best_score = None
        best_label = None

        for i in range(1, num):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < self.PLAYER_MIN_AREA:
                continue

            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])

            cx, cy = centroids[i]
            cx = float(cx)
            cy = float(cy)

            aspect = float(w) / float(max(h, 1))
            if aspect > self.TRAIL_ASPECT_CUTOFF and area < (self.PLAYER_MIN_AREA * 8):
                continue

            # Reject rectangle-like blobs (ice platforms/walls often show up as clean rectangles)
            try:
                component_mask = (labels == i).astype(np.uint8) * 255
                contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cnt = max(contours, key=cv2.contourArea)
                    peri = float(cv2.arcLength(cnt, True))
                    if peri > 0.0:
                        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                        rectangularity = float(area) / float(max(w * h, 1))

                        # Typical platforms: 4 corners + high fill + long/skinny aspect
                        if (
                            len(approx) == 4
                            and rectangularity > 0.75
                            and (aspect > 1.8 or aspect < 0.55)
                            and area > (self.PLAYER_MIN_AREA * 3)
                        ):
                            continue
            except Exception:
                # If contour logic fails, fall back to existing heuristics
                pass

            if self.prev_my_center is not None:
                dx = cx - self.prev_my_center[0]
                dy = cy - self.prev_my_center[1]
                dist = float((dx * dx + dy * dy) ** 0.5)
            else:
                dist = 0.0

            score = dist - 0.002 * area

            if best_score is None or score < best_score:
                best_score = score
                best_center = (cx, cy)
                best_area = area
                best_label = i

        if not debug:
            return best_center, best_area

        if best_label is None:
            selected = np.zeros_like(mask)
        else:
            selected = (labels == int(best_label)).astype(np.uint8) * 255

        return best_center, best_area, selected

    # ---------------------------
    # AIM
    # ---------------------------
    
    # Moves the mouse cursor to a position relative to the player's position based on angle and radius. (Aiming system)
    def _aim_cursor_relative_to_player(self, angle_deg: float, radius: int = config.MOUSE_AIM_RADIUS) -> None: 
        if self.prev_my_center is None:
            return

        player_x = self.game_x + float(self.prev_my_center[0])
        player_y = self.game_y + float(self.prev_my_center[1])

        # 0° = right, 90° = down, 180° = left, 270° = up (screen coords: +y is down)
        theta = math.radians(angle_deg)
        dx = math.cos(theta) * float(radius)
        dy = math.sin(theta) * float(radius)

        target_x = int(round(player_x + dx))
        target_y = int(round(player_y + dy))
        pydirectinput.moveTo(target_x, target_y)
        self._clamp_cursor_to_monitor()

    # ---------------------------
    def _clamp_cursor_to_monitor(self): # Ensures the mouse cursor stays within the game monitor area.
        x, y = pydirectinput.position()
        min_x = self.monitor_area["left"]
        min_y = self.monitor_area["top"]
        max_x = min_x + self.monitor_area["width"] - 1
        max_y = min_y + self.monitor_area["height"] - 1

        cx = max(min_x, min(int(x), max_x))
        cy = max(min_y, min(int(y), max_y))

        if cx != x or cy != y:
            pydirectinput.moveTo(cx, cy)

    # ---------------------------
    # INPUT
    # ---------------------------
    def _apply_action_state(self, action): # Applies the action to the input state (keyboard/mouse).
        target_keys = set()
        target_mouse = set()

        try:
            move_action = int(action[0])
            atk_action = int(action[1])
            aim_action = int(action[2])
        except Exception:
            move_action, atk_action, aim_action = 0, 0, 0

        if move_action == 1:
            target_keys.add("a")
        elif move_action == 2:
            target_keys.add("d")
        elif move_action == 3:
            target_keys.add("space")

        if atk_action == 1:
            target_mouse.add("left")
        elif atk_action == 2:
            target_mouse.add("right")
        elif atk_action == 3:
            target_mouse.add("middle")

        aim_deg = float((aim_action % 360))
        self._aim_cursor_relative_to_player(aim_deg, radius=config.MOUSE_AIM_RADIUS)

        for k in list(self.held_keys):
            if k not in target_keys:
                pydirectinput.keyUp(k)
                self.held_keys.remove(k)

        for b in list(self.held_mouse):
            if b not in target_mouse:
                pydirectinput.mouseUp(button=b)
                self.held_mouse.remove(b)

        for k in target_keys:
            if k not in self.held_keys:
                pydirectinput.keyDown(k)
                self.held_keys.add(k)

        for b in target_mouse:
            if b not in self.held_mouse:
                pydirectinput.mouseDown(button=b)
                self.held_mouse.add(b)

    def _release_all_inputs(self): # Releases all held keyboard and mouse inputs.
        for k in list(self.held_keys):
            pydirectinput.keyUp(k)
        self.held_keys.clear()

        for b in list(self.held_mouse):
            pydirectinput.mouseUp(button=b)
        self.held_mouse.clear()

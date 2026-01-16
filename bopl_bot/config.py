"""Centralized configuration constants.

Keep constants here so training/env code stays clean and imports are stable.
"""

# --- TRAINING ---
TOTAL_TIMESTEPS = 50_000_000_000_000_000_000  # effectively infinite
MODEL_DIR = "./models/bopl_pure"
LOG_DIR = "./logs/bopl_pure_logs"

# --- VISION ---
TEMPLATE_FILE = "win_template.png"
MATCH_THRESHOLD = 0.60

# --- MOUSE ---
MOUSE_STEP = 300
MOUSE_AIM_RADIUS = 300

# --- HOTKEYS (Windows virtual-key codes) ---
# F8 = 0x77, F9 = 0x78
VK_F8 = 0x77
VK_F9 = 0x78

# --- MONITORS ---
GAME_MONITOR_INDEX = 0
DEBUG_MONITOR_INDEX = 1

# --- COLORS (HSV) ---
ICE_BACKGROUND = ((101, 169, 146), (106, 189, 167)) # Background colors don't include the full background, just recognizable parts.
BOPL_ORANGE = ((16, 223, 248), (20, 227, 252)) # Colors of characters also detect some UI elements
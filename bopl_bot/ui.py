"""UI utilities (OpenCV windows + simple terminal/UI helpers).

This module is intentionally side-effect free until you call `ensure_windows()`
/`update_dashboard()`/`show_agent_view()`.
"""

from __future__ import annotations

import cv2
import numpy as np


class PointTracker:
    def __init__(self, initial_points=0):
        self.points = initial_points
        self.GREEN = "\033[92m"
        self.RESET = "\033[0m"
        self.BOLD = "\033[1m"

    def add_back(self, amount, reason="Refund"):
        self.points += amount
        print(f"\n{self.GREEN}{self.BOLD}[+] {amount} POINTS ADDED ({reason}){self.RESET}")


class EnvUI:
    """Owns the OpenCV windows created by the environment."""

    DASHBOARD_TITLE = "AI Dashboard"
    AGENT_VIEW_TITLE = "AGENT VIEW (exact obs)"
    PLAYER_BLOB_TITLE = "PLAYER BLOB (mask)"

    def __init__(self, debug_origin: tuple[int, int]):
        self.debug_origin = debug_origin
        self._windows_initialized = False
        self.dashboard = np.zeros((200, 630, 3), dtype=np.uint8)

    def ensure_windows(self) -> None:
        if self._windows_initialized:
            return

        dx, dy = self.debug_origin

        cv2.namedWindow(self.DASHBOARD_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.DASHBOARD_TITLE, 720, 260)
        cv2.moveWindow(self.DASHBOARD_TITLE, dx + 50, dy + 50)

        cv2.namedWindow(self.AGENT_VIEW_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.AGENT_VIEW_TITLE, 400, 400)
        cv2.moveWindow(self.AGENT_VIEW_TITLE, dx + 50, dy + 350)

        cv2.namedWindow(self.PLAYER_BLOB_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.PLAYER_BLOB_TITLE, 400, 400)
        cv2.moveWindow(self.PLAYER_BLOB_TITLE, dx + 470, dy + 350)

        self._windows_initialized = True

    def update_dashboard(
        self,
        *,
        reward: float,
        action,
        match_val: float,
        match_threshold: float,
        paused: bool,
        missing_player_frames: int,
        held_keys: set[str],
        held_mouse: set[str],
    ) -> None:
        self.ensure_windows()
        self.dashboard[:] = (40, 40, 40)

        if reward > 0:
            color = (0, 255, 0)
            sign = "+"
        elif reward < -0.1:
            color = (0, 0, 255)
            sign = ""
        else:
            color = (200, 200, 200)
            sign = ""

        cv2.putText(self.dashboard, "AI MONITOR", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(self.dashboard, f"Last: {sign}{reward:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        match_color = (0, 255, 0) if match_val > match_threshold else (0, 255, 255)
        cv2.putText(self.dashboard, f"Match: {match_val:.2f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, match_color, 1)

        move_names = ["MOVE: NONE ", "MOVE: LEFT ", "MOVE: RIGHT", "MOVE: JUMP "]
        atk_names = ["ATK: NONE   ", "ATK: L-CLICK", "ATK: R-CLICK", "ATK: M-CLICK"]

        try:
            move_a = int(action[0])
            atk_a = int(action[1])
            aim_a = int(action[2])
        except Exception:
            move_a, atk_a, aim_a = 0, 0, 0

        aim_deg_display = (aim_a % 360) + 1
        act_name = f"{move_names[move_a]} | {atk_names[atk_a]}\t |AIM: {aim_deg_display}"

        held_text = "HELD: " + ", ".join(list(held_keys) + list(held_mouse))
        cv2.putText(self.dashboard, held_text, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(self.dashboard, f"Action: {act_name}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.putText(
            self.dashboard,
            f"Missing: {missing_player_frames}",
            (260, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
        )
        cv2.putText(
            self.dashboard,
            f"Paused: {paused} (F8/F9)",
            (260, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
        )

        cv2.imshow(self.DASHBOARD_TITLE, self.dashboard)
        cv2.waitKey(1)

    def show_agent_view(self, obs: np.ndarray) -> None:
        """Show exactly what the policy receives (84x84x1)."""
        self.ensure_windows()
        view = obs[:, :, 0]
        view_big = cv2.resize(view, (336, 336), interpolation=cv2.INTER_NEAREST)
        cv2.imshow(self.AGENT_VIEW_TITLE, view_big)
        cv2.waitKey(1)

    def show_player_blob(self, blob_mask: np.ndarray) -> None:
        """Show the selected player blob mask (binary uint8)."""
        self.ensure_windows()

        if blob_mask is None or blob_mask.size == 0:
            view = np.zeros((84, 84), dtype=np.uint8)
        else:
            view = blob_mask
            if view.ndim == 3:
                view = view[:, :, 0]
            if view.dtype != np.uint8:
                view = view.astype(np.uint8)

        view_big = cv2.resize(view, (336, 336), interpolation=cv2.INTER_NEAREST)
        cv2.imshow(self.PLAYER_BLOB_TITLE, view_big)
        cv2.waitKey(1)

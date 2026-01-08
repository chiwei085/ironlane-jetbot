import time

from .utils import clamp11


def _is_fresh(ts, max_age_s):
    """Check if a timestamp is within a maximum age.

    Args:
        ts: Timestamp (seconds).
        max_age_s: Maximum allowed age in seconds.

    Returns:
        True if ts is within max_age_s of now; otherwise False.
    """
    if ts <= 0.0:
        return False
    return (time.time() - float(ts)) <= float(max_age_s)


def _apply_deadband(x, deadband):
    """Zero small values around 0.0 within a deadband.

    Args:
        x: Input value.
        deadband: Symmetric deadband threshold.

    Returns:
        0.0 if |x| < deadband, else x.
    """
    if deadband <= 0.0:
        return x
    if -deadband < x < deadband:
        return 0.0
    return x


class IronLaneController:
    """Fuse steer_xy, lane, and yolo to produce left/right speeds.

    Inputs:
    - steer_xy: dict {"x": float, "y": float}
    - lane: dict {"offset": [-1,1], "confidence": [0,1]}
    - yolo: policy dict from yolo_policy.yolo_outputs_to_policy (optional)

    Output: (left_speed, right_speed) in [-1, 1].
    """

    def __init__(
        self,
        base_speed=0.35,
        slow_speed=0.20,
        stop_speed=0.0,
        # steering PD + bias (old sliders)
        steering_kp=1.0,
        steering_kd=0.0,
        steering_bias=0.0,
        # optional post gain for mixing / limit
        steer_gain=1.0,
        lane_gain=0.35,
        lane_conf_min=0.15,
        steer_max_age_s=0.25,
        lane_max_age_s=0.25,
        yolo_max_age_s=0.50,
        steer_deadband=0.03,
        lane_deadband=0.03,
        steer_limit=1.0,
        # geometry for y transform: (y_center - y) / y_scale
        y_center=0.5,
        y_scale=2.0,
        avoid_speed_w=0.5,
    ):
        """Initialize controller parameters and internal state."""
        self.base_speed = float(base_speed)
        self.slow_speed = float(slow_speed)
        self.stop_speed = float(stop_speed)

        self.steering_kp = float(steering_kp)
        self.steering_kd = float(steering_kd)
        self.steering_bias = float(steering_bias)

        self.steer_gain = float(steer_gain)
        self.lane_gain = float(lane_gain)
        self.lane_conf_min = float(lane_conf_min)

        self.steer_max_age_s = float(steer_max_age_s)
        self.lane_max_age_s = float(lane_max_age_s)
        self.yolo_max_age_s = float(yolo_max_age_s)

        self.steer_deadband = float(steer_deadband)
        self.lane_deadband = float(lane_deadband)

        self.steer_limit = float(steer_limit)
        self.avoid_speed_w = float(avoid_speed_w)

        self.y_center = float(y_center)
        self.y_scale = float(y_scale) if float(y_scale) != 0.0 else 1.0

        self._angle_last = 0.0

        self._yolo_hold_kind = None
        self._yolo_until_ts = 0.0
        self.yolo_ignore_after_hold_s = 2.0
        self._yolo_ignore_until_ts = 0.0
        self._yolo_resume_speed = None
        self._avoid_turn = 0.0
        self._avoid_strength = 0.0

    def reset(self):
        """Reset stateful steering history."""
        self._angle_last = 0.0

    def _steer_from_xy(self, xy):
        """Compute steering command from steer_xy input.

        Args:
            xy: Dict with keys "x" and "y".

        Returns:
            Tuple (steer_cmd, angle, y_term, pd_term).
        """
        try:
            x = float(xy.get("x", 0.0))
            y_raw = float(xy.get("y", 0.0))
        except Exception:
            return 0.0, 0.0, 0.0, 0.0

        y_term = (self.y_center - y_raw) / self.y_scale

        import math

        angle = math.atan2(x, y_term)

        d_angle = angle - self._angle_last
        self._angle_last = angle

        pd = (angle * self.steering_kp) + (d_angle * self.steering_kd)
        steer_cmd = pd + self.steering_bias

        steer_cmd = clamp11(steer_cmd)
        steer_cmd = _apply_deadband(steer_cmd, self.steer_deadband)

        return steer_cmd, angle, y_term, pd

    def _update_yolo_hold(self, yolo, yolo_ts, current_speed):
        """Update YOLO-based hold/avoid state.

        Args:
            yolo: Policy dict from yolo_outputs_to_policy, or None.
            yolo_ts: Timestamp for the yolo payload.
            current_speed: Current base speed used for resume.

        Returns:
            Tuple (hold_state, resume_speed).
        """
        now = time.time()

        if self._yolo_hold_kind is not None and now >= self._yolo_until_ts:
            # hold ended
            prev_kind = self._yolo_hold_kind
            resume = self._yolo_resume_speed

            if prev_kind == "stop":
                self._yolo_ignore_until_ts = now + float(self.yolo_ignore_after_hold_s)

            self._yolo_hold_kind = None
            self._yolo_until_ts = 0.0
            self._yolo_resume_speed = None
            self._avoid_turn = 0.0
            self._avoid_strength = 0.0
            return ("ended", resume)

        if self._yolo_hold_kind is not None:
            return ("holding", None)

        if (
            (yolo is None)
            or (not _is_fresh(yolo_ts, self.yolo_max_age_s))
            or (not isinstance(yolo, dict))
        ):
            return ("none", None)

        kind = yolo.get("kind")

        if now < self._yolo_ignore_until_ts and kind in ("stop", "halt"):
            return ("ignored", None)

        # permanent halt
        if kind == "halt":
            self._yolo_hold_kind = "halt"
            self._yolo_until_ts = float("inf")
            return ("halt", None)

        until_ts = yolo.get("until_ts")
        try:
            until_ts = float(until_ts) if until_ts is not None else 0.0
        except Exception:
            until_ts = 0.0

        if kind in ("stop", "slow", "avoid") and until_ts > now:
            self._yolo_hold_kind = str(kind)
            self._yolo_until_ts = until_ts

            if kind == "stop":
                # remember "pre-stop speed" the first time we enter stop
                if self._yolo_resume_speed is None:
                    self._yolo_resume_speed = float(current_speed)
            elif kind == "slow":
                if self._yolo_resume_speed is None:
                    self._yolo_resume_speed = float(current_speed)
            elif kind == "avoid":
                try:
                    self._avoid_turn = float(yolo.get("avoid_turn", 0.0))
                    self._avoid_strength = float(yolo.get("avoid_strength", 0.0))
                except Exception:
                    self._avoid_turn = 0.0
                    self._avoid_strength = 0.0

            return ("latched", None)

        return ("none", None)

    def compute(self, resulthub):
        """Compute left/right speeds from the latest hub results.

        Args:
            resulthub: ResultHub instance.

        Returns:
            Tuple (left_speed, right_speed, debug_dict).
        """
        now = time.time()

        xy, xy_ts = resulthub.get("steer_xy", default=None)
        lane, lane_ts = resulthub.get("lane", default=None)
        yolo, yolo_ts = resulthub.get("yolo", default=None)

        hold_state, resume_speed = self._update_yolo_hold(yolo, yolo_ts, self.base_speed)

        # hard stop forever
        if self._yolo_hold_kind == "halt":
            return (
                0.0,
                0.0,
                {
                    "note": "halt",
                    "yolo": yolo,
                    "yolo_hold_kind": self._yolo_hold_kind,
                    "yolo_until_ts": self._yolo_until_ts,
                },
            )

        # decide speed
        if self._yolo_hold_kind == "stop":
            lspeed = self.stop_speed
            rspeed = self.stop_speed
        elif self._yolo_hold_kind == "slow":
            lspeed = self.slow_speed
            rspeed = self.slow_speed
        elif self._yolo_hold_kind == "avoid":
            if self._avoid_turn > 0:
                lspeed = self.base_speed * self.avoid_speed_w
                rspeed = self.base_speed
            else:
                lspeed = self.base_speed
                rspeed = self.base_speed * self.avoid_speed_w

        else:
            # no active hold; if just ended, optionally resume the remembered speed
            if resume_speed is not None:
                try:
                    lspeed = float(resume_speed)
                    rspeed = float(resume_speed)
                except Exception:
                    lspeed = self.base_speed
                    rspeed = self.base_speed
            else:
                lspeed = self.base_speed
                rspeed = self.base_speed

        # If stopping, return early (still report debug)
        if min(lspeed, rspeed) == self.stop_speed:
            return (
                0.0,
                0.0,
                {
                    "speed": self.stop_speed,
                    "steer_xy": xy,
                    "lane": lane,
                    "yolo": yolo,
                    "yolo_hold_kind": self._yolo_hold_kind,
                    "yolo_until_ts": self._yolo_until_ts,
                    "note": "stop_hold" if self._yolo_hold_kind == "stop" else "stop",
                },
            )

        # ===== freshness gating =====
        xy_ok = (xy is not None) and _is_fresh(xy_ts, self.steer_max_age_s) and isinstance(xy, dict)
        lane_ok = (lane is not None) and _is_fresh(lane_ts, self.lane_max_age_s)

        # ===== steer (PD+bias) from resnet xy =====
        steer_cmd = 0.0
        angle = 0.0
        y_term = 0.0
        pd = 0.0
        if xy_ok:
            steer_cmd, angle, y_term, pd = self._steer_from_xy(xy)

        # ===== lane offset =====
        lane_offset = 0.0
        lane_conf = 0.0
        if lane_ok and isinstance(lane, dict):
            try:
                lane_offset = float(lane.get("offset", 0.0))
                lane_conf = float(lane.get("confidence", 0.0))
            except Exception:
                lane_offset = 0.0
                lane_conf = 0.0

        lane_used = 0.0
        if lane_ok and (lane_conf >= self.lane_conf_min):
            lane_used = _apply_deadband(clamp11(lane_offset), self.lane_deadband)

        # ===== fuse steer + lane =====
        lane_gain = self.lane_gain

        if self._yolo_hold_kind == "avoid":
            lane_gain = lane_gain * 0.30

        fused = (self.steer_gain * steer_cmd) + (lane_gain * lane_used)

        # ===== add avoid bias (NEW) =====
        avoid_bias = 0.0
        if self._yolo_hold_kind == "avoid":
            # avoid_turn: [-1,1], avoid_strength: [-1,1] or [0,1]
            avoid_bias = clamp11(self._avoid_turn) * clamp11(self._avoid_strength) * 0.2
            fused += avoid_bias

        # ===== clamp/limit =====
        fused = clamp11(fused)
        if abs(fused) > self.steer_limit:
            fused = self.steer_limit if fused > 0.0 else -self.steer_limit

        # ===== differential drive =====
        left = clamp11(lspeed + fused)
        right = clamp11(rspeed - fused)

        dbg = {
            "speed": min(lspeed, rspeed),
            "steer_xy": xy,
            "steer_xy_ts_age": (now - xy_ts) if xy_ts else None,
            "angle": angle,
            "y_term": y_term,
            "pd": pd,
            "steer_cmd": steer_cmd,
            "lane": lane,
            "lane_ts_age": (now - lane_ts) if lane_ts else None,
            "lane_used": lane_used,
            "lane_gain_used": lane_gain,
            "avoid_bias": avoid_bias,
            "fused": fused,
            "left": left,
            "right": right,
            "yolo": yolo,
            "yolo_ts_age": (now - yolo_ts) if yolo_ts else None,
            "yolo_hold_kind": self._yolo_hold_kind,
            "yolo_until_ts": self._yolo_until_ts,
            "hold_state": hold_state,
            "yolo_ignore_left": max(0.0, self._yolo_ignore_until_ts - now),
        }
        return left, right, dbg

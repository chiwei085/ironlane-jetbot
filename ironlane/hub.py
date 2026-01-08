import threading
import time


class FrameHub:
    """Latest-frame hub for producer/consumer frame exchange."""

    def __init__(self):
        """Initialize hub state."""
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)

        self._frame = None  # should be a numpy ndarray (RGB uint8)
        self._ts = 0.0
        self._seq = 0
        self._closed = False

    def close(self):
        """Close the hub and unblock all waiters."""
        with self._cv:
            self._closed = True
            self._cv.notify_all()

    def set(self, frame, ts=None):
        """Publish the latest frame and return the new sequence number.

        Args:
            frame: Frame data (RGB uint8 ndarray).
            ts: Optional timestamp (seconds).

        Returns:
            New sequence number.
        """
        if ts is None:
            ts = time.time()

        with self._cv:
            if self._closed:
                return self._seq
            self._frame = frame
            self._ts = float(ts)
            self._seq += 1
            self._cv.notify_all()
            return self._seq

    def get_latest(self):
        """Return the latest frame tuple.

        Returns:
            Tuple (frame, ts, seq); frame can be None if not set yet.
        """
        with self._lock:
            return self._frame, self._ts, self._seq

    def wait_for_next(self, last_seq, timeout=None):
        """Block until seq > last_seq, or timeout/close.

        Args:
            last_seq: Last observed sequence number.
            timeout: Optional timeout in seconds.

        Returns:
            True if a newer frame is available; False on timeout/close.
        """
        with self._cv:
            if self._closed:
                return False
            if self._seq > last_seq:
                return True

            if timeout is None:
                while (not self._closed) and (self._seq <= last_seq):
                    self._cv.wait()
                return (not self._closed) and (self._seq > last_seq)

            # timeout path
            deadline = time.time() + float(timeout)
            while (not self._closed) and (self._seq <= last_seq):
                remaining = deadline - time.time()
                if remaining <= 0.0:
                    break
                self._cv.wait(remaining)
            return (not self._closed) and (self._seq > last_seq)


class ResultHub:
    """Latest-result store per key."""

    def __init__(self):
        """Initialize result storage."""
        self._lock = threading.Lock()
        self._data = {}  # key -> (value, ts)

    def update(self, key, value, ts=None):
        """Update a key with value and timestamp.

        Args:
            key: Result key.
            value: Result value.
            ts: Optional timestamp (seconds).
        """
        if ts is None:
            ts = time.time()
        with self._lock:
            self._data[str(key)] = (value, float(ts))

    def get(self, key, default=None):
        """Fetch a key from the result store.

        Args:
            key: Result key.
            default: Default value if missing.

        Returns:
            Tuple (value, ts) or (default, 0.0) if missing.
        """
        with self._lock:
            item = self._data.get(str(key))
            if item is None:
                return default, 0.0
            return item[0], item[1]

    def snapshot(self):
        """Return a shallow copy of key -> (value, ts)."""
        with self._lock:
            return dict(self._data)


class StopFlag:
    """Simple stop flag for threads."""

    def __init__(self):
        """Initialize stop flag state."""
        self._evt = threading.Event()

    def stop(self):
        """Set the stop flag."""
        self._evt.set()

    def is_set(self):
        """Check if the stop flag is set."""
        return self._evt.is_set()

    def wait(self, timeout=None):
        """Block until stop flag set or timeout.

        Args:
            timeout: Optional timeout in seconds.

        Returns:
            True if stop flag is set; False on timeout.
        """
        return self._evt.wait(timeout)

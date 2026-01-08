import threading
import time

_GPU_LOCK = threading.Lock()


def _lazy_import_trt():
    """Import TensorRT lazily to avoid hard dependency at module import time."""
    import tensorrt as trt  # noqa: F401

    return trt


def _lazy_import_cuda():
    """Import PyCUDA lazily to avoid hard dependency at module import time."""
    import pycuda.driver as cuda  # noqa: F401

    return cuda


def _lazy_import_cv2():
    """Import OpenCV lazily to avoid hard dependency at module import time."""
    import cv2  # noqa: F401

    return cv2


class TrtEngineWrapper:
    """Minimal TensorRT engine wrapper with preallocated I/O buffers.

    Args:
        engine_path: Path to TensorRT engine file.
        logger_severity: Optional TRT logger severity.
    """

    def __init__(self, engine_path, logger_severity=None):
        trt = _lazy_import_trt()
        cuda = _lazy_import_cuda()

        if logger_severity is None:
            logger_severity = trt.Logger.WARNING

        self._trt = trt
        self._cuda = cuda
        self._logger = trt.Logger(logger_severity)

        self.engine = self._load_engine(engine_path)
        if self.engine is None:
            raise RuntimeError("Failed to load TensorRT engine: %s" % engine_path)

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create execution context")

        self.bindings = [0] * self.engine.num_bindings
        self.inputs = []
        self.outputs = []
        self.stream = cuda.Stream()

        # Iterate bindings in engine order
        import numpy as np

        for binding in self.engine:
            idx = self.engine.get_binding_index(binding)
            shape = tuple(self.engine.get_binding_shape(binding))
            if any([d < 0 for d in shape]):
                raise RuntimeError(
                    "Dynamic shape not supported in this minimal wrapper: %s %s" % (binding, shape)
                )
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            size = int(np.prod(shape))

            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings[idx] = int(device_mem)

            io_desc = {
                "name": binding,
                "index": idx,
                "shape": shape,
                "dtype": dtype,
                "host_mem": host_mem,
                "device_mem": device_mem,
            }

            if self.engine.binding_is_input(binding):
                self.inputs.append(io_desc)
            else:
                self.outputs.append(io_desc)

        if not self.inputs:
            raise RuntimeError("Engine has no inputs")
        if not self.outputs:
            raise RuntimeError("Engine has no outputs")

    def _load_engine(self, engine_path):
        """Load and deserialize a TensorRT engine.

        Args:
            engine_path: Path to engine file.

        Returns:
            Deserialized engine instance.
        """
        trt = self._trt
        with open(engine_path, "rb") as f, trt.Runtime(self._logger) as runtime:
            engine_bytes = f.read()
            return runtime.deserialize_cuda_engine(engine_bytes)

    def io(self):
        """Return stripped I/O metadata for introspection.

        Returns:
            Dict with "inputs" and "outputs" lists.
        """

        def _strip(x):
            return {
                "name": x["name"],
                "index": x["index"],
                "shape": x["shape"],
                "dtype": str(x["dtype"]),
            }

        return {
            "inputs": [_strip(i) for i in self.inputs],
            "outputs": [_strip(o) for o in self.outputs],
        }

    def infer(self, input_array):
        """Run inference and return reshaped output arrays.

        Args:
            input_array: Input ndarray matching inputs[0]["shape"].

        Returns:
            List of output ndarrays in engine output shapes.
        """
        cuda = self._cuda
        import numpy as np

        inp = self.inputs[0]
        host_mem = inp["host_mem"]
        device_mem = inp["device_mem"]
        shape = inp["shape"]
        dtype = inp["dtype"]

        flat = np.asarray(input_array).astype(dtype, copy=False).ravel()
        if flat.size != host_mem.size:
            raise ValueError(
                "Input size mismatch: got %d, expected %d (engine input shape: %s)"
                % (flat.size, host_mem.size, shape)
            )

        np.copyto(host_mem, flat)

        # H2D
        cuda.memcpy_htod_async(device_mem, host_mem, self.stream)

        # inference
        ok = self.context.execute_async_v2(
            bindings=self.bindings, stream_handle=int(self.stream.handle)
        )
        if not ok:
            raise RuntimeError("execute_async_v2 returned False")

        # D2H for all outputs
        outs = []
        for out in self.outputs:
            o_host = out["host_mem"]
            o_dev = out["device_mem"]
            o_shape = out["shape"]
            cuda.memcpy_dtoh_async(o_host, o_dev, self.stream)
            outs.append(o_host.reshape(o_shape))

        self.stream.synchronize()
        return outs


def preprocess_rgb_u8_to_engine_input(
    frame_rgb_u8, engine_input_shape, out_dtype, mean=None, std=None
):
    """Preprocess RGB uint8 frame to contiguous NCHW engine input.

    Args:
        frame_rgb_u8: HxWx3 uint8 frame in RGB.
        engine_input_shape: Expected (N, C, H, W) shape.
        out_dtype: Output dtype (np.float16/np.float32).
        mean: Optional per-channel mean (RGB).
        std: Optional per-channel std (RGB).

    Returns:
        Contiguous ndarray shaped (N, C, H, W).
    """
    import numpy as np

    cv2 = _lazy_import_cv2()

    n, c, h, w = [int(x) for x in engine_input_shape]
    if n != 1 or c != 3:
        raise ValueError(
            "Only N=1,C=3 supported in this minimal preprocess, got %s" % (engine_input_shape,)
        )

    if frame_rgb_u8.shape[0] != h or frame_rgb_u8.shape[1] != w:
        img = cv2.resize(frame_rgb_u8, (w, h), interpolation=cv2.INTER_AREA)
    else:
        img = frame_rgb_u8

    x = img.astype(np.float32) * (1.0 / 255.0)  # HWC in [0,1]

    if mean is not None and std is not None:
        # mean/std should be array-like length 3 (RGB)
        m = np.asarray(mean, dtype=np.float32).reshape((1, 1, 3))
        s = np.asarray(std, dtype=np.float32).reshape((1, 1, 3))
        x = (x - m) / s

    x = np.transpose(x, (2, 0, 1))  # CHW
    x = np.expand_dims(x, axis=0)  # NCHW
    x = np.ascontiguousarray(x.astype(out_dtype, copy=False))
    return x


class _WorkerBase(threading.Thread):
    def __init__(self, name, framehub, resulthub, stopflag, max_hz):
        """Base worker for frame-based processing loops."""
        threading.Thread.__init__(self, name=name)
        self.daemon = True
        self.framehub = framehub
        self.resulthub = resulthub
        self.stopflag = stopflag
        self.max_hz = max_hz
        self._seq = 0

    def _sleep_to_rate(self, last_tick):
        """Sleep to maintain the configured maximum loop rate.

        Args:
            last_tick: Previous loop timestamp.

        Returns:
            Current timestamp after sleeping as needed.
        """
        if self.max_hz is None:
            return time.time()
        now = time.time()
        min_dt = 1.0 / float(self.max_hz)
        dt = now - last_tick
        if dt < min_dt:
            time.sleep(min_dt - dt)
        return time.time()


class SteerPointWorker(_WorkerBase):
    """ResNet steering worker consuming RGB frames and publishing steer_xy.

    Args:
        framehub: FrameHub instance.
        resulthub: ResultHub instance.
        stopflag: StopFlag instance.
        engine_path: Path to TensorRT engine file.
        max_hz: Maximum inference rate.
        mean: Optional per-channel mean (RGB).
        std: Optional per-channel std (RGB).
    """

    def __init__(
        self,
        framehub,
        resulthub,
        stopflag,
        engine_path,
        max_hz=30.0,
        mean=None,
        std=None,
    ):
        _WorkerBase.__init__(self, "SteerPointWorker", framehub, resulthub, stopflag, max_hz)
        self.engine_path = engine_path
        self.mean = mean
        self.std = std

    def run(self):
        cuda = _lazy_import_cuda()
        cuda.init()
        ctx = cuda.Device(0).make_context()

        try:
            engine = TrtEngineWrapper(self.engine_path)
            self.resulthub.update("steer.engine_io", engine.io(), ts=time.time())

            inp = engine.inputs[0]
            in_shape = inp["shape"]
            in_dtype = inp["dtype"]

            # infer loop
            last_tick = 0.0
            while not self.stopflag.is_set():
                last_tick = self._sleep_to_rate(last_tick)

                if not self.framehub.wait_for_next(self._seq, timeout=0.5):
                    continue
                frame_rgb, ts, seq = self.framehub.get_latest()
                self._seq = seq
                if frame_rgb is None:
                    continue

                x = preprocess_rgb_u8_to_engine_input(
                    frame_rgb, in_shape, in_dtype, mean=self.mean, std=self.std
                )
                with _GPU_LOCK:
                    outs = engine.infer(x)

                import numpy as np

                flat = np.asarray(outs[0]).ravel().astype(np.float32)
                if flat.size < 2:
                    continue
                x = float(flat[0])
                y = float(flat[1])
                self.resulthub.update("steer_xy", {"x": x, "y": y}, ts=ts)
        finally:
            try:
                ctx.pop()
            except Exception:
                pass


class SpeedMindWorker(_WorkerBase):
    """YOLO worker that publishes raw outputs or a postprocessed policy.

    Args:
        framehub: FrameHub instance.
        resulthub: ResultHub instance.
        stopflag: StopFlag instance.
        engine_path: Path to TensorRT engine file.
        max_hz: Maximum inference rate.
        postprocess_fn: Optional callable to produce policy dicts.
    """

    def __init__(
        self,
        framehub,
        resulthub,
        stopflag,
        engine_path,
        max_hz=8.0,
        postprocess_fn=None,
    ):
        _WorkerBase.__init__(self, "SpeedMindWorker", framehub, resulthub, stopflag, max_hz)
        self.engine_path = engine_path
        self.postprocess_fn = postprocess_fn

    def run(self):
        cuda = _lazy_import_cuda()
        cuda.init()
        ctx = cuda.Device(0).make_context()
        try:
            engine = TrtEngineWrapper(self.engine_path)
            self.resulthub.update("yolo.engine_io", engine.io(), ts=time.time())

            inp = engine.inputs[0]
            in_shape = inp["shape"]
            in_dtype = inp["dtype"]

            last_tick = 0.0
            while not self.stopflag.is_set():
                last_tick = self._sleep_to_rate(last_tick)

                if not self.framehub.wait_for_next(self._seq, timeout=0.5):
                    continue
                frame_rgb, ts, seq = self.framehub.get_latest()
                self._seq = seq
                if frame_rgb is None:
                    continue

                x = preprocess_rgb_u8_to_engine_input(frame_rgb, in_shape, in_dtype)
                with _GPU_LOCK:
                    outs = engine.infer(x)

                # publish
                if self.postprocess_fn is None:
                    # keep the "outputs list" semantics like lane.py
                    self.resulthub.update("yolo.outputs", outs, ts=ts)
                else:
                    payload = self.postprocess_fn(outs, ts=ts, in_shape=in_shape)
                    self.resulthub.update("yolo", payload, ts=ts)
        finally:
            try:
                ctx.pop()
            except Exception:
                pass


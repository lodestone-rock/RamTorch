"""Bounded Kineto capture with reliable CPU annotations for 2BW workers.

The caller must enter the capture before starting work, join its workers, and
synchronize all participating CUDA devices *before leaving* the context. This
module never synchronizes a device or calls ``profiler.step()``. CPU spans show
host execution/enqueue time, not GPU execution time; only Kineto supplies GPU
kernels. GPU tracing requires a CUDA-enabled PyTorch build and working CUPTI.
"""
from __future__ import annotations

import contextlib
import gzip
import json
import os
import sys
import tempfile
import threading
import time
import warnings
from collections import Counter
from pathlib import Path

__all__ = ["TraceCapture", "inspect_trace"]


def _read_trace(path):
    opener = gzip.open if os.fspath(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


class TraceCapture:
    """Capture one bounded run, including annotations made on worker threads.

    ``TraceCapture(None)`` and its spans are no-ops; even importing PyTorch is
    deferred until an enabled capture starts. ``path`` accepts strings and
    path-like objects, with standard-library gzip output for a ``.gz`` suffix.
    Parent directories must already exist. Enabled instances are single-use;
    overlapping/nested PyTorch profilers are not supported.

    Use ``with TraceCapture(path) as trace`` on the caller thread, and
    ``with trace.span("F", stage=0, mb=3, group=1, version=0, slot=0)`` in workers.
    Metadata must be JSON serializable. All spans and GPU work must finish before
    context exit. Exit stops the profiler and exports automatically, including
    on exceptional exit; the caller still owns cleanup/synchronization then.

    Kineto may omit worker-thread ``record_function`` events. Every span is also
    timed with a monotonic clock and missing CPU annotations are inserted after
    export, calibrated to a caller-thread Kineto marker. Native annotations are
    retained (not duplicated) and receive the same metadata. No GPU events are
    inferred from CPU spans. Capture/export memory grows with the bounded run.
    """

    def __init__(self, path=None):
        self.path = None if path is None else os.fsdecode(path)
        self._profiler = None
        self._record_function = None
        self._active = False
        self._stopped = False
        self._exported = False
        self._used = False
        self._lock = threading.Lock()
        self._spans = []
        self._next_span = 0
        self._prefix = "__ramtorch_2bw_{}".format(id(self))
        self._clock_name = self._prefix + "_clock_sync"
        self._clock_ns = 0
        self._wall_ns = 0

    def __enter__(self):
        if self.path is None:
            return self
        if self._used:
            raise RuntimeError("TraceCapture instances can only capture one run")
        import torch

        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self._record_function = torch.profiler.record_function
        self._profiler = torch.profiler.profile(activities=activities)
        self._profiler.__enter__()
        self._used = True
        try:
            # Like offload.py, use an actual caller-thread Kineto annotation as
            # the clock anchor: trace timestamps need not be Unix timestamps.
            before = time.monotonic_ns()
            with self._record_function(self._clock_name):
                after = time.monotonic_ns()
                self._clock_ns = (before + after) // 2
            before = time.monotonic_ns()
            self._wall_ns = time.time_ns()
            after = time.monotonic_ns()
            self._wall_clock_ns = (before + after) // 2
        except BaseException:
            self._profiler.__exit__(*sys.exc_info())
            raise
        self._active = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.path is None:
            return False
        self._active = False
        # The runtime must already have joined workers and drained every GPU.
        self._profiler.__exit__(exc_type, exc_value, traceback)
        self._stopped = True
        try:
            self.export()
        except Exception as error:
            if exc_type is None:
                raise
            warnings.warn("Could not export 2BW trace: {}".format(error),
                          RuntimeWarning, stacklevel=2)
        return False

    @contextlib.contextmanager
    def span(self, name, stage=-1, **metadata):
        """Annotate CPU work without imposing any device synchronization.

        ``stage`` and arbitrary metadata (typically ``mb``, ``group``,
        ``version``, and ``slot``) appear in each annotation's Chrome-trace args.
        Spans outside an active capture are no-ops. Exceptions are propagated
        after recording the completed portion of a span.
        """
        if not self._active:
            yield
            return
        with self._lock:
            sequence = self._next_span
            self._next_span += 1
        record_name = "{}_span_{}".format(self._prefix, sequence)
        thread = threading.current_thread()
        tid = threading.get_native_id()
        args = dict(metadata, stage=stage)
        start_ns = time.monotonic_ns()
        record = self._record_function(record_name)
        try:
            record.__enter__()
        except RuntimeError:
            # Some builds cannot install a record_function on worker threads.
            # Monotonic timing remains available even without that annotation.
            record = None
        try:
            yield
        finally:
            end_ns = time.monotonic_ns()
            try:
                if record is not None:
                    record.__exit__(None, None, None)
            finally:
                with self._lock:
                    self._spans.append((record_name, str(name), tid, thread.name,
                                        start_ns, end_ns - start_ns, args))

    def export(self):
        """Return the trace path, exporting once after the profiler has stopped.

        Context exit calls this automatically. Calling while capture is active
        (or before entering it) raises rather than stopping a live profiler.
        Repeated calls after a successful export are harmless. Disabled captures
        return ``None``.
        """
        if self.path is None:
            return None
        if not self._stopped:
            raise RuntimeError("Exit TraceCapture before exporting the trace")
        if self._exported:
            return self.path
        # Export uncompressed first: gzip support differs across torch versions.
        with tempfile.TemporaryDirectory(prefix="ramtorch_2bw_trace_") as tmp:
            raw_path = str(Path(tmp) / "kineto.json")
            self._profiler.export_chrome_trace(raw_path)
            document = _read_trace(raw_path)
        self._inject_spans(document)
        opener = gzip.open if self.path.endswith(".gz") else open
        with opener(self.path, "wt", encoding="utf-8") as stream:
            json.dump(document, stream)
        self._exported = True
        return self.path

    def _inject_spans(self, document):
        events = document.setdefault("traceEvents", [])
        marker = next((event for event in events
                       if event.get("name") == self._clock_name
                       and event.get("ph") == "X"), None)
        if marker is not None:
            offset_us = marker["ts"] - self._clock_ns / 1000
            pid = marker.get("pid", os.getpid())
        else:
            # Modern Kineto timestamps are Unix time minus this base. This is
            # only an approximate fallback if the calibration event is missing.
            offset_us = (self._wall_ns - self._wall_clock_ns
                         - document.get("baseTimeNanoseconds", 0)) / 1000
            pid = os.getpid()
            warnings.warn("Kineto clock marker missing; using approximate "
                          "wall-clock alignment for 2BW CPU annotations",
                          RuntimeWarning, stacklevel=2)
        native = {event.get("name"): event for event in events
                  if event.get("ph") == "X"
                  and event.get("cat") == "user_annotation"}
        named_threads = {(event.get("pid"), event.get("tid")) for event in events
                         if event.get("ph") == "M"
                         and event.get("name") == "thread_name"}
        with self._lock:
            spans = list(self._spans)
        for record_name, name, tid, thread_name, start_ns, duration_ns, args in spans:
            event = native.get(record_name)
            if event is not None:
                event["name"] = name
                event.setdefault("args", {}).update(args)
                event["args"].update(ramtorch_2bw=True, annotation_source="kineto")
                continue
            events.append({
                "ph": "X", "cat": "user_annotation", "name": name,
                "pid": pid, "tid": tid,
                "ts": start_ns / 1000 + offset_us, "dur": duration_ns / 1000,
                "args": dict(args, ramtorch_2bw=True,
                             annotation_source="monotonic_fallback"),
            })
            if (pid, tid) not in named_threads:
                named_threads.add((pid, tid))
                events.append({
                    "ph": "M", "name": "thread_name", "pid": pid, "tid": tid,
                    "args": {"name": "2BW " + thread_name},
                })
        # Internal clock markers should not inflate annotation counts.
        events[:] = [event for event in events
                     if event.get("name") != self._clock_name]


def inspect_trace(path):
    """Summarize a Chrome/Kineto JSON or JSON.gz trace using only the stdlib.

    Returns ``gpu_kernels`` (total), ``gpu_kernels_per_device`` (string device
    IDs to counts), ``cpu_annotations`` (all complete user annotations),
    ``cpu_annotations_by_name``, ``pipeline_annotations``, and
    ``fallback_cpu_annotations``. ``gpu_memcopies``,
    ``gpu_memcopies_by_direction``, and ``gpu_memcopies_per_device`` count real
    complete ``gpu_memcpy`` events (H2D/D2H/D2D/P2P/unknown). Missing memcpy
    events may mean this Kineto build does not expose them. GPU kernel counts
    include only actual complete ``kernel`` events, never CPU launch spans,
    copies, or synthetic annotations.
    Zero kernels is valid on CPU; on CUDA it can also indicate missing CUPTI or
    missing end-of-run synchronization, not that no GPU work executed.
    """
    document = _read_trace(path)
    events = document if isinstance(document, list) else document.get("traceEvents", [])
    kernels = Counter()
    memcopies = Counter()
    memcopies_per_device = {}
    annotations = Counter()
    pipeline_annotations = 0
    fallback_annotations = 0
    for event in events:
        if event.get("ph") != "X":
            continue
        categories = set(event.get("cat", "").split(","))
        args = event.get("args", {})
        if "kernel" in categories:
            # Kineto normally supplies args.device; its GPU pid is the device
            # ID on older exports. Avoid inferring kernels from event names.
            device = args.get("device", event.get("pid", "unknown"))
            kernels[str(device)] += 1
        if "gpu_memcpy" in categories:
            # Count only actual Kineto CUDA copies, never CPU launch spans.
            # Direction names are Kineto's usual "Memcpy HtoD (Pinned)" etc.
            name = event.get("name", "").lower()
            direction = next((direction for marker, direction in (
                ("htod", "H2D"), ("dtoh", "D2H"), ("dtod", "D2D"),
                ("ptop", "P2P"),
            ) if marker in name), "unknown")
            device = str(args.get("device", event.get("pid", "unknown")))
            memcopies[direction] += 1
            memcopies_per_device.setdefault(device, Counter())[direction] += 1
        if "user_annotation" in categories:
            annotations[event.get("name", "")] += 1
            if args.get("ramtorch_2bw") is True:
                pipeline_annotations += 1
                if args.get("annotation_source") == "monotonic_fallback":
                    fallback_annotations += 1
    return {
        "gpu_kernels": sum(kernels.values()),
        "gpu_kernels_per_device": dict(sorted(kernels.items())),
        "gpu_memcopies": sum(memcopies.values()),
        "gpu_memcopies_by_direction": dict(sorted(memcopies.items())),
        "gpu_memcopies_per_device": {
            device: dict(sorted(counts.items()))
            for device, counts in sorted(memcopies_per_device.items())
        },
        "cpu_annotations": sum(annotations.values()),
        "cpu_annotations_by_name": dict(sorted(annotations.items())),
        "pipeline_annotations": pipeline_annotations,
        "fallback_cpu_annotations": fallback_annotations,
    }

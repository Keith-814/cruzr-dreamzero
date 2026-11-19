"""Pick/present/return orchestration, independent of MuJoCo and ROS.

Backends own motion, feedback and coordinate calibration. A named waypoint is
an oracle Cartesian target in simulation or a commissioned joint pose on hardware.
"""
from dataclasses import dataclass, field
from typing import Protocol
import math


WAYPOINTS = ('pregrasp', 'grasp', 'lift', 'scan', 'return_high', 'return_low', 'retreat')


class TaskBackend(Protocol):
    def move(self, waypoint: str, width: float, *, holding: bool = False): ...
    def grip(self, width: float, *, grasp: bool = False): ...
    def wait(self, seconds: float, *, holding: bool = False): ...
    def capture(self) -> dict: ...
    def verify_return(self) -> bool | None: ...
    def stop(self): ...


@dataclass(frozen=True)
class ScanTaskConfig:
    open_width_m: float = 0.045
    closed_width_m: float = 0.004
    dwell_s: float = 0.8
    retry_wait_s: float = 0.3
    max_scan_attempts: int = 3
    expected_code: str | None = None

    def __post_init__(self):
        numbers = (self.open_width_m, self.closed_width_m, self.dwell_s, self.retry_wait_s)
        if not all(math.isfinite(x) for x in numbers):
            raise ValueError('Task parameters must be finite')
        if not 0 <= self.closed_width_m < self.open_width_m <= 0.05:
            raise ValueError('Invalid gripper targets in metres')
        if self.dwell_s < 0 or self.retry_wait_s < 0:
            raise ValueError('Waits must be nonnegative')
        if isinstance(self.max_scan_attempts, bool) or not isinstance(self.max_scan_attempts, int) or not 1 <= self.max_scan_attempts <= 20:
            raise ValueError('Scan attempts must be an integer in [1,20]')
        if self.expected_code is not None and not self.expected_code.strip():
            raise ValueError('Expected code cannot be blank')


@dataclass
class TaskResult:
    success: bool = False
    motion_complete: bool = False
    placement_verified: bool | None = None
    barcode: str | None = None
    scan_source: str | None = None
    phases: list[str] = field(default_factory=list)
    scan_attempts: list = field(default_factory=list)
    error: str | None = None
    stop_error: str | None = None


@dataclass(frozen=True)
class PresentTaskConfig:
    open_width_m: float = 0.045
    closed_width_m: float = 0.004
    dwell_s: float = 1.0

    def __post_init__(self):
        if not 0 <= self.closed_width_m < self.open_width_m <= 0.05:
            raise ValueError('Invalid gripper targets in metres')
        if not math.isfinite(self.dwell_s) or self.dwell_s < 0:
            raise ValueError('dwell_s must be finite and nonnegative')


def run_pick_present_return(backend: TaskBackend, config=None, *, event=None):
    """Pick an object, hold it at the taught ``scan``/face waypoint, and return.

    The waypoint is named ``scan`` for compatibility with existing taught
    plans; no scanner, barcode, or visual recognition is required here.
    """
    c = config or PresentTaskConfig()
    result = TaskResult()

    def phase(name):
        result.phases.append(name)
        if event:
            event({'phase': name})

    try:
        phase('approach')
        backend.grip(c.open_width_m)
        backend.move('pregrasp', c.open_width_m)
        backend.move('grasp', c.open_width_m)
        phase('grasp')
        backend.grip(c.closed_width_m, grasp=True)
        phase('lift')
        backend.move('lift', c.closed_width_m, holding=True)
        phase('present')
        backend.move('scan', c.closed_width_m, holding=True)
        backend.wait(c.dwell_s, holding=True)
        # Save a synchronized observation for later visual inspection when the
        # backend supports it; it is evidence only, never a success predicate.
        backend.capture()
        phase('return')
        backend.move('return_high', c.closed_width_m, holding=True)
        backend.move('return_low', c.closed_width_m, holding=True)
        phase('release')
        backend.grip(c.open_width_m)
        phase('retreat')
        backend.move('retreat', c.open_width_m)
        backend.wait(0.6)
        result.motion_complete = True
        result.placement_verified = backend.verify_return()
        result.success = result.placement_verified is True
        if result.placement_verified is not True:
            result.error = 'Return motion completed; object placement not verified'
        phase('complete')
    except Exception as exc:
        result.error = f'{type(exc).__name__}: {exc}'
        phase('fault')
    finally:
        try:
            backend.stop()
        except Exception as exc:
            result.stop_error = f'{type(exc).__name__}: {exc}'
            result.success = False
    return result


def run_pick_scan_return(backend: TaskBackend, scanner, config=None, *, event=None):
    """A decode failure returns the held object; a motion fault aborts in place.

    No automatic recovery motion after a failed grasp, stale state, or dropped
    object. Placement verification must be supplied by the backend/inspection.
    """
    c = config or ScanTaskConfig()
    result = TaskResult()

    def phase(name):
        result.phases.append(name)
        if event:
            event({'phase': name})

    try:
        phase('approach')
        backend.grip(c.open_width_m)
        backend.move('pregrasp', c.open_width_m)
        backend.move('grasp', c.open_width_m)
        phase('grasp')
        backend.grip(c.closed_width_m, grasp=True)
        phase('lift')
        backend.move('lift', c.closed_width_m, holding=True)
        phase('scan')
        backend.move('scan', c.closed_width_m, holding=True)
        backend.wait(c.dwell_s, holding=True)
        seen = set()
        for attempt in range(c.max_scan_attempts):
            # Capture/motion failures are not decoder failures: do not continue
            # moving with unknown robot or camera state.
            capture = backend.capture()
            if capture['id'] in seen:
                raise RuntimeError('Repeated camera frame; refusing stale scan evidence')
            seen.add(capture['id'])
            record = {'attempt': attempt+1, 'capture_id': capture['id'], 'codes': []}
            try:
                decoded = scanner(capture)
                record.update(decoded)
                matches = [code['text'] for code in decoded['codes']
                           if code.get('text') and (c.expected_code is None or code['text'] == c.expected_code)]
                # Ambiguous scenes must be narrowed with expected_code/ROI.
                if len(set(matches)) == 1:
                    result.barcode = matches[0]
                    result.scan_source = decoded.get('source', 'camera')
            except Exception as exc:
                record['decode_error'] = f'{type(exc).__name__}: {exc}'
            result.scan_attempts.append(record)
            if event:
                event({'scan': record})
            if result.barcode:
                break
            backend.wait(c.retry_wait_s, holding=True)
        phase('return')
        backend.move('return_high', c.closed_width_m, holding=True)
        backend.move('return_low', c.closed_width_m, holding=True)
        phase('release')
        backend.grip(c.open_width_m)
        phase('retreat')
        backend.move('retreat', c.open_width_m)
        backend.wait(0.6)
        result.motion_complete = True
        result.placement_verified = backend.verify_return()
        result.success = bool(result.barcode and result.placement_verified is True)
        if not result.barcode:
            result.error = 'Scan failed or ambiguous; object return sequence completed'
        elif result.placement_verified is not True:
            result.error = 'Return motion completed; object placement not verified'
        phase('complete')
    except Exception as exc:
        result.error = f'{type(exc).__name__}: {exc}'
        phase('fault')
    finally:
        try:
            backend.stop()
        except Exception as exc:
            result.stop_error = f'{type(exc).__name__}: {exc}'
            result.success = False
    return result

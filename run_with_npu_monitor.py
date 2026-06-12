#!/usr/bin/env python3
"""
NPU utilization monitoring & command execution script.

Usage:
    python run_with_npu_monitor.py [command...]
    python run_with_npu_monitor.py  # Default: demo/demo_offline.py test_files --finegrained

Examples:
    python run_with_npu_monitor.py python demo/demo_offline.py test_files --finegrained
    python run_with_npu_monitor.py --interval 0.5 python demo/demo_offline.py test_files

Output:
    - Real-time NPU usage (1s interval)
    - Post-execution summary (Avg, Max, Idle time, etc.)
"""

import argparse
import os
import re
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path


ANSI_ESCAPE = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]|\x1b\(B|\r')
DEVICE_RE = re.compile(r'Device\s*:(\d+)')
CORE_UTIL_RE = re.compile(r'Core\s*:(\d+)\s+Util:\s+([\d.]+)%')
NPU_MEM_RE = re.compile(r'NPU Memory:.*?([\d.]+)\s*(KiB|MiB|GiB)\s*/\s*([\d.]+)\s*(GiB)')
TOTAL_DEVICES_RE = re.compile(r'Total Devices:\s*(\d+)')


def parse_dxtop_output(raw: str) -> dict:
    """Parse per-Device Core Util% and NPU Memory from dxtop output.

    Returns:
        {
            'devices': {0: {0: util%, 1: util%, 2: util%}, 1: {...}, ...},
            'cores': {(dev, core): util%, ...},  # flat view (for backward-compat overall average)
            'memory_pct': float,
            'num_devices': int,
        }
    """
    clean = ANSI_ESCAPE.sub('', raw)
    result = {'devices': {}, 'cores': {}, 'memory_pct': 0.0, 'num_devices': 0}

    # Detect Total Devices
    td_match = TOTAL_DEVICES_RE.search(clean)
    if td_match:
        result['num_devices'] = int(td_match.group(1))

    # dxtop redraws the whole screen periodically — use the last complete block.
    # Locate the last complete output block by the "Total Devices:" marker.
    blocks = clean.split('Total Devices:')
    if len(blocks) >= 2:
        # Use the data region just before the last complete block
        last_block = blocks[-2] if len(blocks) > 2 else blocks[-2]
    else:
        last_block = clean

    # Parse per-Device cores: split on the Device header
    device_sections = DEVICE_RE.split(last_block)
    # device_sections: [before_first_device, dev_id_str, section, dev_id_str, section, ...]
    current_dev = 0
    for i in range(1, len(device_sections), 2):
        dev_id = int(device_sections[i])
        section = device_sections[i + 1] if i + 1 < len(device_sections) else ""
        cores = {}
        for m in CORE_UTIL_RE.finditer(section):
            core_id = int(m.group(1))
            util = float(m.group(2))
            cores[core_id] = util
            result['cores'][(dev_id, core_id)] = util
        if cores:
            result['devices'][dev_id] = cores

    # fallback: no Device header (older single-device format)
    if not result['devices']:
        all_matches = list(CORE_UTIL_RE.finditer(last_block))
        if all_matches:
            # Detect the core count dynamically (on a duplicate core_id, keep only the last set)
            seen_ids = []
            for m in all_matches:
                cid = int(m.group(1))
                if seen_ids and cid <= seen_ids[-1]:
                    seen_ids = []  # new device starts
                seen_ids.append(cid)
            # core count = size of one set
            cores_per_device = len(seen_ids) if seen_ids else 3
            latest = all_matches[-cores_per_device:]
            cores = {}
            for m in latest:
                core_id = int(m.group(1))
                util = float(m.group(2))
                cores[core_id] = util
                result['cores'][(0, core_id)] = util
            result['devices'][0] = cores

    if not result['num_devices']:
        result['num_devices'] = len(result['devices']) or 1

    # Memory (use the last match)
    mem_matches = list(NPU_MEM_RE.finditer(clean))
    if mem_matches:
        mem_match = mem_matches[-1]
        used_val = float(mem_match.group(1))
        used_unit = mem_match.group(2)
        total_val = float(mem_match.group(3))
        if used_unit == 'KiB':
            used_val /= (1024 * 1024)
        elif used_unit == 'MiB':
            used_val /= 1024
        result['memory_pct'] = (used_val / total_val * 100) if total_val > 0 else 0

    return result


def sample_dxtop() -> dict:
    """Sample dxtop once and return the result. (standalone fallback)"""
    import tempfile
    tmp = tempfile.mktemp(suffix='.txt')
    try:
        subprocess.run(
            ['timeout', '--signal=INT', '2', 'script', '-qc', 'dxtop', tmp],
            capture_output=True, timeout=5
        )
        if os.path.exists(tmp):
            with open(tmp, 'r', errors='replace') as f:
                raw = f.read()
            os.unlink(tmp)
            return parse_dxtop_output(raw)
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    if os.path.exists(tmp):
        os.unlink(tmp)
    return {'devices': {}, 'cores': {}, 'memory_pct': 0.0, 'num_devices': 0}


def make_bar(pct: float, width: int = 20) -> str:
    """Render a percentage as an ASCII bar."""
    filled = int(pct / 100 * width)
    return '█' * filled + '░' * (width - filled)


def format_core_line(devices: dict) -> str:
    """Format per-Device core utilization on one line. devices = {dev_id: {core_id: util}}"""
    if not devices:
        return "  (no data)"
    parts = []
    for dev_id in sorted(devices.keys()):
        cores = devices[dev_id]
        for cid in sorted(cores.keys()):
            pct = cores[cid]
            bar = make_bar(pct, 10)
            parts.append(f"D{dev_id}C{cid} {bar} {pct:5.1f}%")
    return "  ".join(parts)


class NpuMonitor:
    """Run dxtop long-lived in the background and periodically re-parse its output file."""

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread = None
        self._dxtop_proc = None
        self._dxtop_file = '/tmp/dxtop_monitor_live.txt'

    def start(self):
        # Remove any stale file
        if os.path.exists(self._dxtop_file):
            os.unlink(self._dxtop_file)
        # Run dxtop long-lived (wrap in `script` for a pty to capture ANSI output)
        self._dxtop_proc = subprocess.Popen(
            ['script', '-qc', 'dxtop', self._dxtop_file],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid
        )
        # Wait until dxtop produces its first output
        time.sleep(3)
        self._thread = threading.Thread(target=self._run, daemon=True, name="npu-monitor")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        if self._dxtop_proc:
            try:
                import signal
                os.killpg(os.getpgid(self._dxtop_proc.pid), signal.SIGINT)
                self._dxtop_proc.wait(timeout=3)
            except Exception:
                try:
                    self._dxtop_proc.kill()
                except Exception:
                    pass
        if os.path.exists(self._dxtop_file):
            os.unlink(self._dxtop_file)

    def _run(self):
        start_time = time.perf_counter()
        while not self._stop.is_set():
            data = self._read_latest()
            if data['devices']:
                # Compute the overall core average
                all_utils = [u for cores in data['devices'].values() for u in cores.values()]
                avg = sum(all_utils) / len(all_utils) if all_utils else 0
                sample = {
                    'time': time.time(),
                    'elapsed': time.perf_counter() - start_time,
                    'devices': data['devices'],
                    'cores': data['cores'],
                    'memory_pct': data['memory_pct'],
                    'num_devices': data['num_devices'],
                    'avg': avg,
                }
                self.samples.append(sample)
                elapsed = sample['elapsed']
                # Multi-device: show the per-device average
                if data['num_devices'] > 1:
                    dev_parts = []
                    for dev_id in sorted(data['devices'].keys()):
                        cores = data['devices'][dev_id]
                        d_avg = sum(cores.values()) / len(cores) if cores else 0
                        dev_parts.append(f"D{dev_id}:{d_avg:4.0f}%")
                    dev_str = " ".join(dev_parts)
                    print(f"\r\033[K  NPU │ {dev_str} │ total avg {avg:5.1f}% │ {elapsed:.0f}s", end='', flush=True)
                else:
                    print(f"\r\033[K  NPU │ {format_core_line(data['devices'])} │ avg {avg:5.1f}% │ {elapsed:.0f}s", end='', flush=True)
            self._stop.wait(self.interval)

    def _read_latest(self) -> dict:
        """Read and parse the dxtop output file."""
        try:
            if not os.path.exists(self._dxtop_file):
                return {'devices': {}, 'cores': {}, 'memory_pct': 0.0, 'num_devices': 0}
            with open(self._dxtop_file, 'r', errors='replace') as f:
                raw = f.read()
            return parse_dxtop_output(raw)
        except Exception:
            return {'devices': {}, 'cores': {}, 'memory_pct': 0.0, 'num_devices': 0}

    def summary(self) -> str:
        """Build summary statistics from the collected samples."""
        if not self.samples:
            return "  (no NPU samples)"

        n = len(self.samples)
        num_devices = self.samples[-1].get('num_devices', 1)

        # Overall statistics
        all_avgs = []
        device_core_utils = defaultdict(lambda: defaultdict(list))  # {dev_id: {core_id: [utils]}}
        device_avgs = defaultdict(list)  # {dev_id: [per-sample avg]}
        zero_count = 0

        for s in self.samples:
            avg = s.get('avg', 0)
            all_avgs.append(avg)
            if avg < 1.0:
                zero_count += 1
            for dev_id, cores in s.get('devices', {}).items():
                dev_vals = list(cores.values())
                if dev_vals:
                    device_avgs[dev_id].append(sum(dev_vals) / len(dev_vals))
                for cid, val in cores.items():
                    device_core_utils[dev_id][cid].append(val)

        total_avg = sum(all_avgs) / len(all_avgs)
        total_max = max(all_avgs)
        zero_pct = zero_count / n * 100

        lines = [
            "",
            "┌─────────────────────────────────────────────────────────────────┐",
            "│                    NPU Utilization Summary                        │",
            "├─────────────────────────────────────────────────────────────────┤",
            f"│  Devices: {num_devices}    Samples: {n:>4}  ({self.interval}s interval)                 │",
            f"│  Overall avg: {total_avg:5.1f}%    max: {total_max:5.1f}%                          │",
            f"│  Idle (0%):  {zero_count:>4} samples ({zero_pct:.1f}%)                          │",
            "├─────────────────────────────────────────────────────────────────┤",
        ]

        # Per-Device statistics
        for dev_id in sorted(device_core_utils.keys()):
            dev_vals = device_avgs.get(dev_id, [])
            d_avg = sum(dev_vals) / len(dev_vals) if dev_vals else 0
            d_max = max(dev_vals) if dev_vals else 0
            if num_devices > 1:
                lines.append(f"│  Device:{dev_id}  avg={d_avg:5.1f}%  max={d_max:5.1f}%                        │")
            for cid in sorted(device_core_utils[dev_id].keys()):
                vals = device_core_utils[dev_id][cid]
                c_avg = sum(vals) / len(vals)
                c_max = max(vals)
                c_min = min(vals)
                prefix = f"D{dev_id}" if num_devices > 1 else " "
                lines.append(
                    f"│  {prefix} Core:{cid}  avg={c_avg:5.1f}%  max={c_max:5.1f}%  min={c_min:5.1f}%      │"
                )

        lines.append("├─────────────────────────────────────────────────────────────────┤")

        # Timeline trend (10 buckets)
        if n >= 10:
            chunk_size = n // 10
            lines.append("│  Timeline trend (10 buckets):                                    │")
            for i in range(10):
                start = i * chunk_size
                end = start + chunk_size if i < 9 else n
                chunk_avgs = all_avgs[start:end]
                chunk_avg = sum(chunk_avgs) / len(chunk_avgs)
                bar = make_bar(chunk_avg, 25)
                pct_s = i * 10
                pct_e = (i + 1) * 10
                lines.append(f"│  {pct_s:>3}%-{pct_e:>3}%  {bar} {chunk_avg:5.1f}%       │")
        else:
            lines.append("│  Timeline trend: (not enough samples)                            │")

        lines.append("└─────────────────────────────────────────────────────────────────┘")
        return "\n".join(lines)


def main():
    # Before '--' are monitor options; after it is the command to run
    argv = sys.argv[1:]
    if '--' in argv:
        sep = argv.index('--')
        monitor_args = argv[:sep]
        cmd = argv[sep + 1:]
    else:
        # Split out only --interval as a monitor option
        monitor_args = []
        cmd = []
        i = 0
        while i < len(argv):
            if argv[i] == '--interval' and i + 1 < len(argv):
                monitor_args.extend([argv[i], argv[i + 1]])
                i += 2
            else:
                cmd = argv[i:]
                break
            i += 1

    parser = argparse.ArgumentParser(
        description="Run a command while monitoring NPU utilization",
        usage="%(prog)s [--interval SEC] [--] command..."
    )
    parser.add_argument('--interval', type=float, default=1.0,
                        help='Sampling interval in seconds (default: 1.0)')
    args = parser.parse_args(monitor_args)

    if not cmd:
        cmd = ['python', 'demo/demo_offline.py', 'test_files', '--finegrained']

    print(f"{'='*60}")
    print(f"  NPU Monitor + Command Runner")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Interval: {args.interval}s")
    print(f"{'='*60}")
    print()

    monitor = NpuMonitor(interval=args.interval)
    monitor.start()

    t_start = time.perf_counter()
    try:
        result = subprocess.run(cmd, env=os.environ.copy())
        returncode = result.returncode
    except KeyboardInterrupt:
        returncode = -1
        print("\n\n  ⚠️  Interrupted by user")
    finally:
        monitor.stop()
        elapsed = time.perf_counter() - t_start

    print()  # newline after the live output
    print(f"\n{'='*60}")
    print(f"  Done: {elapsed:.1f}s (exit code: {returncode})")
    print(f"{'='*60}")
    print(monitor.summary())


if __name__ == '__main__':
    main()

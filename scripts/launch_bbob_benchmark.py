#!/usr/bin/env python3
"""
BBOB Benchmark Launcher with RAM cleanup and process management.

This script:
1. Runs all generated TOML experiments in parallel (nohup)
2. Cleans up RAM after each experiment
3. Tracks progress and completion
4. Logs all output for analysis

Usage:
    python launch_bbob_benchmark.py --toml-dir examples/_DEMO_COMPOSER/bbob_benchmark \
                                   --max-parallel 2 \
                                   --cleanup-ram
"""

import argparse
import subprocess
import time
from pathlib import Path
from typing import List
import sys
import os
import psutil


class BBOBLauncher:
    """Manage BBOB experiment execution with RAM cleanup."""
    
    def __init__(
        self,
        toml_dir: Path,
        max_parallel: int = 1,
        cleanup_ram: bool = True,
        output_dir: Path = None,
    ):
        self.toml_dir = Path(toml_dir)
        self.max_parallel = max_parallel
        self.should_cleanup_ram = cleanup_ram
        
        if output_dir is None:
            output_dir = self.toml_dir.parent
        
        self.output_dir = Path(output_dir)
        self.nohup_dir = self.output_dir / "nohup"
        self.log_dir = self.output_dir / "logs"
        self.status_log = self.log_dir / "completion.log"
        
        # Create directories
        self.nohup_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.running_processes = {}
        self.completed = []
        self.failed = []
    
    def get_toml_files(self) -> List[Path]:
        """Get all TOML files in the directory."""
        toml_files = sorted(self.toml_dir.glob("bbob_fn*.toml"))
        return toml_files
    
    def cleanup_ram(self):
        """Attempt to clean up RAM (requires sudo for full effect)."""
        if not self.should_cleanup_ram:
            return
        
        try:
            # Try to drop caches with sudo (may fail if not available)
            # Use non-blocking subprocess call with short timeout
            subprocess.run(
                "sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1",
                shell=True,
                timeout=3,  # Reduced timeout
                capture_output=True,
            )
        except subprocess.TimeoutExpired:
            # If sudo times out, just continue (not critical)
            pass
        except Exception as e:
            # Silent fail for other errors
            pass
        
        try:
            # Python-level garbage collection (always works)
            import gc
            gc.collect()
        except Exception:
            pass
        
        time.sleep(0.5)
    
    def get_memory_usage(self) -> str:
        """Get current memory usage."""
        try:
            mem = psutil.virtual_memory()
            return f"{mem.percent}% ({mem.used / (1024**3):.1f}GB / {mem.total / (1024**3):.1f}GB)"
        except Exception:
            return "N/A"
    
    def run_experiment(self, toml_file: Path) -> subprocess.Popen:
        """Launch a single experiment in nohup."""
        experiment_name = toml_file.stem
        nohup_file = self.nohup_dir / f"{experiment_name}.out"
        
        # Python command to run Composer with proper error handling
        python_cmd = (
            f"import sys; "
            f"from malthusjax.composer import Composer; "
            f"try: "
            f"    result = Composer.from_toml(r'{toml_file.absolute()}'); "
            f"    print('✓ Experiment {experiment_name} completed'); "
            f"    sys.exit(0); "
            f"except Exception as e: "
            f"    import traceback; "
            f"    print(f'ERROR: {{type(e).__name__}}: {{e}}', file=sys.stderr); "
            f"    traceback.print_exc(file=sys.stderr); "
            f"    sys.exit(1);"
        )
        
        # Launch in background using nohup
        with open(nohup_file, "w") as nohup_out:
            process = subprocess.Popen(
                ["nohup", "python", "-c", python_cmd],
                stdout=nohup_out,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setpgrp,  # Detach from parent process group
            )
        
        return process
    
    def launch_all(self):
        """Launch all experiments with parallelism control."""
        toml_files = self.get_toml_files()
        
        if not toml_files:
            print(f"❌ No TOML files found in {self.toml_dir}")
            return
        
        print(f"🚀 Starting BBOB benchmark suite")
        print(f"   TOML directory: {self.toml_dir}")
        print(f"   Max parallel runs: {self.max_parallel}")
        print(f"   RAM cleanup: {self.should_cleanup_ram}")
        print()
        print(f"📊 Found {len(toml_files)} experiments")
        print()
        
        current_idx = 0
        
        while current_idx < len(toml_files) or self.running_processes:
            # Start new experiments up to max_parallel
            while (
                current_idx < len(toml_files)
                and len(self.running_processes) < self.max_parallel
            ):
                toml_file = toml_files[current_idx]
                print(f"[{current_idx + 1}/{len(toml_files)}] 🟢 Launching: {toml_file.name}")
                print(f"     Memory: {self.get_memory_usage()}")
                
                process = self.run_experiment(toml_file)
                self.running_processes[toml_file.name] = {
                    "process": process,
                    "pid": process.pid,
                    "start_time": time.time(),
                }
                
                current_idx += 1
            
            # Check for completed processes
            completed_names = []
            for exp_name, proc_info in list(self.running_processes.items()):
                proc = proc_info["process"]
                
                if proc.poll() is not None:
                    # Process finished
                    exit_code = proc.returncode
                    elapsed = time.time() - proc_info["start_time"]
                    
                    if exit_code == 0:
                        status = "✅ PASSED"
                        self.completed.append(exp_name)
                    else:
                        status = f"❌ FAILED (exit code: {exit_code})"
                        self.failed.append(exp_name)
                    
                    print(
                        f"     {status} ({elapsed:.1f}s) [PID {proc_info['pid']}]"
                    )
                    
                    # Show error output if failed
                    if exit_code != 0:
                        nohup_file = self.nohup_dir / f"{exp_name}.out"
                        if nohup_file.exists():
                            content = nohup_file.read_text()
                            if content.strip():
                                print(f"     ⚠ Full output from {nohup_file}:")
                                for line in content.split('\n'):
                                    if line.strip():
                                        print(f"       {line}")
                            else:
                                print(f"     ⚠ Nohup file is empty: {nohup_file}")
                        else:
                            print(f"     ⚠ No nohup file found: {nohup_file}")
                    
                    # Log completion
                    with open(self.status_log, "a") as f:
                        f.write(
                            f"{exp_name}: {status} ({elapsed:.1f}s) at {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                        )
                    
                    # RAM cleanup
                    if self.should_cleanup_ram:
                        print(f"     🧹 Cleaning up RAM...")
                        self.cleanup_ram()
                        print(f"     Memory after cleanup: {self.get_memory_usage()}")
                    
                    completed_names.append(exp_name)
            
            # Remove completed processes
            for exp_name in completed_names:
                del self.running_processes[exp_name]
            
            # Sleep briefly before checking again
            if self.running_processes:
                time.sleep(5)
        
        # Final summary
        print()
        print("=" * 70)
        print("BENCHMARK SUITE COMPLETED")
        print("=" * 70)
        print(f"✅ Completed: {len(self.completed)}")
        print(f"❌ Failed: {len(self.failed)}")
        print()
        
        if self.failed:
            print("Failed experiments:")
            for name in self.failed:
                print(f"  - {name}")
            print()
        
        print(f"Logs: {self.log_dir}")
        print(f"Nohup output: {self.nohup_dir}")
        print(f"Completion log: {self.status_log}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Launch BBOB benchmark suite with parallel execution and RAM cleanup"
    )
    parser.add_argument(
        "--toml-dir",
        type=Path,
        default=Path("examples/_DEMO_COMPOSER/bbob_benchmark"),
        help="Directory containing TOML files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for logs and nohup files (default: parent of toml-dir)",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="Maximum number of parallel experiments",
    )
    parser.add_argument(
        "--cleanup-ram",
        action="store_true",
        default=True,
        help="Clean up RAM after each experiment",
    )
    parser.add_argument(
        "--no-cleanup-ram",
        action="store_false",
        dest="cleanup_ram",
        help="Disable RAM cleanup",
    )
    
    args = parser.parse_args()
    
    launcher = BBOBLauncher(
        toml_dir=args.toml_dir,
        max_parallel=args.max_parallel,
        cleanup_ram=args.cleanup_ram,
        output_dir=args.output_dir,
    )
    
    launcher.launch_all()


if __name__ == "__main__":
    main()

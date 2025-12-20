#!/usr/bin/env python3
"""
Launcher script for ResNet training that properly handles background execution.
This avoids the nohup + ProcessPoolExecutor deadlock issue.
"""
import subprocess
import sys
import os

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    training_script = os.path.join(script_dir, "train_oracle_resnet.py")
    log_file = "/tmp/resnet_training.log"

    # Build command
    cmd = [
        sys.executable, "-u", training_script,
        "--games", "10000",
        "--epochs", "120",
        "--curriculum",
        "--batch-size", "4096",
        "--workers", "20",  # Use 20 of 28 cores, leave some for system
    ]

    # Open log file
    with open(log_file, "w") as log:
        # Start process with proper stdin handling
        process = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,  # This is key - don't inherit stdin
            start_new_session=True,  # Detach from parent session
        )
        print(f"Training started with PID: {process.pid}")
        print(f"Log file: {log_file}")
        print(f"Monitor with: tail -f {log_file}")

if __name__ == "__main__":
    main()

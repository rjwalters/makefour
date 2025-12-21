#!/usr/bin/env python3
"""
Launcher script for CNN training that properly handles background execution.
"""
import subprocess
import sys
import os

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    training_script = os.path.join(script_dir, "train_oracle_cnn.py")
    log_file = "/tmp/cnn_training.log"

    # Build command - CNN trains faster, use more games
    cmd = [
        sys.executable, "-u", training_script,
        "--games", "15000",
        "--epochs", "200",
        "--curriculum",
        "--batch-size", "2048",
        "--workers", "20",
    ]

    # Open log file
    with open(log_file, "w") as log:
        process = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        print(f"CNN Training started with PID: {process.pid}")
        print(f"Log file: {log_file}")
        print(f"Monitor with: tail -f {log_file}")

if __name__ == "__main__":
    main()

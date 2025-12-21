#!/bin/bash

# Activate Environment (Ensure we are in the right state)
# source ~/.bashrc  # Uncomment if needed
# conda activate mjx_env_gpu # Uncomment if needed

echo "🚀 STARTING GECCO 2026 GRAND PRIX"
echo "=================================================="

# Function to run a benchmark with memory cleaning
run_task() {
    config_file=$1
    echo ""
    echo "--------------------------------------------------"
    echo "▶️  STARTING TASK: $config_file"
    echo "--------------------------------------------------"
    
    # 1. Run the Python Process
    # This isolates memory. When this line finishes, python dies and frees all VRAM.
    python -m benchmarks.cli "$config_file"
    
    # 2. Check Exit Status
    if [ $? -ne 0 ]; then
        echo "❌ ERROR: $config_file FAILED. Stopping pipeline."
        exit 1
    fi
    
    # 3. System Level Cleanup (The "Janitor")
    echo "🧹 Cleaning Memory..."
    sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true # Try to clear page cache (requires sudo, usually skipped safely)
    sleep 5 # Cool down
}

# --- THE SCHEDULE ---

run_task "configs/run_sphere.toml"
run_task "configs/run_rosenbrock.toml"
run_task "configs/run_ellipsoidal_rotated.toml"
run_task "configs/run_rastrigin.toml"
run_task "configs/run_schaffers_f7.toml"

echo ""
echo "=================================================="
echo "🏆 GRAND PRIX COMPLETE. ALL RESULTS SAVED."
echo "=================================================="
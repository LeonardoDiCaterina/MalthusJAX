# Environment and cluster context

Snapshot context: 9 November 2026, Europe/Rome. Distinguish observed versions from desired environments. No approved publication GPU lock exists yet.

## Local workstation

Framework checkout is `/Users/leonardodicaterina/Documents/GitHub/MalthusJAX`; sibling paper checkout is `../MalthusJAX_Paper`. Shell is zsh. Local metadata snapshot is `local_environment.json`, captured with `/opt/anaconda3/bin/python3`: Python 3.12.5 on Apple Silicon macOS, JAX/jaxlib 0.11.0, NumPy 2.5.1. This is the CPU-tested environment, not the H100 environment. `requirements/analysis-cpu-candidate.txt` pins the scoped tested analysis dependencies; it is a candidate, not a publication approval.

Package declarations require Python >=3.12 and JAX >=0.11,<0.13. Python >=3.12 is a declared range; verification here is scoped to Python 3.12. The three satellites are under `packages/malthus_optimize`, `packages/malthus_gp`, `packages/malthus_neat`. Use `git status` before modifications and retain the remediation branch. Paper working-tree modifications belong to the user and must not be overwritten.

## Observed cluster

- Host: `DAH2` / `dah2.isegi.unl.pt`; Slurm 22.05.8; Linux x86_64, kernel 6.1.0-38-amd64. User: `20240485`.
- Current checkout: `/home/20240485/post_submission`. Older jobs used `/home/20240485/post_submission_remediation`.
- Activated Conda environment: `post_sub`; Python executable `/home/20240485/.conda/envs/post_sub/bin/python`.
- Job 1630 metadata: Python 3.12.16, custom JAX 0.12.0.1, GPU backend. Earlier job 1516 also reported jaxlib 0.12.0.1 and NumPy 2.5.4; these latter versions have not been independently confirmed for 1630/1649. Do not silently label them a fresh complete lock.
- Two NVIDIA H100 NVL devices, approximately 95,830 MiB each. Observed driver 580.65.06; `nvidia-smi` displayed CUDA 13.0 (driver capability display, not proof of the exact JAX CUDA runtime package).
- Slurm node reports `Gres=gpu:h100:2`, 224 logical CPUs and configured RAM 2,000,000 MB; these are node resources, not entitlement for a job. GPU/cpu partitions exist.
- GPU 0 (PCI `00000000:59:00.0`) was running another user's vLLM workload, ~87,453 MiB and high utilization. Physical GPU 1 (PCI `00000000:C2:00.0`) was idle before submission. These are historical snapshots, not current availability.
- Job 1649 explicitly reported `SLURM_JOB_GPUS=1`, `CUDA_VISIBLE_DEVICES=0`: Slurm's physical allocation was GPU 1, exposed locally as CUDA 0. Preserve this remapping. The guarded submission stops if Slurm does not identify GPU ID 1; it does not forcibly reserve that physical ID.

No SSH/remote tool session was established. The user copies commands into the cluster terminal and returns output. Never run a cluster `sbatch` command on the local Mac assuming it submits remotely.

## Existing launch paths

`submit_timing_validation.sh` requests one H100, eight CPUs, 32 GB, 30 minutes; uses the submitting activated environment and preserves Slurm's CUDA visibility. Required arguments are `--expect-revision FULL_HASH --output NEW_DIRECTORY`; default expected backend/device are GPU/H100. Logs: `logs/mjx_pt5_timing_JOB_ID.out/.err`. Frozen diagnostic: `configs/timing/pt5.toml`. Do not change it to create publication evidence or rerun the gate by default.

`submit_empirical_curves.sh` requests one GPU, 16 CPUs, 64 GB, eight hours by default, forwarding generator arguments. It tries `post_sub`/`malthusjax` activation, so always verify logged Python/import paths. Generator requires explicit `--out NEW_DIRECTORY/empirical_curves.json`. Five datasets include expensive optional physics workloads. This path does not yet implement the scoped new planning campaign.

Create `logs/` **before** `sbatch`, since Slurm opens output files before script execution. Use new result directories. Standard Slurm commands: `squeue -j JOB_ID`, `sacct -j JOB_ID`, `scancel JOB_ID`, `tail -f logs/NAME_JOB_ID.out`. Job numbers and GPU occupancy change; inspect rather than reusing old allocations. No planned campaign jobs have been submitted.

## Metadata still needed for publication

Capture an exact environment/import/source snapshot within the intended allocation, plus package source revisions for editable/git dependencies and input artifact hashes. Retain original PT5 report/trials/compilation logs and smoke raw manifests on the cluster; only pasted summaries are local. Record precision/XLA flags, GPU UUID, device assignment, CPU resources, cache policy, clocks/power and neighboring workload observations.

`capture_cluster_environment.py` provides a bounded metadata collector. It prints JSON or writes a new file with exclusive creation. Default collection does not initialize JAX; `--probe-jax` additionally resolves actual JAX backend/devices and should be used inside an allocated compute job. It reads only selected nonsecret environment keys and package metadata; no full environment dump or credential-bearing remote URLs. It does not replace complete run manifests or an approved lock.

```bash
# Optional future collection inside the intended allocated job; not launched here.
python docs/handoff/capture_cluster_environment.py --probe-jax \
  --output results/NEW_RUN/environment_capture.json
```

Docker Desktop was unavailable locally. User explicitly said **leave Docker verification to CI**. Hosted CI/container results have not been inspected; preserve this uncertainty.

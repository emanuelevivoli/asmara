# How to run SLURM

```
sbatch -o <path_to_stdout_logs_of_slurm> \
       -e <path_to_stderr_logs_of_slurm> \
       <my_script>.sh
```
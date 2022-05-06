# %% imports
# NOTE: `pip install pyro-ppl` to use FULLYBAYESIAN (SAASBO)
from submitit import AutoExecutor

import cloudpickle as pickle

from utils.matbench import matbench_fold_GPEI, collect_results, task, savepath, dummy

print(f"dummy: {dummy}")

# %% submission
log_folder = "log_ax/%j"
walltime = 28 * 60  # 4320  # 4320 min == 3 days
# partition, account = ["notchpeak-gpu-guest", "owner-gpu-guest"]
partition, account = ["notchpeak-gpu", "notchpeak-gpu"]
# partition, account = ["notchpeak-guest", "owner-guest"]
executor = AutoExecutor(folder=log_folder)
executor.update_parameters(
    timeout_min=walltime,
    slurm_partition=partition,
    slurm_gpus_per_task=1,
    slurm_mem_per_gpu=6000,
    slurm_cpus_per_gpu=4,
    slurm_additional_parameters={"account": account},
)
jobs = executor.map_array(matbench_fold_GPEI, task.folds)  # sbatch array
job_ids = [job.job_id for job in jobs]
# https://www.hpc2n.umu.se/documentation/batchsystem/job-dependencies
job_ids_str = ":".join(job_ids)  # e.g. "3937257_0:3937257_1:..."

with open("jobs.pkl", "wb") as f:
    pickle.dump(jobs, f)


collect_folder = "log_matbench/%j"
walltime = 10
collector = AutoExecutor(folder=collect_folder)
collector.update_parameters(
    timeout_min=walltime,
    slurm_partition=partition,
    slurm_additional_parameters={
        "account": account,
        "dependency": f"afterok:{job_ids_str}",
    },
)
collector_job = collector.submit(collect_results, False)  # sbatch array

print(
    f"Waiting for submission jobs ({job_ids_str}) to complete before running collector job ({collector_job.job_id}). Use the matbench output file that will be saved to {savepath} after all jobs have run."
)

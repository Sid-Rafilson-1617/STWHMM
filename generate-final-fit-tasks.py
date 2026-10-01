from pathlib import Path
import numpy as np

task_file = Path("tasks/hmm_tasks.txt")
task_file.parent.mkdir(exist_ok=True)

data_dir = '/mnt/home/srafilson/code/STWHMM/data'
data = f"{data_dir}/maze_clipped_spike_counts_pyr_only_Bilat_R02_20251106.npz"

save_dir = "results_pyr_only_20ms_10iters_H_mask/full_fit"
Path(save_dir).mkdir(parents=True, exist_ok=True)


nStates = [101]
nTaus = [5]

restarts = range(1000)
#start = 386
#end = 2000
#restarts = np.arange(start, end)


with task_file.open("w") as f:
    for M in nStates:
        for S in nTaus:
            for restart in restarts:
                seed = 100000 + 1000 * M + restart

                cmd = (
                    "python -m fit_one_full "
                    f"--data {data} "
                    f"--save-dir {save_dir} "
                    f"--nStates {M} "
                    f"--nTaus {S} "
                    f"--restart {restart} "
                    f"--seed {seed} "
                    f"--Niters 20"

                )

                f.write(cmd + "\n")
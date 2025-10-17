import os
import time
import subprocess
import torch
import shutil
import json
import copy
import glob

from mmeval.registery import series_mapping, series_infer_env_mapping
from mmeval.utils.argparser import parse_args


def get_series(model_name: str):
    for series, models in series_mapping.items():
        if model_name in models:
            return series
    raise ValueError(f"Model {model_name} not found in registery.")
    

if __name__ == "__main__":
    args = parse_args()
    model_name_or_path = args.model_name_or_path
    series = get_series(model_name_or_path.split("/")[-1])
    
    infer_file = series_infer_env_mapping[series]["infer_file"]
    infer_env = series_infer_env_mapping[series]["env"]
    parallel_per_task  = args.parallel_per_task
    gpu_per_parallel = args.gpu_per_parallel
    total_gpus = torch.cuda.device_count()

    if os.path.exists(os.path.join(args.out_dir, "result.json")) and args.resume:
        # exit and return success
        print(f"🌟 Result file {os.path.join(args.out_dir, 'result.json')} exists. Task finished, exiting...")
        exit(0)
    
    if gpu_per_parallel > total_gpus:
        raise RuntimeError(
            f"Minimal {gpu_per_parallel} GPUs per parallel is required, but only {total_gpus} GPUs available"
        )
        

    # Initialize GPU pool and task list
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        available_gpus = [int(x.strip()) for x in cvd.split(",") if x.strip()]
    else:
        available_gpus = list(range(total_gpus))
        
    running_tasks = []
    next_rank = 0
    ###########################################################################################################################################
    # FIX: how to resume when parallel_per_task is different from last time? All cache are saved in cache.db with conflict management.
    ############################################################################################################################################
    
    ###################################################################################################
    # TODO: more fine-grained scheduling (first run a check on the vmem of parallel task)
    #       Then, schedule based on rest free vmem of each gpu
    ###################################################################################################

    ###################################################################################################
    # TODO: run multiple datasets as a sequence of tasks, and schedule them all
    ###################################################################################################

    # Loop until all shards are scheduled and completed
    while next_rank < parallel_per_task or running_tasks:
        # Launch new jobs if resources and tasks remain
        while (next_rank < parallel_per_task
               and len(running_tasks) < parallel_per_task
               and len(available_gpus) >= gpu_per_parallel):
            
            # Allocate GPUs for this job
            allocated = available_gpus[:gpu_per_parallel]
            available_gpus = available_gpus[gpu_per_parallel:]

            # Prepare environment for subprocess
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, allocated))

            cur_args = copy.deepcopy(args)
            cur_args.rank = next_rank
            def append_args(ns):
                for key in vars(ns):
                    val = getattr(ns, key)
                    # Skip None flags entirely
                    if val is None:
                        continue
                    # Boolean flags: --flag (only if True)
                    if isinstance(val, bool):
                        if val:
                            cmd.append(f"--{key}")
                    else:
                        cmd.extend([f"--{key}", str(val)])
            # Build the conda-run command
            # if infer_env is a directory, then use -p
            if os.path.isdir(infer_env):
                cmd = [
                    "conda", "run", "--no-capture-output", "-p", infer_env, 
                    "python", os.path.join("mmeval/infer", infer_file),
                ]
            # if infer_env is env name, then use -n
            else:
                cmd = [
                    "conda", "run", "--no-capture-output", "-n", infer_env, 
                    "python", os.path.join("mmeval/infer", infer_file),
                ]
                
            append_args(cur_args)

            #########################################################
            # TODO: show the process of subprocess (need to discuss)
            #########################################################

            # Launch the process
            proc = subprocess.Popen(cmd, env=env, stdout=None, stderr=None)
            running_tasks.append({"proc": proc, "gpus": allocated, "rank": next_rank})
            print(f"🌟 Launched shard {next_rank} on GPUs {allocated}")
            next_rank += 1

        # Poll running tasks and reclaim GPUs as they finish
        for task in running_tasks:
            if task["proc"].poll() is not None:  # process has exited
                running_tasks.remove(task)
                available_gpus.extend(task["gpus"])
                print(f"✅ Completed shard {task['rank']}, freed GPUs {task['gpus']}")
                break
        else:
            # No task finished just now — wait briefly before polling again
            time.sleep(5)

    print("All inference shards completed.")

    # TODO: need to check if the cache is complete
    
    from mmeval.utils.sqlitkv import SQLiteKVStore
    cache = SQLiteKVStore(os.path.join(args.out_dir, "cache.db"))
    result = cache.dump_dict()
    result = sorted(list(result.values()), key=lambda x: x["eval-id"])

    with open(os.path.join(args.out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=4)
    
    # delete cache.db
    os.remove(os.path.join(args.out_dir, "cache.db"))



        

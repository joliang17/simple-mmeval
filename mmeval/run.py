import os
import time
import subprocess
import json
import copy

from mmeval.registry import series_mapping, series_infer_env_mapping
from mmeval.utils.argparser import parse_args, BOOL_DEFAULTS


def get_series(model_name: str):
    for series, models in series_mapping.items():
        if model_name in models:
            return series
    raise ValueError(f"Model {model_name} not found in registry.")


def get_series_from_model_path(model_name_or_path: str):
    model_name = model_name_or_path.rstrip("/").split("/")[-1]
    try:
        return get_series(model_name)
    except ValueError:
        pass

    config_path = os.path.join(model_name_or_path, "config.json")
    if os.path.isdir(model_name_or_path) and os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)

        model_type = config.get("model_type")
        architectures = config.get("architectures") or []
        if model_type == "qwen2_vl" or "Qwen2VLForConditionalGeneration" in architectures:
            return "qwenvl2"
        if model_type == "qwen2_5_vl" or "Qwen2_5_VLForConditionalGeneration" in architectures:
            return "qwenvl2d5"
        if model_type == "qwen3_vl" or any("Qwen3VL" in arch for arch in architectures):
            return "qwen3_vl"

    raise ValueError(f"Model {model_name_or_path} not found in registry.")
    

if __name__ == "__main__":
    args = parse_args()

    model_name_or_path = args.model_name_or_path
    series = get_series_from_model_path(model_name_or_path)
    
    infer_file = series_infer_env_mapping[series]["infer_file"]
    infer_env = series_infer_env_mapping[series]["env"]
    parallel_per_task  = args.parallel_per_task
    gpu_per_parallel = args.gpu_per_parallel

    # Initialize GPU pool: respect CUDA_VISIBLE_DEVICES
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        # Treat CUDA_VISIBLE_DEVICES entries as opaque strings (supports numeric IDs and GPU/MIG UUIDs)
        available_gpus = [x.strip() for x in cvd.split(",") if x.strip()]
        total_gpus = len(available_gpus)
    else:
        # Get GPU count via nvidia-smi when CVD not set (handles MIG)
        try:
            result = subprocess.run(
                ["nvidia-smi", "-L"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                total_gpus = 0
            else:
                lines = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
                # MIG mode: CUDA sees MIG instances, not parent GPUs; count "MIG ... Device X:" lines
                mig_lines = [ln for ln in lines if "MIG" in ln and "Device" in ln]
                if mig_lines:
                    total_gpus = len(mig_lines)
                else:
                    # Normal mode: count "GPU X:" lines
                    total_gpus = len([ln for ln in lines if ln.startswith("GPU") and ":" in ln])
        except (FileNotFoundError, subprocess.TimeoutExpired):
            # nvidia-smi is unavailable or unresponsive; assume no GPUs
            total_gpus = 0
        available_gpus = list(range(total_gpus))

    if os.path.exists(os.path.join(args.out_dir, "result.json")) and args.resume:
        # exit and return success
        print(f"🌟 Result file {os.path.join(args.out_dir, 'result.json')} exists. Task finished, exiting...")
        exit(0)
    
    if gpu_per_parallel > total_gpus:
        raise RuntimeError(
            f"Minimal {gpu_per_parallel} GPUs per parallel is required, but only {total_gpus} GPUs available"
        )
        

    running_tasks = []
    failed_tasks = []  # Track failed tasks
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
                    # Boolean flags depend on their default values.
                    if isinstance(val, bool):
                        default = BOOL_DEFAULTS.get(key)
                        if default is True:
                            if val is False:
                                cmd.append(f"--no_{key}")
                        elif default is False:
                            if val is True:
                                cmd.append(f"--{key}")
                        else:
                            if val is True:
                                cmd.append(f"--{key}")
                            elif val is False:
                                cmd.append(f"--no_{key}")
                    else:
                        cmd.extend([f"--{key}", str(val)])
            # Build the command
            if args.no_conda:
                cmd = [
                    "python", os.path.join("mmeval/infer", infer_file),
                ]
            # if infer_env is a directory, then use -p
            elif os.path.isdir(infer_env):
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
            exit_code = task["proc"].poll()
            if exit_code is not None:  # process has exited
                running_tasks.remove(task)
                available_gpus.extend(task["gpus"])
                
                if exit_code != 0:
                    print(f"❌ Shard {task['rank']} failed with exit code {exit_code}, freed GPUs {task['gpus']}")
                    failed_tasks.append({"rank": task["rank"], "exit_code": exit_code})
                else:
                    print(f"✅ Completed shard {task['rank']}, freed GPUs {task['gpus']}")
                break
        else:
            # No task finished just now — wait briefly before polling again
            time.sleep(5)
    
    # Check for any failed tasks
    if failed_tasks:
        print(f"\n❌ Inference failed! {len(failed_tasks)} shard(s) failed:")
        for task in failed_tasks:
            print(f"   - Shard {task['rank']} (exit code: {task['exit_code']})")
        exit(1)

    print("All inference shards completed.")

    # Collect results from cache and save to JSON
    from mmeval.utils.sqlitkv import SQLiteKVStore
    
    cache = SQLiteKVStore(os.path.join(args.out_dir, "cache.db"))
    result = cache.dump_dict()
    result = sorted(list(result.values()), key=lambda x: x["eval-id"])

    with open(os.path.join(args.out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=4)
    
    # delete cache.db
    os.remove(os.path.join(args.out_dir, "cache.db"))
    print(f"✅ Results saved to {os.path.join(args.out_dir, 'result.json')} ({len(result)} samples)")



        

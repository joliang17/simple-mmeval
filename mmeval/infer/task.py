import os
import sys
import tqdm
import json

from mmeval.data import load_dataset
from mmeval.utils.res_handler import ResponseHandler

class Task:
    def __init__(self, args):
        self.max_retry = args.max_retry
        self.max_retry_sample = args.max_retry_sample
        self.out_dir = args.out_dir
        self.rank = getattr(args, 'rank', 0)  # Store rank for display
        os.makedirs(self.out_dir, exist_ok=True)

        self.res_handler = ResponseHandler(args)
        
        prev_cache = {}
        try:
            tmp_dir = os.path.join(self.out_dir, "tmp")
            fpath = os.path.join(tmp_dir, "prev_run_cache.json")
            if os.path.exists(fpath):
                with open(fpath, "r") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        prev_cache.update(data)
                    elif isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and "eval-id" in item:
                                prev_cache[item["eval-id"]] = item
        except Exception as e:
            print(f"Warning: Failed to load cache file {fpath}: {e}")
            
        self.dataset = load_dataset(args)
        self.dataset.setup_parallel(prev_cache)
        
        self.load_model(args)

    def run_sample(self, sample:dict):
        raise NotImplementedError("run_sample is not implemented")

    def run_batch(self, samples:list):
        raise NotImplementedError("run_batch is not implemented")

    def load_model(self, args):
        raise NotImplementedError("load_model is not implemented")

    def inference_dataset(self):
        run_count = 0
        while not self.res_handler.check_complete(self.dataset) and run_count < self.max_retry:
            run_count += 1

            # Format tqdm progress bar
            for sample in tqdm.tqdm(
                self.dataset, 
                total=len(self.dataset), 
                desc=f"Shard {self.rank} ({len(self.dataset)} samples)",
                ncols=80,                   
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                colour='green',
                position=self.rank,              
                leave=True,                  
                file=sys.stdout,            
                mininterval=0.1,           
                maxinterval=1.0,       
                smoothing=0.3         
            ):
                
                cnt = 0
                try:
                    ret = self.run_sample(sample)
                    self.res_handler.save(ret)

                except Exception as e:
                    tqdm.tqdm.write(f"[Shard {self.rank}] ❌ Error: {e}")
                    cnt += 1
                    if cnt >= self.max_retry_sample:
                        tqdm.tqdm.write(f"[Shard {self.rank}] ⚠️  Max retries reached, skipping sample")
                        continue
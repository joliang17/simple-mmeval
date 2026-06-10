import os
import sys
import tqdm
import traceback

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
        
        self.dataset = load_dataset(args)
        self.dataset.setup_parallel(self.res_handler.cache)
        
        self.load_model(args)

    def run_sample(self, sample:dict):
        raise NotImplementedError("run_sample is not implemented")

    def run_batch(self, samples:list):
        raise NotImplementedError("run_batch is not implemented")

    def load_model(self, args):
        raise NotImplementedError("load_model is not implemented")

    def inference_dataset(self):
        """Run inference on the dataset with two-level retry logic."""
        incomplete_count = -1  # Sentinel: -1 means loop never ran
        try:
            for retry_count_dataset in range(1, self.max_retry + 1):
                # Create progress bar for samples
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
                    # Skip samples already in cache
                    if self.res_handler.in_cache(sample["eval-id"]):
                        continue
                    
                    # Retry loop with a per-sample limit. A shared counter makes
                    # one backend failure cause every later sample to be skipped.
                    retry_limit = self.max_retry_sample
                    while True:
                        try:
                            ret = self.run_sample(sample)
                            self.res_handler.save(ret)
                            break  # Success, move to next sample
                        except Exception as e:
                            retry_limit -= 1
                            tqdm.tqdm.write(f"[Shard {self.rank}] ❌ Error (retry left: {retry_limit}/{self.max_retry_sample}):\n{e}")
                            tqdm.tqdm.write(f"[Shard {self.rank}] 📋 Traceback:\n{traceback.format_exc()}")
                            if retry_limit <= 0:
                                tqdm.tqdm.write(f"[Shard {self.rank}] ⚠️ Retry limit reached, skipping sample {sample.get('eval-id', 'unknown')}")
                                break  # No more retries, skip this sample
                            tqdm.tqdm.write(f"[Shard {self.rank}] ⚠️ Retrying sample {sample.get('eval-id', 'unknown')}...")
                
                # Check completion after each dataset pass
                incomplete_count = self.res_handler.check_complete(self.dataset)
                if incomplete_count == 0:
                    break  # All complete
                tqdm.tqdm.write(f"[Shard {self.rank}] ⚠️ Pass {retry_count_dataset}/{self.max_retry}: {incomplete_count} samples incomplete, retrying...")
            
            # Exit with error if incomplete (reuse last incomplete_count)
            if incomplete_count == -1:
                tqdm.tqdm.write(f"[Shard {self.rank}] ❌ Inference failed.")
                self.res_handler.flush()
                sys.exit(1)
            elif incomplete_count > 0:
                tqdm.tqdm.write(f"[Shard {self.rank}] ❌ Inference incomplete: {incomplete_count} samples failed after {self.max_retry} passes.")
                self.res_handler.flush()
                sys.exit(1)
                
        finally:
            # Ensure all pending results are saved before exit
            self.res_handler.flush()

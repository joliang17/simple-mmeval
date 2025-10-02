import os
import json
import numpy as np
from PIL import Image

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class ResponseHandler:
    def __init__(self, args):
        self.rank = args.rank
        self.tmp_dir = os.path.join(args.out_dir, "tmp")    
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.cache_file = os.path.join(self.tmp_dir, f"result_{self.rank}.json.tmp")
        self.output_file = os.path.join(self.tmp_dir, f"result_{self.rank}.json")
        self.save_freq = args.save_freq
        self.load_cache()

    def load_cache(self):
        if os.path.exists(self.cache_file):
            self.cache = json.load(open(self.cache_file, "r"))
        else:
            self.cache = {}

    @property
    def length(self):
        return len(self.cache)

    def in_cache(self, id_:str):
        return id_ in self.cache
    
    def check_complete(self, dataset):
        if len(self.cache) != len(dataset):
            return False
        else:
            for sample in dataset:
                if not self.in_cache(sample["eval-id"]):
                    return False
            print(f"📖 [Shard {self.rank}] Results completed. Saved output file {self.output_file}.")
            if os.path.exists(self.cache_file):
                print(f"🗑️  Deleting cache file {self.cache_file}.")
                os.remove(self.cache_file)
            self._dump_result()
            return True
        
    def save(self, result:dict):
        assert "eval-id" in result, "eval-id is required"
        assert "response" in result, f"no model response for sample {result}"
        
        if "media" in result and result["media"]:
            # Check if any item is a PIL Image, if so, remove the entire media section
            if any(isinstance(item, Image.Image) for item in result["media"]):
                del result["media"]
               
        self.cache[result["eval-id"]] = result
        if len(self.cache) % self.save_freq == 0:
            self._dump_cache()

    def _dump_cache(self):
        json.dump(self.cache, open(self.cache_file, "w"), indent=4, cls=NumpyEncoder)

    def _dump_result(self):
        ret = [self.cache[k] for k in sorted(self.cache.keys())]
        json.dump(ret, open(self.output_file, "w",), indent=4, cls=NumpyEncoder)

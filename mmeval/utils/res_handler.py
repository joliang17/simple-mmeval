import os
import json
import numpy as np
from PIL import Image
from mmeval.utils.sqlitkv import SQLiteKVStore

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
        self.save_freq = args.save_freq
        self.kvstore = SQLiteKVStore(os.path.join(args.out_dir, f"cache.db"))
        self.load_cache()

    def load_cache(self):
        tmp = self.kvstore.dump_dict()
        self.cache = {int(k): v for k, v in tmp.items()}


    @property
    def length(self):
        return len(self.cache)

    def in_cache(self, id_:str):
        return id_ in self.cache
    
    def check_complete(self, dataset):

        for sample in dataset:
            if not self.in_cache(sample["eval-id"]):
                return False
        print(f"📖 [Shard {self.rank}] Results completed. Saved output file {self.output_file}.")
        return True
        
    def save(self, result:dict):
        assert "eval-id" in result, "eval-id is required"
        assert "response" in result, f"no model response for sample {result}"
        
        if "media" in result and result["media"]:
            # Check if any item is a PIL Image, if so, remove the entire media section
            if any(isinstance(item, Image.Image) for item in result["media"]):
                del result["media"]
        
        if len(self.cache) % self.save_freq == 0:
            self.kvstore.put(str(result["eval-id"]), result)

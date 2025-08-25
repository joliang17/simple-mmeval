import os
import re
import json
from PIL import Image

from mmeval.data.base import BaseDataset

class LocalJSONDataset(BaseDataset):
    """Dataset class for loading local JSON files.
    """
    
    def __init__(self, args):
        self.data_file = args.infile
        self.img_dir = args.img_dir
        super().__init__(args)

    def _load_raw_data(self, args):
        data = json.load(open(self.data_file, "r"))

        data_list = []
        for i, sample in enumerate(data):
            assert "eval-id" not in sample, "eval-id already exists"
            sample["eval-id"] = i
            sample["media"] = [os.path.join(self.img_dir, f) for f in sample["media"]]
            data_list.append(sample)
        
        return data_list
    
    def _process_sample(self, idx: int):
        sample = self._raw_dataset[idx]
        placeholder_list = re.findall(r"<(?:video|image)>", sample["prompt"])
        assert len(placeholder_list) == len(sample["media"]), "Number of media placeholders does not match number of media files"
        
        sample["media"] = [
            self.load_image(f) if placeholder == "<image>" else f
            for placeholder, f in zip(placeholder_list, sample["media"])
        ]

        return sample
    

    def __repr__(self):
        if self.parallel_per_task > 1:
            return f"local@{self.data_file.split('/')[-1]}(rank={self.rank}/{self.parallel_per_task}, local={len(self)}, global={self.global_length})"
        else:
            return f"local@{self.data_file.split('/')[-1]}(samples={len(self)})"
    
    def __str__(self):
        return self.__repr__()
    
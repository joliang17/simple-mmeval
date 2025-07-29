import re
import copy
import torch
from transformers import AutoModel, AutoTokenizer
from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args

class TaskRunner(Task):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self, args):
        model_path = f"THUDM/{args.model_name_or_path}"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()

    def run_sample(self, sample: dict):
        ori_sample = copy.deepcopy(sample)
        question, modality = self.parse_input(sample)

        if not self.args.score_target:
            ori_sample["response"] = self._generate_response(question, sample["media_dir"], None)
        else:
            ori_sample.update(self._score_choices(question, sample["media_dir"], None, sample))

        return ori_sample

    def _generate_response(self, text, image_inputs, video_inputs):
        response, history = self.model.chat(self.tokenizer, image_inputs[0], text, history=[])
        return response

    def _score_choices(self, text, image_inputs, video_inputs, sample):
        raise NotImplementedError("Scoring is not supported yet.")

    def parse_input(self, sample:dict):
        question = sample["prompt"]
        # extract placeholder
        placeholders = re.findall(r'<[^>]*>', question)
        assert len(placeholders) == 1, f"VideoLLaMA2 supports one image or video, but got {len(placeholder)}"
        
        placeholder = placeholders[0]
        modality = "image" if placeholder == constants.image else "video"

        # remove the placeholder in the question
        question = question.replace(placeholder, "").strip()

        return question, modality



if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()
import copy
import torch
from PIL import Image

from parrot.model.parrot_arch import ParrotMetaForCausalLM
from parrot.utils.constants import DEFAULT_IMAGE_TOKEN
from parrot.utils.utils import disable_torch_init
from parrot.utils.mm_utils import process_images

from mmeval.infer.task import Task
from mmeval.utils.argparser import parse_args, parse_model_kwargs, parse_gen_kwargs


class TaskRunner(Task):
	def __init__(self, args):
		self.args = args
		self.dtype = getattr(args, "dtype") or torch.bfloat16
		self.device = torch.device(getattr(args, "device", "cpu"))

		self.default_model_kwargs = {
			"low_cpu_mem_usage": True,
		}
		self.model_kwargs = parse_model_kwargs(args, self.default_model_kwargs)

		self.default_gen_kwargs = {}
		self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)

		super().__init__(args)

	def load_model(self, args):
		disable_torch_init()
		
		model_name = 'parrot_qwen2'
		mm_vision_tower = 'openai/clip-vit-large-patch14-336'
		self.model, self.tokenizer, self.conversation_formatter = ParrotMetaForCausalLM.build(
			model_name,
			args.model_name_or_path,
			mm_vision_tower=mm_vision_tower,
			torch_dtype=self.dtype,
			**self.model_kwargs,
		)
		self.model = self.model.to(self.device)
		self.image_processor = self.model.get_vision_tower().image_processor

	def _get_input(self, text, media):
		# Count existing <image> tokens and align with number of images
		token = DEFAULT_IMAGE_TOKEN
		num_tokens = text.count(token)
		if num_tokens == 0 and media:
			text = token + '\n' + text
			num_tokens = 1

		images = []
		for i in range(min(num_tokens, len(media))):
			im = media[i]
			if isinstance(im, Image.Image):
				images.append(im.convert('RGB'))
			else:
				images.append(Image.open(im).convert('RGB'))

		prompt, input_ids = self.conversation_formatter.format_query(text)
		image_tensor = process_images(images, self.image_processor, self.model.config)
		return prompt, input_ids, image_tensor

	def _generate_response(self, text, media):
		prompt, input_ids, image_tensor = self._get_input(text=text, media=media)

		input_ids = input_ids.to(device=self.device).unsqueeze(0)
		if image_tensor is not None:
			image_tensor = image_tensor.to(dtype=self.model.dtype, device=self.device)

		gen_kwargs = dict(self.gen_kwargs)
		max_new_tokens = gen_kwargs.pop('max_new_tokens', 1024)
		gen_kwargs.pop('eos_token_id', None)

		with torch.inference_mode():
			output_ids = self.model.generate(
				input_ids,
				images=image_tensor,
				repetition_penalty=None,
				max_new_tokens=max_new_tokens,
				eos_token_id=self.tokenizer.eos_token_id,
				**gen_kwargs,
			)

		input_token_len = input_ids.shape[1]
		output = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
		return prompt, output.strip()

	def run_sample(self, sample: dict):
		ori_sample = copy.deepcopy(sample)
		text = ori_sample["prompt"]
		media = ori_sample['media']

		if not self.args.score_target:
			_, output = self._generate_response(text, media)
			ori_sample["response"] = output
		else:
			pass

		return ori_sample


if __name__ == "__main__":
	args = parse_args()
	model_evaluator = TaskRunner(args)
	model_evaluator.inference_dataset()


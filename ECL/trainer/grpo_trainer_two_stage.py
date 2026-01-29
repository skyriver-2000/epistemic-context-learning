import os
import re
import copy
from typing import Any, Union
from contextlib import nullcontext
from torch.nn.utils.rnn import pad_sequence as pad
from accelerate.utils import broadcast_object_list, gather, gather_object
import torch
import torch.nn as nn
from ECL.utils.logging_utils import setup_logger
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import Trainer, AutoModelForSequenceClassification, PreTrainedModel, AutoTokenizer
from transformers.utils import is_flash_attn_2_available
from trl import GRPOTrainer
from trl.models import unwrap_model_for_generation
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template, prepare_multimodal_messages
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.import_utils import is_vllm_available
from trl.trainer.utils import (
    nanmax,
    nanmin,
    nanstd,
    pad,
    shuffle_sequence_dict,
    split_pixel_values_by_grid,
    split_tensor_dict,
    truncate_with_protected_tokens,
    unsplit_pixel_values_by_grid,
)

if is_vllm_available():
    from vllm import SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

SAVE_PATH = "grpo_trainer_two_stage_logs"

logger = setup_logger()

def parse_summary(text: str) -> str:
    pattern = r"^<think>(.*?)</think>\s*<summary>(.*?)</summary>\s*\Z"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(2).strip(), True
    return text, False


def parse_summary_llama(text: str) -> str:
    """Parse summary from Llama format with #### Summary separator."""
    # Look for #### Summary section
    if '#### Summary' in text:
        parts = text.split('#### Summary')
        return parts[-1].strip(), True
    return text, False


def parse_summary_qwen3(text: str) -> str:
    """Parse summary from Qwen3 format - no summary tags, just content after [/think]."""
    pattern = r"^\[think\](.*?)\[/think\]\s*\n\n(.*?)\s*\Z"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(2).strip(), True
    return text, False


def parse_summary_ds(text: str) -> str:
    """Parse summary from DS format - no summary tags, just content after </think>."""
    pattern = r"^<think>(.*?)</think>\s*\n\n(.*?)\s*\Z"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(2).strip(), True
    return text, False


def extract_answer(text: str) -> str:
    """
    Extract reasoning and answer from text with format:
    <think>$REASONING_PROCESS</think>
    <answer>$ANSWER</answer>
    
    Returns combined format:
    $REASONING_PROCESS
    
    $ANSWER
    """
    # Extract reasoning process from <think></think> tags
    think_pattern = r"<think>(.*?)</think>"
    think_match = re.search(think_pattern, text, re.DOTALL)
    reasoning = think_match.group(1).strip() if think_match else ""
    
    # Extract answer from <answer></answer> tags
    answer_pattern = r"<answer>(.*?)</answer>"
    answer_match = re.search(answer_pattern, text, re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else ""
    
    # Combine reasoning and answer with double newline
    if reasoning and answer:
        return f"{reasoning}\n\n{answer}"
    elif answer:
        return answer
    elif reasoning:
        return reasoning
    else:
        return ""


def extract_answer_qwen3(text: str) -> str:
    """
    Extract reasoning and answer from Qwen3 format:
    [think]$REASONING_PROCESS[/think]
    
    $ANSWER (no tags)
    
    Returns combined format:
    $REASONING_PROCESS
    
    $ANSWER
    """
    # Extract reasoning process from [think][/think] tags
    think_pattern = r"\[think\](.*?)\[/think\]"
    think_match = re.search(think_pattern, text, re.DOTALL)
    reasoning = think_match.group(1).strip() if think_match else ""
    
    # Extract answer - everything after [/think]\n\n
    answer_pattern = r"\[/think\]\s*\n\n(.*?)\s*\Z"
    answer_match = re.search(answer_pattern, text, re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else ""
    
    # Combine reasoning and answer with double newline
    if reasoning and answer:
        return f"{reasoning}\n\n{answer}"
    elif answer:
        return answer
    elif reasoning:
        return reasoning
    else:
        return ""


def extract_answer_ds(text: str) -> str:
    """
    Extract reasoning and answer from DS format:
    <think>$REASONING_PROCESS</think>
    
    $ANSWER (no tags)
    
    Returns combined format:
    $REASONING_PROCESS
    
    $ANSWER
    """
    # Extract reasoning process from <think></think> tags
    think_pattern = r"<think>(.*?)</think>"
    think_match = re.search(think_pattern, text, re.DOTALL)
    reasoning = think_match.group(1).strip() if think_match else ""
    
    # Extract answer - everything after </think>\n\n
    answer_pattern = r"</think>\s*\n\n(.*?)\s*\Z"
    answer_match = re.search(answer_pattern, text, re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else ""
    
    # Combine reasoning and answer with double newline
    if reasoning and answer:
        return f"{reasoning}\n\n{answer}"
    elif answer:
        return answer
    elif reasoning:
        return reasoning
    else:
        return ""

class GRPOTrainerTwoStage(GRPOTrainer):
    
    def __init__(self, *args, **kwargs):
        # Check whether stage 2 training with OR is helpful
        self.ablation_stage2_training = kwargs.pop("ablation_stage2_training", False)
        self.decouple_internal_belief = kwargs.pop("decouple_internal_belief", False)

        model_init_kwargs = kwargs["args"].model_init_kwargs or {}
        reward_funcs_stage1 = kwargs.pop("reward_funcs_stage1", None)
        reward_processing_classes_stage1 = kwargs.pop("reward_processing_classes_stage1", None)
        
        # Detect if using Llama, Qwen3, or DS model format
        self.is_llama_format = kwargs.pop("is_llama_format", False)
        self.is_qwen3_format = kwargs.pop("is_qwen3_format", False)
        self.is_ds_format = kwargs.pop("is_ds_format", False)
        
        if not isinstance(reward_funcs_stage1, list):
            reward_funcs_stage1 = [reward_funcs_stage1]
        self.reward_func_names_stage1 = []
        for i, reward_func in enumerate(reward_funcs_stage1):
            if isinstance(reward_func, str):
                reward_funcs_stage1[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
            if isinstance(reward_funcs_stage1[i], nn.Module):  # Use Module over PretrainedModel for compat w/ compiled models
                self.reward_func_names_stage1.append(reward_funcs_stage1[i].config._name_or_path.split("/")[-1])
            else:
                self.reward_func_names_stage1.append(reward_funcs_stage1[i].__name__)
        self.reward_funcs_stage1 = reward_funcs_stage1

        # Reward weights
        if kwargs["args"].reward_weights_stage1 is not None:
            if len(kwargs["args"].reward_weights_stage1) != len(reward_funcs_stage1):
                raise ValueError(
                    f"Number of reward weights ({len(kwargs['args'].reward_weights_stage1)}) must match number of reward "
                    f"functions ({len(reward_funcs_stage1)})"
                )
            self.reward_weights_stage1 = torch.tensor(kwargs["args"].reward_weights_stage1, dtype=torch.float32)
        else:
            self.reward_weights_stage1 = torch.ones(len(reward_funcs_stage1), dtype=torch.float32)

        # Reward processing class
        if reward_processing_classes_stage1 is None:
            reward_processing_classes_stage1 = [None] * len(reward_funcs_stage1)
        elif not isinstance(reward_processing_classes_stage1, list):
            reward_processing_classes_stage1 = [reward_processing_classes_stage1]
        if len(reward_processing_classes_stage1) != len(reward_funcs_stage1):
            raise ValueError(
                f"The number of reward processing classes ({len(reward_processing_classes_stage1)}) must match the number of "
                f"reward functions ({len(reward_funcs_stage1)})."
            )

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes_stage1, reward_funcs_stage1)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes_stage1[i] = reward_processing_class

        self.reward_processing_classes_stage1 = reward_processing_classes_stage1
        
        # Detect if using PRR (Peer Recognition Reward) - check if any stage1 reward function contains "peer_recognition"
        self.use_prr = any("peer_recognition" in name.lower() for name in self.reward_func_names_stage1)

        super().__init__(*args, **kwargs)
        
    @profiling_decorator
    def _prepare_inputs(
        self, generation_batch: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        # Prepares inputs for model training/evaluation by managing completion generation and batch handling.
        # During training:
        #   - Receives the local generation batch (Per-GPU batch size × steps per generation)
        #     from the modified training dataloader instead of the standard local batch
        #   - Generates completions once for the entire generation batch and splits it into batches of size
        #     `per_device_train_batch_size`
        #   - Buffers these completions and returns the appropriate slice for the current accumulation step
        #   - Optimizes by regenerating completions only periodically (every steps_per_generation * num_iterations)
        # During evaluation:
        #   - The input is treated as a standard local batch (no accumulation, no multiple iterations)
        #   - Completions are generated for each batch without buffering or reuse
        # Returns a single local batch in both cases.

        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                # self._buffered_inputs=None can occur when resuming from a checkpoint
                generation_batch = self._generate_and_score_completions(generation_batch)
                if "stage1" in generation_batch and "stage2" in generation_batch:
                    generation_batch["stage1"] = split_pixel_values_by_grid(generation_batch["stage1"])
                    generation_batch["stage2"] = split_pixel_values_by_grid(generation_batch["stage2"])
                    if "raw" in generation_batch:
                        generation_batch["raw"] = split_pixel_values_by_grid(generation_batch["raw"])
                    generation_batch["stage1"] = shuffle_sequence_dict(generation_batch["stage1"])
                    generation_batch["stage2"] = shuffle_sequence_dict(generation_batch["stage2"])
                    if "raw" in generation_batch:
                        generation_batch["raw"] = shuffle_sequence_dict(generation_batch["raw"])
                    stage1_batches = split_tensor_dict(generation_batch["stage1"], self.args.steps_per_generation)
                    stage2_batches = split_tensor_dict(generation_batch["stage2"], self.args.steps_per_generation)
                    if "raw" in generation_batch:
                        raw_batches = split_tensor_dict(generation_batch["raw"], self.args.steps_per_generation)
                        self._buffered_inputs = [{
                            "stage1": unsplit_pixel_values_by_grid(stage1_batch),
                            "stage2": unsplit_pixel_values_by_grid(stage2_batch),
                            "raw": unsplit_pixel_values_by_grid(raw_batch)
                        } for stage1_batch, stage2_batch, raw_batch in zip(stage1_batches, stage2_batches, raw_batches)]
                    else:
                        self._buffered_inputs = [{
                            "stage1": unsplit_pixel_values_by_grid(stage1_batch),
                            "stage2": unsplit_pixel_values_by_grid(stage2_batch)
                        } for stage1_batch, stage2_batch in zip(stage1_batches, stage2_batches)]
                else:
                    generation_batch["stage2"] = split_pixel_values_by_grid(generation_batch["stage2"])
                    generation_batch["stage2"] = shuffle_sequence_dict(generation_batch["stage2"])
                    generation_batches = split_tensor_dict(generation_batch["stage2"], self.args.steps_per_generation)
                    self._buffered_inputs = [{"stage2": unsplit_pixel_values_by_grid(batch)} for batch in generation_batches]
            inputs = self._buffered_inputs[self._step % self.args.steps_per_generation]
            self._step += 1
        else:
            # In evaluation, there is neither batch grouping for generation, nor multiple iterations, hence
            # local generation batch == local eval batch
            inputs = self._generate_and_score_completions(generation_batch)
        return inputs
    
    def _generate_stage1_completions(
        self, prompts: list, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> tuple[list[str], dict[str, Union[torch.Tensor, Any]]]:
        """
        Generate completions for stage 1 prompts.
        Returns a tuple of (completion_texts, output_dict) where output_dict contains the same fields as stage 2.
        """
        device = self.accelerator.device
        
        # Handle images if present
        kwargs = {}
        has_images = "image" in inputs[0]
        if has_images:
            images = [example.get("image") for example in inputs]
            kwargs = {"images": [[img] for img in images]}
            for prompt in prompts:
                if isinstance(prompt, list):  # conversational data
                    prepare_multimodal_messages(prompt, num_images=1)
        
        # Apply chat template
        prompts_text = [maybe_apply_chat_template({"prompt": p}, self.processing_class, enable_thinking=False)["prompt"] for p in prompts]
        
        # Tokenize prompts
        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
            **kwargs,
        )
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        
        # Handle max_prompt_length truncation
        if self.max_prompt_length is not None:
            protected = [self.image_token_id, self.vision_start_token_id, self.vision_end_token_id]
            protected = [token for token in protected if token is not None]
            prompt_ids, prompt_mask = truncate_with_protected_tokens(
                prompt_ids, prompt_mask, self.max_prompt_length, protected
            )
            
            prompts_text = self.processing_class.batch_decode(
                prompt_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            prompts_text = [re.sub(rf"^({re.escape(self.pad_token)})+", "", text) for text in prompts_text]
            
            # Handle image tokens
            if self.image_token is not None:
                escaped_img_token = re.escape(self.image_token)
                if re.search(escaped_img_token, self.processing_class.chat_template):
                    prompts_text = [
                        re.sub(rf"({escaped_img_token})+", self.image_token, text) for text in prompts_text
                    ]
                else:
                    if self.vision_end_token_id is not None:
                        escaped_eoi_token = re.escape(
                            self.processing_class.tokenizer.decode([self.vision_end_token_id])
                        )
                        prompts_text = [
                            re.sub(rf"({escaped_img_token})+{escaped_eoi_token}", "", text) for text in prompts_text
                        ]
                    else:
                        prompts_text = [re.sub(rf"({escaped_img_token})+", "", text) for text in prompts_text]
        
        # Generate completions using vLLM or transformers
        if self.use_vllm:
            if self.vllm_mode == "colocate" and self.args.vllm_enable_sleep_mode:
                torch.cuda.empty_cache()
                self.llm.wake_up()
            
            # Update vLLM weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step
            
            if self.vllm_mode == "server":
                all_prompts_text = gather_object(prompts_text)
                if has_images:
                    all_images = gather_object(images)
                
                if self.accelerator.is_main_process:
                    # Use only unique prompts (no duplicates for stage 1)
                    with profiling_context(self, "vLLM.generate_stage1"):
                        output = self.vllm_client.generate(
                            prompts=all_prompts_text,
                            images=all_images if has_images else None,
                            n=1,  # Generate only 1 completion per prompt for stage 1
                            repetition_penalty=self.repetition_penalty,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            top_k=-1 if self.top_k is None else self.top_k,
                            min_p=0.0 if self.min_p is None else self.min_p,
                            max_tokens=self.max_completion_length,
                            guided_decoding_regex=self.guided_decoding_regex,
                            generation_kwargs=self.args.generation_kwargs,
                        )
                        payload = (output["completions"], output["completion_ids"], output["logprobs"])
                else:
                    payload = None
                
                obj_list = [payload]
                broadcast_object_list(obj_list, from_process=0)
                completions_text, completion_ids_list, all_logprobs = obj_list[0]
                
                # Get local slice
                process_slice = slice(
                    self.accelerator.process_index * len(prompts),
                    (self.accelerator.process_index + 1) * len(prompts),
                )
                completions_text = completions_text[process_slice]
                completion_ids_list = completion_ids_list[process_slice]
                all_logprobs = all_logprobs[process_slice]
            
            elif self.vllm_mode == "colocate":
                if self.guided_decoding_regex:
                    guided_decoding = GuidedDecodingParams(regex=self.guided_decoding_regex)
                else:
                    guided_decoding = None
                
                generation_kwargs = {
                    "n": 1,
                    "repetition_penalty": self.repetition_penalty,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": -1 if self.top_k is None else self.top_k,
                    "min_p": 0.0 if self.min_p is None else self.min_p,
                    "max_tokens": self.max_completion_length,
                    "guided_decoding": guided_decoding,
                    "logprobs": 0,
                }
                if self.args.generation_kwargs is not None:
                    generation_kwargs.update(self.args.generation_kwargs)
                sampling_params = SamplingParams(**generation_kwargs)
                
                if self.vllm_tensor_parallel_size > 1:
                    orig_size = len(prompts_text)
                    gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                    torch.distributed.all_gather_object(gathered_prompts, prompts_text, group=self.tp_group)
                    all_prompts_text = [p for sublist in gathered_prompts for p in sublist]
                    
                    if has_images:
                        gathered_images = [None for _ in range(self.vllm_tensor_parallel_size)]
                        torch.distributed.all_gather_object(gathered_images, images, group=self.tp_group)
                        all_images = [img for sublist in gathered_images for img in sublist]
                    else:
                        all_images = None
                else:
                    all_prompts_text = prompts_text
                    all_images = images if has_images else None
                
                if has_images and all_images:
                    vllm_inputs = []
                    for prompt, image in zip(all_prompts_text, all_images):
                        if image is not None:
                            vllm_inputs.append({"prompt": prompt, "multi_modal_data": {"image": image}})
                        else:
                            vllm_inputs.append(prompt)
                else:
                    vllm_inputs = all_prompts_text
                
                with profiling_context(self, "vLLM.generate_stage1"):
                    all_outputs = self.llm.generate(vllm_inputs, sampling_params=sampling_params, use_tqdm=False)
                
                completions_text = [output.outputs[0].text for output in all_outputs]
                completion_ids_list = [output.outputs[0].token_ids for output in all_outputs]
                all_logprobs = [
                    [next(iter(lp.values())).logprob for lp in output.outputs[0].logprobs]
                    for output in all_outputs
                ]
                
                if self.vllm_tensor_parallel_size > 1:
                    local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                    tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                    completions_text = completions_text[tp_slice]
                    completion_ids_list = completion_ids_list[tp_slice]
                    all_logprobs = all_logprobs[tp_slice]
                
                if self.args.vllm_enable_sleep_mode:
                    self.llm.sleep(level=1)
            
            # Convert to tensors and pad for vLLM
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
            completion_ids = pad(completion_ids, padding_value=self.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            sampling_per_token_logps = [
                torch.tensor(logprobs, device=device, dtype=torch.float32) for logprobs in all_logprobs
            ]
            sampling_per_token_logps = pad(sampling_per_token_logps, padding_value=0.0)
        
        elif self.use_transformers_paged:
            paged_prompt_inputs = self.processing_class(text=prompts_text, **kwargs)
            previous_attn = self.model_wrapped.config._attn_implementation
            
            if is_flash_attn_2_available():
                self.model_wrapped.config._attn_implementation = "paged_attention"
            else:
                self.model_wrapped.config._attn_implementation = "sdpa_paged"
            
            with (
                profiling_context(self, "transformers.generate_batch_stage1"),
                unwrap_model_for_generation(
                    self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
            ):
                if self.args.bf16:
                    unwrapped_model.to(torch.bfloat16)
                elif self.args.fp16:
                    unwrapped_model.to(torch.float16)
                with torch.inference_mode():
                    all_outputs = unwrapped_model.generate_batch(
                        paged_prompt_inputs.input_ids, generation_config=self.generation_config, progress_bar=False
                    )
            
            completion_ids = [output.generated_tokens for output in all_outputs.values()]
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
            prompt_ids_for_concat = [torch.tensor(ids, device=device) for ids in paged_prompt_inputs.input_ids]
            prompt_ids_for_concat = pad(prompt_ids_for_concat, padding_value=self.pad_token_id, padding_side="left")
            prompt_completion_ids = torch.cat([prompt_ids_for_concat, completion_ids], dim=1)
            completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
            self.model_wrapped.config._attn_implementation = previous_attn
        
        else:
            # Regular generation
            with (
                profiling_context(self, "transformers.generate_stage1"),
                unwrap_model_for_generation(
                    self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
            ):
                prompt_inputs["input_ids"], prompt_inputs["attention_mask"] = prompt_ids, prompt_mask
                prompt_completion_ids = unwrapped_model.generate(
                    **prompt_inputs, generation_config=self.generation_config, disable_compile=True
                )
            
            prompt_length = prompt_ids.size(1)
            completion_ids = prompt_completion_ids[:, prompt_length:]
            completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        
        # Build the same output structure as stage 2
        # Mask everything after the first EOS token
        is_eos = completion_ids == self.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        
        # Convert tensor to a list of lists of token IDs for reward calculation
        completion_ids_list = [row[mask_row].tolist() for row, mask_row in zip(completion_ids, completion_mask.bool())]
        
        # Concatenate prompt_mask with completion_mask
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        
        # Compute logprobs and other metrics for stage 1
        mode = "train" if self.model.training else "eval"
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size
        logits_to_keep = completion_ids.size(1)
        
        with torch.no_grad():
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self.args.gradient_accumulation_steps % generate_every != 0 or (
                self.use_vllm and self.vllm_importance_sampling_correction
            ):
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    pixel_values=prompt_inputs.get("pixel_values"),
                    image_grid_thw=prompt_inputs.get("image_grid_thw"),
                    pixel_attention_mask=prompt_inputs.get("pixel_attention_mask"),
                    image_sizes=prompt_inputs.get("image_sizes"),
                )
            else:
                old_per_token_logps = None
                
            # Compute the importance sampling ratio when using vLLM, to correct for potential distribution mismatch
            if self.use_vllm and self.vllm_importance_sampling_correction:
                importance_sampling_ratio = torch.exp(old_per_token_logps - sampling_per_token_logps)
                importance_sampling_ratio = torch.clamp(
                    importance_sampling_ratio, max=self.vllm_importance_sampling_cap
                )
            
            # Compute reference model logprobs if needed
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        pixel_values=prompt_inputs.get("pixel_values"),
                        image_grid_thw=prompt_inputs.get("image_grid_thw"),
                        pixel_attention_mask=prompt_inputs.get("pixel_attention_mask"),
                        image_sizes=prompt_inputs.get("image_sizes"),
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                            pixel_values=prompt_inputs.get("pixel_values"),
                            image_grid_thw=prompt_inputs.get("image_grid_thw"),
                            pixel_attention_mask=prompt_inputs.get("pixel_attention_mask"),
                            image_sizes=prompt_inputs.get("image_sizes"),
                        )
            else:
                ref_per_token_logps = None
        
        # Just decode completions, don't calculate rewards yet
        # Rewards will be calculated later in _generate_and_score_completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        
        # Build output dict for stage 1 (without rewards)
        output_stage1 = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "attention_mask": attention_mask,
            "prompts": prompts,  # Store prompts for later reward calculation
            "completions_text": completions_text,  # Store completions text for later reward calculation
            "completion_ids_list": completion_ids_list,  # Store completion_ids_list for later reward calculation
        }
        
        if old_per_token_logps is not None:
            output_stage1["old_per_token_logps"] = old_per_token_logps
        if self.use_vllm and self.vllm_importance_sampling_correction:
            output_stage1["importance_sampling_ratio"] = importance_sampling_ratio
        if ref_per_token_logps is not None:
            output_stage1["ref_per_token_logps"] = ref_per_token_logps
        if "pixel_values" in prompt_inputs:
            output_stage1["pixel_values"] = prompt_inputs["pixel_values"]
        if "image_grid_thw" in prompt_inputs:
            output_stage1["image_grid_thw"] = prompt_inputs["image_grid_thw"]
        if "pixel_attention_mask" in prompt_inputs:
            output_stage1["pixel_attention_mask"] = prompt_inputs["pixel_attention_mask"]
        if "image_sizes" in prompt_inputs:
            output_stage1["image_sizes"] = prompt_inputs["image_sizes"]
        
        return completions_text, output_stage1
    
    def _generate_raw_completions(
        self, prompts: list, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> tuple[list[str], dict[str, Union[torch.Tensor, Any]]]:
        """
        Generate completions for raw prompts (only current question, no peers/history).
        Returns a tuple of (extracted_answers, output_dict) similar to stage 1 and stage 2.
        """
        device = self.accelerator.device
        
        # Handle images if present
        kwargs = {}
        has_images = "image" in inputs[0]
        if has_images:
            images = [example.get("image") for example in inputs]
            kwargs = {"images": [[img] for img in images]}
            for prompt in prompts:
                if isinstance(prompt, list):  # conversational data
                    prepare_multimodal_messages(prompt, num_images=1)
        
        # Apply chat template
        prompts_text = [maybe_apply_chat_template({"prompt": p}, self.processing_class, enable_thinking=False)["prompt"] for p in prompts]
        
        # Tokenize prompts
        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
            **kwargs,
        )
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        
        # Handle max_prompt_length truncation
        if self.max_prompt_length is not None:
            protected = [self.image_token_id, self.vision_start_token_id, self.vision_end_token_id]
            protected = [token for token in protected if token is not None]
            prompt_ids, prompt_mask = truncate_with_protected_tokens(
                prompt_ids, prompt_mask, self.max_prompt_length, protected
            )
            
            prompts_text = self.processing_class.batch_decode(
                prompt_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            prompts_text = [re.sub(rf"^({re.escape(self.pad_token)})+", "", text) for text in prompts_text]
            
            # Handle image tokens
            if self.image_token is not None:
                escaped_img_token = re.escape(self.image_token)
                if re.search(escaped_img_token, self.processing_class.chat_template):
                    prompts_text = [
                        re.sub(rf"({escaped_img_token})+", self.image_token, text) for text in prompts_text
                    ]
                else:
                    if self.vision_end_token_id is not None:
                        escaped_eoi_token = re.escape(
                            self.processing_class.tokenizer.decode([self.vision_end_token_id])
                        )
                        prompts_text = [
                            re.sub(rf"({escaped_img_token})+{escaped_eoi_token}", "", text) for text in prompts_text
                        ]
                    else:
                        prompts_text = [re.sub(rf"({escaped_img_token})+", "", text) for text in prompts_text]
        
        # Generate completions using vLLM or transformers (same as stage1)
        if self.use_vllm:
            if self.vllm_mode == "colocate" and self.args.vllm_enable_sleep_mode:
                torch.cuda.empty_cache()
                self.llm.wake_up()
            
            # Update vLLM weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step
            
            if self.vllm_mode == "server":
                all_prompts_text = gather_object(prompts_text)
                if has_images:
                    all_images = gather_object(images)
                
                if self.accelerator.is_main_process:
                    # Use only unique prompts (no duplicates for raw)
                    with profiling_context(self, "vLLM.generate_raw"):
                        output = self.vllm_client.generate(
                            prompts=all_prompts_text,
                            images=all_images if has_images else None,
                            n=1,  # Generate only 1 completion per prompt for raw
                            repetition_penalty=self.repetition_penalty,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            top_k=-1 if self.top_k is None else self.top_k,
                            min_p=0.0 if self.min_p is None else self.min_p,
                            max_tokens=self.max_completion_length,
                            guided_decoding_regex=self.guided_decoding_regex,
                            generation_kwargs=self.args.generation_kwargs,
                        )
                        payload = (output["completions"], output["completion_ids"], output["logprobs"])
                else:
                    payload = None
                
                obj_list = [payload]
                broadcast_object_list(obj_list, from_process=0)
                completions_text, completion_ids_list, all_logprobs = obj_list[0]
                
                # Get local slice
                process_slice = slice(
                    self.accelerator.process_index * len(prompts),
                    (self.accelerator.process_index + 1) * len(prompts),
                )
                completions_text = completions_text[process_slice]
                completion_ids_list = completion_ids_list[process_slice]
                all_logprobs = all_logprobs[process_slice]
            
            elif self.vllm_mode == "colocate":
                if self.guided_decoding_regex:
                    guided_decoding = GuidedDecodingParams(regex=self.guided_decoding_regex)
                else:
                    guided_decoding = None
                
                generation_kwargs = {
                    "n": 1,
                    "repetition_penalty": self.repetition_penalty,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": -1 if self.top_k is None else self.top_k,
                    "min_p": 0.0 if self.min_p is None else self.min_p,
                    "max_tokens": self.max_completion_length,
                    "guided_decoding": guided_decoding,
                    "logprobs": 0,
                }
                if self.args.generation_kwargs is not None:
                    generation_kwargs.update(self.args.generation_kwargs)
                sampling_params = SamplingParams(**generation_kwargs)
                
                if self.vllm_tensor_parallel_size > 1:
                    orig_size = len(prompts_text)
                    gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                    torch.distributed.all_gather_object(gathered_prompts, prompts_text, group=self.tp_group)
                    all_prompts_text = [p for sublist in gathered_prompts for p in sublist]
                    
                    if has_images:
                        gathered_images = [None for _ in range(self.vllm_tensor_parallel_size)]
                        torch.distributed.all_gather_object(gathered_images, images, group=self.tp_group)
                        all_images = [img for sublist in gathered_images for img in sublist]
                    else:
                        all_images = None
                else:
                    all_prompts_text = prompts_text
                    all_images = images if has_images else None
                
                if has_images and all_images:
                    vllm_inputs = []
                    for prompt, image in zip(all_prompts_text, all_images):
                        if image is not None:
                            vllm_inputs.append({"prompt": prompt, "multi_modal_data": {"image": image}})
                        else:
                            vllm_inputs.append(prompt)
                else:
                    vllm_inputs = all_prompts_text
                
                with profiling_context(self, "vLLM.generate_raw"):
                    all_outputs = self.llm.generate(vllm_inputs, sampling_params=sampling_params, use_tqdm=False)
                
                completions_text = [output.outputs[0].text for output in all_outputs]
                completion_ids_list = [output.outputs[0].token_ids for output in all_outputs]
                all_logprobs = [
                    [next(iter(lp.values())).logprob for lp in output.outputs[0].logprobs]
                    for output in all_outputs
                ]
                
                if self.vllm_tensor_parallel_size > 1:
                    local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                    tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                    completions_text = completions_text[tp_slice]
                    completion_ids_list = completion_ids_list[tp_slice]
                    all_logprobs = all_logprobs[tp_slice]
                
                if self.args.vllm_enable_sleep_mode:
                    self.llm.sleep(level=1)
            
            # Convert to tensors and pad for vLLM
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
            completion_ids = pad(completion_ids, padding_value=self.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            sampling_per_token_logps = [
                torch.tensor(logprobs, device=device, dtype=torch.float32) for logprobs in all_logprobs
            ]
            sampling_per_token_logps = pad(sampling_per_token_logps, padding_value=0.0)
        
        elif self.use_transformers_paged:
            paged_prompt_inputs = self.processing_class(text=prompts_text, **kwargs)
            previous_attn = self.model_wrapped.config._attn_implementation
            
            if is_flash_attn_2_available():
                self.model_wrapped.config._attn_implementation = "flash_attention_2"
            else:
                self.model_wrapped.config._attn_implementation = "sdpa"
            
            with (
                profiling_context(self, "transformers.generate_batch_raw"),
                unwrap_model_for_generation(
                    self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
            ):
                all_outputs = unwrapped_model.generate(
                    paged_prompt_inputs,
                    batch_size=len(prompts_text),
                    max_new_tokens=self.max_completion_length,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
            
            completion_ids = [output.generated_tokens for output in all_outputs.values()]
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
            prompt_ids_for_concat = [torch.tensor(ids, device=device) for ids in paged_prompt_inputs.input_ids]
            prompt_ids_for_concat = pad(prompt_ids_for_concat, padding_value=self.pad_token_id, padding_side="left")
            prompt_completion_ids = torch.cat([prompt_ids_for_concat, completion_ids], dim=1)
            completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
            self.model_wrapped.config._attn_implementation = previous_attn
        
        else:
            # Regular generation
            with (
                profiling_context(self, "transformers.generate_raw"),
                unwrap_model_for_generation(
                    self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
            ):
                prompt_inputs["input_ids"], prompt_inputs["attention_mask"] = prompt_ids, prompt_mask
                prompt_completion_ids = unwrapped_model.generate(
                    **prompt_inputs, generation_config=self.generation_config, disable_compile=True
                )
            
            prompt_length = prompt_ids.size(1)
            completion_ids = prompt_completion_ids[:, prompt_length:]
            completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        
        # Mask everything after the first EOS token
        is_eos = completion_ids == self.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        
        # Convert tensor to a list of lists of token IDs for reward calculation
        completion_ids_list = [row[mask_row].tolist() for row, mask_row in zip(completion_ids, completion_mask.bool())]
        
        # Concatenate prompt_mask with completion_mask
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        
        # Compute logprobs and other metrics for raw completions
        mode = "train" if self.model.training else "eval"
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size
        logits_to_keep = completion_ids.size(1)
        
        with torch.no_grad():
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self.args.gradient_accumulation_steps % generate_every != 0 or (
                self.use_vllm and self.vllm_importance_sampling_correction
            ):
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    pixel_values=prompt_inputs.get("pixel_values"),
                    image_grid_thw=prompt_inputs.get("image_grid_thw"),
                    pixel_attention_mask=prompt_inputs.get("pixel_attention_mask"),
                    image_sizes=prompt_inputs.get("image_sizes"),
                )
            else:
                old_per_token_logps = None
                
            # Compute the importance sampling ratio when using vLLM, to correct for potential distribution mismatch
            if self.use_vllm and self.vllm_importance_sampling_correction:
                importance_sampling_ratio = torch.exp(old_per_token_logps - sampling_per_token_logps)
                importance_sampling_ratio = torch.clamp(
                    importance_sampling_ratio, max=self.vllm_importance_sampling_cap
                )
            
            # Compute reference model logprobs if needed
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        pixel_values=prompt_inputs.get("pixel_values"),
                        image_grid_thw=prompt_inputs.get("image_grid_thw"),
                        pixel_attention_mask=prompt_inputs.get("pixel_attention_mask"),
                        image_sizes=prompt_inputs.get("image_sizes"),
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                            pixel_values=prompt_inputs.get("pixel_values"),
                            image_grid_thw=prompt_inputs.get("image_grid_thw"),
                            pixel_attention_mask=prompt_inputs.get("pixel_attention_mask"),
                            image_sizes=prompt_inputs.get("image_sizes"),
                        )
            else:
                ref_per_token_logps = None
        
        # Decode completions and extract answers
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        # Use appropriate extract function based on model format
        # Note: Check DS before Llama because deepseek-distill-llama contains both keywords
        if self.is_qwen3_format:
            extracted_answers = [extract_answer_qwen3(text) for text in completions_text]
        elif self.is_ds_format:
            extracted_answers = [extract_answer_ds(text) for text in completions_text]
        elif self.is_llama_format:
            extracted_answers = [extract_answer_llama(text) for text in completions_text]
        else:
            extracted_answers = [extract_answer(text) for text in completions_text]
        
        # Build output dict for raw completions
        output_raw = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "attention_mask": attention_mask,
            "prompts": prompts,  # Store prompts for later reward calculation
            "completions_text": completions_text,  # Store completions text for later reward calculation
            "completion_ids_list": completion_ids_list,  # Store completion_ids_list for later reward calculation
        }
        
        if old_per_token_logps is not None:
            output_raw["old_per_token_logps"] = old_per_token_logps
        if self.use_vllm and self.vllm_importance_sampling_correction:
            output_raw["importance_sampling_ratio"] = importance_sampling_ratio
        if ref_per_token_logps is not None:
            output_raw["ref_per_token_logps"] = ref_per_token_logps
        if "pixel_values" in prompt_inputs:
            output_raw["pixel_values"] = prompt_inputs["pixel_values"]
        if "image_grid_thw" in prompt_inputs:
            output_raw["image_grid_thw"] = prompt_inputs["image_grid_thw"]
        if "pixel_attention_mask" in prompt_inputs:
            output_raw["pixel_attention_mask"] = prompt_inputs["pixel_attention_mask"]
        if "image_sizes" in prompt_inputs:
            output_raw["image_sizes"] = prompt_inputs["image_sizes"]
        
        return extracted_answers, output_raw
                    
    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        # Check if two-stage generation is needed
        is_two_stage = "prompt_stage2" in inputs[0]
        output_stage1 = None
        output_raw = None
        
        if is_two_stage:
            # Raw stage: Generate with prompt_raw (only current question, no peers/history)
            if "prompt_raw" in inputs[0] and self.decouple_internal_belief:
                prompts_raw = [x["prompt_raw"] for x in inputs]
                extracted_answers, output_raw = self._generate_raw_completions(prompts_raw, inputs)
            else:
                extracted_answers = None
            
            # Stage 1: Generate with the first prompt
            prompts_stage1 = [x["prompt"] for x in inputs]
            completions_stage1, output_stage1 = self._generate_stage1_completions(prompts_stage1, inputs)
            
            # Replace {output_stage1} in prompt_stage2 with stage1 completions
            # and insert extracted answers if available
            prompts = []
            for i, inp in enumerate(inputs):
                prompt_stage2 = copy.deepcopy(inp["prompt_stage2"])
                completion_text = completions_stage1[i]
                # Use appropriate parsing function based on model type
                # Note: Check DS before Llama because deepseek-distill-llama contains both keywords
                if self.is_qwen3_format:
                    completion_text, matched = parse_summary_qwen3(completion_text)
                elif self.is_ds_format:
                    completion_text, matched = parse_summary_ds(completion_text)
                elif self.is_llama_format:
                    completion_text, matched = parse_summary_llama(completion_text)
                else:
                    completion_text, matched = parse_summary(completion_text)
                
                # Handle both string and conversational prompts
                if isinstance(prompt_stage2, str):
                    prompt_stage2 = prompt_stage2.replace("{output_stage1}", completion_text)
                    # Insert extracted answer if available
                    if extracted_answers is not None and extracted_answers[i]:
                        msg["content"] = msg["content"].replace("$decoupled_belief", extracted_answers[i])
                elif isinstance(prompt_stage2, list):
                    # For conversational format, replace in the last user message
                    for msg in prompt_stage2:
                        if msg["role"] == "user" and "{output_stage1}" in msg["content"]:
                            msg["content"] = msg["content"].replace("{output_stage1}", completion_text)
                            # Insert extracted answer if available
                            if extracted_answers is not None and extracted_answers[i]:
                                msg["content"] = msg["content"].replace("$decoupled_belief", extracted_answers[i])

                    # if self.accelerator.is_main_process:
                    #     for msg in prompt_stage2:
                    #         if msg["role"] == "user":
                    #             print(msg["content"])
                prompts.append(prompt_stage2)
        else:
            prompts = [x["prompt"] for x in inputs]

        # We don't yet support visual reward models/function, so we keep a copy of the original text-only prompts for
        # later use in the reward computation. If images are present, we insert {"type": "image"} as required by the
        # VLM chat template.
        original_prompts = copy.deepcopy(prompts)

        # If the prompts are conversational and the inputs contain images, we need to convert the prompts from
        # [{"role": "user", "content": "What color is the sky?"}] to
        # [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What color is the sky?"}]}]
        kwargs = {}
        has_images = "image" in inputs[0]
        if has_images:
            images = [example.get("image") for example in inputs]
            kwargs = {"images": [[img] for img in images]}
            for prompt in prompts:
                if isinstance(prompt, list):  # i.e., when using conversational data
                    prepare_multimodal_messages(prompt, num_images=1)
        
        prompts_text = [maybe_apply_chat_template({"prompt": p}, self.processing_class, enable_thinking=False)["prompt"] for p in prompts]

        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
            **kwargs,
        )
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            # If max_prompt_length is set, we trim the prompt to keep only the last `max_prompt_length` tokens.
            # Then we decode those tokens back into text. We manually remove leading pad tokens from the decoded text,
            # because we can't use `skip_special_tokens=True` (some special tokens are still needed for generation).
            protected = [self.image_token_id, self.vision_start_token_id, self.vision_end_token_id]
            protected = [token for token in protected if token is not None]
            prompt_ids, prompt_mask = truncate_with_protected_tokens(
                prompt_ids, prompt_mask, self.max_prompt_length, protected
            )

            prompts_text = self.processing_class.batch_decode(
                prompt_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            prompts_text = [re.sub(rf"^({re.escape(self.pad_token)})+", "", text) for text in prompts_text]

            # The chat template sometimes inserts a single image token into the prompt text. However, when this text is
            # later tokenized, the single image token string is expanded into multiple image token IDs, depending on the
            # image size. Since we're detokenizing here, we may see repeated image tokens in the decoded text. We
            # collapse them back into a single token string to match the original chat template in case it originally
            # applies it. Otherwise, it assumes that the chat template uses only vision_start_token_id to indicate images
            # (e.g. Gemma 3) and removes all image_token instances and vision_end_token_id as well, leaving only
            # the vision_start_token_id (e.g. <start_of_image>).
            if self.image_token is not None:
                escaped_img_token = re.escape(self.image_token)
                # Search for the image token in the chat template
                if re.search(escaped_img_token, self.processing_class.chat_template):
                    prompts_text = [
                        re.sub(rf"({escaped_img_token})+", self.image_token, text) for text in prompts_text
                    ]
                else:
                    # If the chat template doesn't use the image token, we remove all instances of it + vision_end_token_id
                    if self.vision_end_token_id is not None:
                        escaped_eoi_token = re.escape(
                            self.processing_class.tokenizer.decode([self.vision_end_token_id])
                        )
                        prompts_text = [
                            re.sub(rf"({escaped_img_token})+{escaped_eoi_token}", "", text) for text in prompts_text
                        ]
                    else:
                        # If vision_end_token_id is None, just remove the image tokens
                        prompts_text = [re.sub(rf"({escaped_img_token})+", "", text) for text in prompts_text]

        # Generate completions using either vLLM or regular generation
        if self.use_vllm:
            if self.vllm_mode == "colocate" and self.args.vllm_enable_sleep_mode:
                # wake up colocated vLLM instances if needed
                torch.cuda.empty_cache()  # required to avoid OOM in some cases
                self.llm.wake_up()

            # First, update the vLLM weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            if self.vllm_mode == "server":
                all_prompts_text = gather_object(prompts_text)
                if has_images:
                    all_images = gather_object(images)

                if self.accelerator.is_main_process:
                    # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                    # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                    # prompt individually.
                    ordered_set_of_prompts = all_prompts_text[:: self.num_generations]

                    if has_images:
                        ordered_set_of_images = all_images[:: self.num_generations]
                    else:
                        ordered_set_of_images = None

                    with profiling_context(self, "vLLM.generate"):
                        output = self.vllm_client.generate(
                            prompts=ordered_set_of_prompts,
                            images=ordered_set_of_images,
                            n=self.num_generations,
                            repetition_penalty=self.repetition_penalty,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            top_k=-1 if self.top_k is None else self.top_k,
                            min_p=0.0 if self.min_p is None else self.min_p,
                            max_tokens=self.max_completion_length,
                            guided_decoding_regex=self.guided_decoding_regex,
                            generation_kwargs=self.args.generation_kwargs,
                        )
                        payload = (output["completion_ids"], output["logprobs"])
                else:
                    payload = None

                # Broadcast the completions from the main process to all processes, ensuring each process receives its corresponding slice.
                obj_list = [payload]
                broadcast_object_list(obj_list, from_process=0)
                completion_ids, all_logprobs = obj_list[0]

                process_slice = slice(
                    self.accelerator.process_index * len(prompts),
                    (self.accelerator.process_index + 1) * len(prompts),
                )
                completion_ids = completion_ids[process_slice]
                all_logprobs = all_logprobs[process_slice]

            # Generate completions using colocated vLLM instances: each device holds vLLM copy and work on their own batch of prompts
            elif self.vllm_mode == "colocate":
                if self.guided_decoding_regex:
                    guided_decoding = GuidedDecodingParams(regex=self.guided_decoding_regex)
                else:
                    guided_decoding = None

                generation_kwargs = {
                    "n": 1,  # vLLM on each GPU generates only 1 in colocate mode
                    "repetition_penalty": self.repetition_penalty,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": -1 if self.top_k is None else self.top_k,
                    "min_p": 0.0 if self.min_p is None else self.min_p,
                    "max_tokens": self.max_completion_length,
                    "guided_decoding": guided_decoding,
                    "logprobs": 0,  # only return the logprob of the generated token
                }
                if self.args.generation_kwargs is not None:
                    generation_kwargs.update(self.args.generation_kwargs)
                sampling_params = SamplingParams(**generation_kwargs)

                if self.vllm_tensor_parallel_size > 1:
                    # Gather prompts from all ranks in the TP group and flatten.
                    # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
                    orig_size = len(prompts_text)
                    gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                    torch.distributed.all_gather_object(gathered_prompts, prompts_text, group=self.tp_group)
                    all_prompts_text = [p for sublist in gathered_prompts for p in sublist]

                    if has_images:
                        gathered_images = [None for _ in range(self.vllm_tensor_parallel_size)]
                        torch.distributed.all_gather_object(gathered_images, images, group=self.tp_group)
                        all_images = [img for sublist in gathered_images for img in sublist]
                    else:
                        all_images = None
                else:
                    all_prompts_text = prompts_text
                    all_images = images if has_images else None

                if has_images and all_images:
                    vllm_inputs = []
                    for prompt, image in zip(all_prompts_text, all_images):
                        if image is not None:
                            vllm_inputs.append({"prompt": prompt, "multi_modal_data": {"image": image}})
                        else:
                            vllm_inputs.append(prompt)
                else:
                    vllm_inputs = all_prompts_text

                with profiling_context(self, "vLLM.generate"):
                    all_outputs = self.llm.generate(vllm_inputs, sampling_params=sampling_params, use_tqdm=False)

                completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]
                all_logprobs = [
                    [next(iter(lp.values())).logprob for lp in output.logprobs]
                    for outputs in all_outputs
                    for output in outputs.outputs
                ]

                if self.vllm_tensor_parallel_size > 1:
                    # Slice completions for this rank within its TP group.
                    # Each rank generates all outputs — we keep only our share.
                    local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                    tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                    completion_ids = completion_ids[tp_slice]
                    all_logprobs = all_logprobs[tp_slice]

                if self.args.vllm_enable_sleep_mode:
                    self.llm.sleep(level=1)

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            sampling_per_token_logps = [
                torch.tensor(logprobs, device=device, dtype=torch.float32) for logprobs in all_logprobs
            ]
            sampling_per_token_logps = pad(sampling_per_token_logps, padding_value=0.0)

        elif self.use_transformers_paged:
            # Re-process inputs for paged generation if needed
            # Note: images are already validated and preprocessed above
            paged_prompt_inputs = self.processing_class(text=prompts_text, **kwargs)
            previous_attn = self.model_wrapped.config._attn_implementation

            if is_flash_attn_2_available():
                self.model_wrapped.config._attn_implementation = "paged_attention"
            else:
                self.model_wrapped.config._attn_implementation = "sdpa_paged"
            with (
                profiling_context(self, "transformers.generate_batch"),
                unwrap_model_for_generation(
                    self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
            ):
                # Cast to the appropriate dtype based on training configuration
                if self.args.bf16:
                    unwrapped_model.to(torch.bfloat16)
                elif self.args.fp16:
                    unwrapped_model.to(torch.float16)
                with torch.inference_mode():
                    all_outputs = unwrapped_model.generate_batch(
                        paged_prompt_inputs.input_ids, generation_config=self.generation_config, progress_bar=False
                    )
            completion_ids = [output.generated_tokens for output in all_outputs.values()]
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
            prompt_ids = [torch.tensor(ids, device=device) for ids in paged_prompt_inputs.input_ids]
            prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            # Restore the original attention implementation, training mode
            self.model_wrapped.config._attn_implementation = previous_attn
        else:
            # Regular generation path
            with (
                profiling_context(self, "transformers.generate"),
                unwrap_model_for_generation(
                    self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
            ):
                prompt_inputs["input_ids"], prompt_inputs["attention_mask"] = prompt_ids, prompt_mask
                prompt_completion_ids = unwrapped_model.generate(
                    **prompt_inputs, generation_config=self.generation_config, disable_compile=True
                )
            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Convert tensor to a list of lists of token IDs. This will be passed to the reward function, avoiding the need
        # to re-tokenize completions if the reward is computed from tokens.
        completion_ids_list = [row[mask_row].tolist() for row, mask_row in zip(completion_ids, completion_mask.bool())]

        # Sum along sequence dimension (dim=1) to get completion length per sequence, used for logging
        completion_lengths = completion_mask.sum(1)
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        num_items_in_batch = agg_completion_lengths.sum()  # this is required for the DAPO loss

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = completion_mask * (~truncated_completions).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        with torch.no_grad():
            # If the generation and optimization steps are misaligned—i.e., if generation does not occur at the end of
            # a full optimizer step (when gradient_accumulation_steps is not a multiple of generate_every)—then the
            # samples may come from an earlier version of the model. In that case, we need to track old_per_token_logps
            # for importance sampling. If the steps are aligned, importance sampling isn't necessary and we set
            # old_per_token_logps to None.
            # When using vLLM, we always compute old_per_token_logps for importance sampling, it was shown that the
            # distribution mismatch between vLLM and the training model can be large and harm the training.
            generate_every = self.args.steps_per_generation * self.num_iterations  # generation frequency
            if self.args.gradient_accumulation_steps % generate_every != 0 or (
                self.use_vllm and self.vllm_importance_sampling_correction
            ):
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    pixel_values=prompt_inputs.get("pixel_values"),
                    image_grid_thw=prompt_inputs.get("image_grid_thw"),
                    pixel_attention_mask=prompt_inputs.get("pixel_attention_mask"),
                    image_sizes=prompt_inputs.get("image_sizes"),
                )
            else:
                old_per_token_logps = None

            # Compute the importance sampling ratio when using vLLM, to correct for potential distribution mismatch
            if self.use_vllm and self.vllm_importance_sampling_correction:
                importance_sampling_ratio = torch.exp(old_per_token_logps - sampling_per_token_logps)
                importance_sampling_ratio = torch.clamp(
                    importance_sampling_ratio, max=self.vllm_importance_sampling_cap
                )

            # Compute the per-token log probabilities for the reference model
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        pixel_values=prompt_inputs.get("pixel_values"),
                        image_grid_thw=prompt_inputs.get("image_grid_thw"),
                        pixel_attention_mask=prompt_inputs.get("pixel_attention_mask"),
                        image_sizes=prompt_inputs.get("image_sizes"),
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                            pixel_values=prompt_inputs.get("pixel_values"),
                            image_grid_thw=prompt_inputs.get("image_grid_thw"),
                            pixel_attention_mask=prompt_inputs.get("pixel_attention_mask"),
                            image_sizes=prompt_inputs.get("image_sizes"),
                        )
            else:
                ref_per_token_logps = None

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        # Calculate rewards for each reward function. rewards_per_func aggregates rewards across all processes. This is
        # important because rewards will be normalized per group, and completions are distributed. We will later slice
        # rewards_per_func to extract each process's subset.
        rewards_per_func = self._calculate_rewards(inputs, original_prompts, completions, completion_ids_list)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        
        # Extract the last reward function's result for stage1 (if two-stage)
        last_reward_for_stage1 = None
        if is_two_stage:
            # Get the last reward function's output (before slicing)
            last_reward_for_stage1 = rewards_per_func[:, -1].clone()

        if self.scale_rewards in ["group", "none"]:
            # If self.scale_rewards = "none", we'll still log group level std
            std_rewards = rewards.view(-1, self.num_generations).std(dim=1)
            std_rewards = std_rewards.repeat_interleave(self.num_generations, dim=0)
        elif self.scale_rewards == "batch":
            # Compute global std
            std_rewards = rewards.std().expand_as(rewards)
        else:
            raise ValueError(
                f"Invalid value for scale_rewards: {self.scale_rewards}. Must be one of 'batch', 'group', or 'none'."
            )

        is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
        if self.scale_rewards != "none":
            advantages = advantages / (std_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()  # keep the aggregated advantages for logging
        advantages = advantages[process_slice]

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Log completion lengths, mean, min, max
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        # Identify sequences that terminated with EOS and log their lengths
        agg_terminated_with_eos = self.accelerator.gather(is_eos.any(dim=1))
        term_completion_lengths = agg_completion_lengths[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_lengths) / len(agg_completion_lengths)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        if len(term_completion_lengths) == 0:  # edge case where no terminated sequences are found
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_func_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_func_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        # Log prompt and completion texts
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())

        if has_images:
            self._logs["image"].extend(gather_object(images))

        if self.use_vllm and self.vllm_importance_sampling_correction:
            delta = torch.abs(old_per_token_logps - sampling_per_token_logps)
            delta = delta[completion_mask.bool()]
            mean_delta = torch.mean(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            max_delta = torch.max(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            self._metrics[mode]["sampling/sampling_logp_difference/mean"].append(
                self.accelerator.gather(mean_delta).mean().item()
            )
            self._metrics[mode]["sampling/sampling_logp_difference/max"].append(
                self.accelerator.gather(max_delta).max().item()
            )

            flat_is_ratio = importance_sampling_ratio[completion_mask.bool()]
            min_importance_sampling_ratio = (
                torch.min(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            mean_importance_sampling_ratio = (
                torch.mean(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            max_importance_sampling_ratio = (
                torch.max(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/min"].append(
                nanmin(self.accelerator.gather(min_importance_sampling_ratio)).item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/mean"].append(
                self.accelerator.gather(mean_importance_sampling_ratio).nanmean().item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/max"].append(
                nanmax(self.accelerator.gather(max_importance_sampling_ratio)).item()
            )

        output_stage2 = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
        }
        if old_per_token_logps is not None:
            output_stage2["old_per_token_logps"] = old_per_token_logps
        if self.use_vllm and self.vllm_importance_sampling_correction:
            output_stage2["importance_sampling_ratio"] = importance_sampling_ratio
        if ref_per_token_logps is not None:
            output_stage2["ref_per_token_logps"] = ref_per_token_logps
        if "pixel_values" in prompt_inputs:
            output_stage2["pixel_values"] = prompt_inputs["pixel_values"]
        if "image_grid_thw" in prompt_inputs:
            output_stage2["image_grid_thw"] = prompt_inputs["image_grid_thw"]
        if "pixel_attention_mask" in prompt_inputs:
            output_stage2["pixel_attention_mask"] = prompt_inputs["pixel_attention_mask"]
        if "image_sizes" in prompt_inputs:
            output_stage2["image_sizes"] = prompt_inputs["image_sizes"]
        
        # If two-stage generation, return both stage1 and stage2 outputs
        if is_two_stage and output_stage1 is not None:
            # Now calculate stage1 rewards after stage2 is complete
            prompts_stage1 = output_stage1["prompts"]
            completions_text_stage1 = output_stage1["completions_text"]
            completion_ids_list_stage1 = output_stage1["completion_ids_list"]
            
            # Prepare completions in the right format for reward calculation
            if is_conversational(inputs[0]):
                completions_stage1 = []
                for prompt, completion in zip(prompts_stage1, completions_text_stage1):
                    # Make a copy to avoid modifying the original
                    prompt_copy = copy.deepcopy(prompt)
                    bootstrap = prompt_copy.pop()["content"] if prompt_copy[-1]["role"] == "assistant" else ""
                    completions_stage1.append([{"role": "assistant", "content": bootstrap + completion}])
            else:
                completions_stage1 = completions_text_stage1
            
            # Calculate rewards for stage 1
            rewards_per_func_stage1_own = self._calculate_rewards_stage1(
                inputs, prompts_stage1, completions_stage1, completion_ids_list_stage1
            )
            
            # When using PRR, don't add stage2's last reward to stage1
            # Otherwise, extend stage1 rewards with stage2's last reward
            if self.use_prr:
                # Use only stage1's own rewards
                rewards_per_func_stage1 = rewards_per_func_stage1_own
                weights_to_use = self.reward_weights_stage1.to(device)
            else:
                # Extend stage1 rewards with stage2's last reward (original behavior for OR)
                # rewards_per_func_stage1_own: shape (total_samples, num_stage1_funcs)
                # last_reward_for_stage1: shape (total_samples,)
                last_reward_for_stage1_expanded = last_reward_for_stage1.unsqueeze(1)  # (total_samples, 1)
                
                # Concatenate to create full rewards_per_func including stage2's last reward
                # Shape: (total_samples, num_stage1_funcs + 1)
                rewards_per_func_stage1 = torch.cat([rewards_per_func_stage1_own, last_reward_for_stage1_expanded], dim=1)
                
                # Extend the weights: add 1.0 for the last reward
                weights_to_use = torch.cat([
                    self.reward_weights_stage1.to(device),
                    torch.tensor([1.0], dtype=torch.float32, device=device)
                ])
            
            # Compute the combined weighted sum of all rewards
            # This is the final reward for each rollout
            rewards_stage1 = (rewards_per_func_stage1 * weights_to_use.unsqueeze(0)).nansum(dim=1)
            
            # Compute grouped-wise rewards (same as stage2)
            mean_grouped_rewards_stage1 = rewards_stage1.view(-1, self.num_generations).mean(dim=1)
            
            # Normalize the rewards to compute the advantages
            mean_grouped_rewards_stage1 = mean_grouped_rewards_stage1.repeat_interleave(self.num_generations, dim=0)
            advantages_stage1 = rewards_stage1 - mean_grouped_rewards_stage1
            
            # Compute std for scaling (same logic as stage2)
            if self.scale_rewards in ["group", "none"]:
                std_rewards_stage1 = rewards_stage1.view(-1, self.num_generations).std(dim=1)
                std_rewards_stage1 = std_rewards_stage1.repeat_interleave(self.num_generations, dim=0)
            elif self.scale_rewards == "batch":
                std_rewards_stage1 = rewards_stage1.std().expand_as(rewards_stage1)
            else:
                raise ValueError(
                    f"Invalid value for scale_rewards: {self.scale_rewards}. Must be one of 'batch', 'group', or 'none'."
                )
            
            is_std_zero_stage1 = torch.isclose(std_rewards_stage1, torch.zeros_like(std_rewards_stage1))
            if self.scale_rewards != "none":
                advantages_stage1 = advantages_stage1 / (std_rewards_stage1 + 1e-4)
            
            # Slice to keep only the local part of the data for stage1
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            
            # Update output_stage1 with rewards and advantages
            # rewards_per_func: extended rewards including stage2_last_reward (local slice)
            # rewards: combined weighted sum (local slice)
            # advantages: computed from combined rewards
            output_stage1["rewards_per_func"] = rewards_per_func_stage1[process_slice]
            output_stage1["rewards"] = rewards_stage1[process_slice]
            output_stage1["advantages"] = advantages_stage1[process_slice]
            
            # Log stage1 completion statistics (same format as stage2)
            completion_mask_stage1 = output_stage1["completion_mask"]
            completion_ids_stage1 = output_stage1["completion_ids"]
            is_eos_stage1 = completion_ids_stage1 == self.eos_token_id
            
            # Sum along sequence dimension to get completion length per sequence
            completion_lengths_stage1 = completion_mask_stage1.sum(1)
            agg_completion_lengths_stage1 = self.accelerator.gather(completion_lengths_stage1)
            
            # Log completion lengths, mean, min, max
            self._metrics[mode]["completions_stage1/mean_length"].append(agg_completion_lengths_stage1.float().mean().item())
            self._metrics[mode]["completions_stage1/min_length"].append(agg_completion_lengths_stage1.float().min().item())
            self._metrics[mode]["completions_stage1/max_length"].append(agg_completion_lengths_stage1.float().max().item())
            
            # Identify sequences that terminated with EOS and log their lengths
            agg_terminated_with_eos_stage1 = self.accelerator.gather(is_eos_stage1.any(dim=1))
            term_completion_lengths_stage1 = agg_completion_lengths_stage1[agg_terminated_with_eos_stage1]
            clipped_completions_ratio_stage1 = 1 - len(term_completion_lengths_stage1) / len(agg_completion_lengths_stage1)
            self._metrics[mode]["completions_stage1/clipped_ratio"].append(clipped_completions_ratio_stage1)
            if len(term_completion_lengths_stage1) == 0:  # edge case where no terminated sequences are found
                term_completion_lengths_stage1 = torch.zeros(1, device=device)
            self._metrics[mode]["completions_stage1/mean_terminated_length"].append(term_completion_lengths_stage1.float().mean().item())
            self._metrics[mode]["completions_stage1/min_terminated_length"].append(term_completion_lengths_stage1.float().min().item())
            self._metrics[mode]["completions_stage1/max_terminated_length"].append(term_completion_lengths_stage1.float().max().item())
            
            # Log stage1 reward metrics (same format as stage2)
            # Log each reward function for stage1 (only own rewards, not including stage2's last reward)
            for i, reward_func_name in enumerate(self.reward_func_names_stage1):
                mean_rewards_s1 = torch.nanmean(rewards_per_func_stage1_own[:, i]).item()
                self._metrics[mode][f"rewards_stage1/{reward_func_name}/mean"].append(mean_rewards_s1)
                std_func_rewards_s1 = nanstd(rewards_per_func_stage1_own[:, i]).item()
                self._metrics[mode][f"rewards_stage1/{reward_func_name}/std"].append(std_func_rewards_s1)
            
            # Log the last reward from stage2 (used in stage1) only when not using PRR
            if not self.use_prr:
                self._metrics[mode][f"rewards_stage1/{self.reward_func_names[-1]}_from_stage2/mean"].append(
                    last_reward_for_stage1.mean().item()
                )
                self._metrics[mode][f"rewards_stage1/{self.reward_func_names[-1]}_from_stage2/std"].append(
                    nanstd(last_reward_for_stage1).item()
                )
            
            # Log combined reward statistics for stage1
            self._metrics[mode]["reward_stage1"].append(mean_grouped_rewards_stage1.mean().item())
            self._metrics[mode]["reward_stage1_std"].append(std_rewards_stage1.mean().item())
            self._metrics[mode]["frac_reward_zero_std_stage1"].append(is_std_zero_stage1.float().mean().item())
            
            # Calculate rewards for raw completions if available
            if output_raw is not None:
                prompts_raw = output_raw["prompts"]
                completions_text_raw = output_raw["completions_text"]
                completion_ids_list_raw = output_raw["completion_ids_list"]
                
                # Prepare completions in the right format for reward calculation
                if is_conversational(inputs[0]):
                    completions_raw = []
                    for prompt, completion in zip(prompts_raw, completions_text_raw):
                        # Make a copy to avoid modifying the original
                        prompt_copy = copy.deepcopy(prompt)
                        bootstrap = prompt_copy.pop()["content"] if prompt_copy[-1]["role"] == "assistant" else ""
                        completions_raw.append([{"role": "assistant", "content": bootstrap + completion}])
                else:
                    completions_raw = completions_text_raw
                
                # Calculate rewards for raw using the SAME reward functions as stage2
                # (format reward and accuracy reward)
                rewards_per_func_raw = self._calculate_rewards(inputs, prompts_raw, completions_raw, completion_ids_list_raw)
                
                # Apply weights to each reward function's output and sum
                rewards_raw = (rewards_per_func_raw * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
                
                # Compute grouped-wise rewards
                mean_grouped_rewards_raw = rewards_raw.view(-1, self.num_generations).mean(dim=1)
                
                # Normalize the rewards to compute the advantages
                mean_grouped_rewards_raw = mean_grouped_rewards_raw.repeat_interleave(self.num_generations, dim=0)
                advantages_raw = rewards_raw - mean_grouped_rewards_raw
                
                # Compute std for scaling
                if self.scale_rewards in ["group", "none"]:
                    std_rewards_raw = rewards_raw.view(-1, self.num_generations).std(dim=1)
                    std_rewards_raw = std_rewards_raw.repeat_interleave(self.num_generations, dim=0)
                elif self.scale_rewards == "batch":
                    std_rewards_raw = rewards_raw.std().expand_as(rewards_raw)
                else:
                    raise ValueError(
                        f"Invalid value for scale_rewards: {self.scale_rewards}. Must be one of 'batch', 'group', or 'none'."
                    )
                
                is_std_zero_raw = torch.isclose(std_rewards_raw, torch.zeros_like(std_rewards_raw))
                if self.scale_rewards != "none":
                    advantages_raw = advantages_raw / (std_rewards_raw + 1e-4)
                
                # Slice to keep only the local part of the data for raw stage
                process_slice = slice(
                    self.accelerator.process_index * len(prompts),
                    (self.accelerator.process_index + 1) * len(prompts),
                )
                
                # Update output_raw with rewards and advantages
                output_raw["rewards_per_func"] = rewards_per_func_raw[process_slice]
                output_raw["rewards"] = rewards_raw[process_slice]
                output_raw["advantages"] = advantages_raw[process_slice]
                
                # Log raw completion statistics
                completion_mask_raw = output_raw["completion_mask"]
                completion_ids_raw = output_raw["completion_ids"]
                is_eos_raw = completion_ids_raw == self.eos_token_id
                
                completion_lengths_raw = completion_mask_raw.sum(1)
                agg_completion_lengths_raw = self.accelerator.gather(completion_lengths_raw)
                
                self._metrics[mode]["completions_raw/mean_length"].append(agg_completion_lengths_raw.float().mean().item())
                self._metrics[mode]["completions_raw/min_length"].append(agg_completion_lengths_raw.float().min().item())
                self._metrics[mode]["completions_raw/max_length"].append(agg_completion_lengths_raw.float().max().item())
                
                agg_terminated_with_eos_raw = self.accelerator.gather(is_eos_raw.any(dim=1))
                term_completion_lengths_raw = agg_completion_lengths_raw[agg_terminated_with_eos_raw]
                clipped_completions_ratio_raw = 1 - len(term_completion_lengths_raw) / len(agg_completion_lengths_raw)
                self._metrics[mode]["completions_raw/clipped_ratio"].append(clipped_completions_ratio_raw)
                if len(term_completion_lengths_raw) == 0:
                    term_completion_lengths_raw = torch.zeros(1, device=device)
                self._metrics[mode]["completions_raw/mean_terminated_length"].append(term_completion_lengths_raw.float().mean().item())
                self._metrics[mode]["completions_raw/min_terminated_length"].append(term_completion_lengths_raw.float().min().item())
                self._metrics[mode]["completions_raw/max_terminated_length"].append(term_completion_lengths_raw.float().max().item())
                
                # Log raw reward metrics
                for i, reward_func_name in enumerate(self.reward_func_names):
                    mean_rewards_raw = torch.nanmean(rewards_per_func_raw[:, i]).item()
                    self._metrics[mode][f"rewards_raw/{reward_func_name}/mean"].append(mean_rewards_raw)
                    std_func_rewards_raw = nanstd(rewards_per_func_raw[:, i]).item()
                    self._metrics[mode][f"rewards_raw/{reward_func_name}/std"].append(std_func_rewards_raw)
                
                self._metrics[mode]["reward_raw"].append(mean_grouped_rewards_raw.mean().item())
                self._metrics[mode]["reward_raw_std"].append(std_rewards_raw.mean().item())
                self._metrics[mode]["frac_reward_zero_std_raw"].append(is_std_zero_raw.float().mean().item())
            
            # Return all three stages if raw is available
            result = {
                "stage1": output_stage1,
                "stage2": output_stage2,
            }
            if output_raw is not None:
                result["raw"] = output_raw
            
            return result
        else:
            # For single-stage, return stage2 output directly for backward compatibility
            return {"stage2": output_stage2}
        
    def _calculate_rewards_stage1(self, inputs, prompts, completions, completion_ids_list):
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs_stage1), device=device)

        # Repeat all input columns (but "prompt", "completion", and "completion_ids") to match the num of generations
        keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}

        # This allows for dynamic reward shaping based on training progress.
        reward_kwargs["trainer_state"] = self.state

        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs_stage1, self.reward_processing_classes_stage1, self.reward_func_names_stage1)
        ):
            with profiling_context(self, reward_func_name):
                if isinstance(reward_func, nn.Module):  # Module (no PretrainedModel) for compat with compiled models
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    reward_inputs = reward_processing_class(
                        text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
                else:
                    output_reward_func = reward_func(
                        prompts=prompts, completions=completions, completion_ids=completion_ids_list, **reward_kwargs
                    )
                    # Convert None values to NaN
                    output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]

                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {
                key: value[nan_row_idx] for key, value in reward_kwargs.items() if key != "trainer_state"
            }
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            logger.warning(
                f"All reward functions returned None for the following kwargs:\n{row_reward_kwargs}\n"
                "Please ensure that at least one reward function returns a valid reward."
            )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)
        return rewards_per_func
        
    def _compute_loss(self, model, inputs):
        # Compute the per-token log probabilities for the model
        total_loss = 0.0
        
        # Check if this is two-stage training
        is_two_stage = "stage1" in inputs and "stage2" in inputs
        has_raw = "raw" in inputs
        
        # Determine which stages to process
        stages_to_process = []
        if has_raw:
            stages_to_process.append("raw")
        if is_two_stage:
            stages_to_process.append("stage1")
        stages_to_process.append("stage2")  # Always process stage2
        
        for key in stages_to_process:
            # Skip stages if they don't have advantages (shouldn't happen, but safety check)
            if "advantages" not in inputs[key]:
                continue
            
            if key == "stage2" and self.ablation_stage2_training:
                continue
                
            prompt_ids, prompt_mask = inputs[key]["prompt_ids"], inputs[key]["prompt_mask"]
            completion_ids, completion_mask = inputs[key]["completion_ids"], inputs[key]["completion_mask"]
            input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

            # Compute the per_token_logps and the entropy at each position in the completion
            per_token_logps, entropies = self._get_per_token_logps_and_entropies(
                model,
                input_ids,
                attention_mask,
                logits_to_keep,
                compute_entropy=True,
                pixel_values=inputs[key].get("pixel_values"),
                image_grid_thw=inputs[key].get("image_grid_thw"),
                pixel_attention_mask=inputs[key].get("pixel_attention_mask"),
                image_sizes=inputs[key].get("image_sizes"),
            )

            if self.top_entropy_quantile < 1.0:
                entropy_mask = self.get_high_entropy_mask(entropies, completion_mask, 1 - self.top_entropy_quantile)
            else:
                entropy_mask = None

            # Compute the KL divergence between the model and the reference model
            if self.beta != 0.0:
                ref_per_token_logps = inputs[key]["ref_per_token_logps"]
                per_token_kl = (
                    torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
                )

            # Compute the loss
            advantages = inputs[key]["advantages"]
            # When num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps,
            # old_per_token_logps == per_token_logps. In this case we can skip its computation
            # (see _generate_and_score_completions) and instead use per_token_logps.detach().
            # The exception is when using vLLM, where we always compute old_per_token_logps
            # for importance sampling
            old_per_token_logps = inputs[key].get("old_per_token_logps")
            old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

            log_ratio = per_token_logps - old_per_token_logps
            if self.importance_sampling_level == "token":
                log_importance_weights = log_ratio
            elif self.importance_sampling_level == "sequence":
                log_importance_weights = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
                log_importance_weights = log_importance_weights.unsqueeze(-1)
            else:
                raise ValueError(
                    f"Unknown importance sampling level: {self.importance_sampling_level}. Possible values are 'token' "
                    "and 'sequence'."
                )
            # From here, log_importance_weights (and all subsequent tensors, coef_1, coef_2, etc.) shape depends on
            # importance_sampling_level: "token" level: (B, T); "sequence" level: (B, 1)

            coef_1 = torch.exp(log_importance_weights)
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

            # Two-sided clipping
            if self.args.delta is not None:
                coef_1 = torch.clamp(coef_1, max=self.args.delta)

            per_token_loss1 = coef_1 * advantages.unsqueeze(1)
            per_token_loss2 = coef_2 * advantages.unsqueeze(1)
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
            if entropy_mask is not None:
                per_token_loss = per_token_loss * entropy_mask

            if self.use_vllm and self.vllm_importance_sampling_correction:
                per_token_loss = per_token_loss * inputs[key]["importance_sampling_ratio"]

            if self.beta != 0.0:
                per_token_loss = per_token_loss + self.beta * per_token_kl

            if self.loss_type == "grpo":
                loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
                loss = loss / self.current_gradient_accumulation_steps
            elif self.loss_type == "bnpo":
                loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
                loss = loss / self.current_gradient_accumulation_steps
            elif self.loss_type == "dr_grpo":
                loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
                loss = loss / self.current_gradient_accumulation_steps
            elif self.loss_type == "dapo":
                normalizer = inputs[key]["num_items_in_batch"] / self.accelerator.num_processes
                loss = (per_token_loss * completion_mask).sum() / normalizer
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")

            total_loss += loss
            
            # Log the metrics
            mode = "train" if self.model.training else "eval"

            completion_token_count = completion_mask.sum().clamp(min=1.0)

            def masked_batch_mean(x):
                if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                    return x.mean()
                else:
                    return (x * completion_mask).sum() / completion_token_count

            if self.beta != 0.0:
                mean_kl = masked_batch_mean(per_token_kl)
                self._metrics[mode][f"kl_{key}"].append(self.accelerator.gather(mean_kl).nanmean().item())

            mean_entropy = masked_batch_mean(entropies)
            self._metrics[mode][f"entropy_{key}"].append(self.accelerator.gather(mean_entropy).nanmean().item())

            # Compute the clipped probability ratios
            is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
            is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
            is_region_clipped = is_low_clipped | is_high_clipped

            low_clip = masked_batch_mean(is_low_clipped.float())
            high_clip = masked_batch_mean(is_high_clipped.float())
            clip_ratio = masked_batch_mean(is_region_clipped.float())

            gathered_low_clip = self.accelerator.gather(low_clip)
            self._metrics[mode][f"clip_ratio_{key}/low_mean"].append(gathered_low_clip.nanmean().item())
            self._metrics[mode][f"clip_ratio_{key}/low_min"].append(nanmin(gathered_low_clip).item())
            gathered_high_clip = self.accelerator.gather(high_clip)
            self._metrics[mode][f"clip_ratio_{key}/high_mean"].append(gathered_high_clip.nanmean().item())
            self._metrics[mode][f"clip_ratio_{key}/high_max"].append(nanmax(gathered_high_clip).item())
            gathered_clip_ratio = self.accelerator.gather(clip_ratio)
            self._metrics[mode][f"clip_ratio_{key}/region_mean"].append(gathered_clip_ratio.nanmean().item())
        
        return total_loss

if __name__ == "__main__":
    test_input_llama = """
#### Reasoning
abc

def

#### Summary
xyz

www
"""
    print("Test Llama format parsing:")
    print(test_input_llama)
    parsed, matched = parse_summary_llama(test_input_llama)
    print("Parsed Summary:", parsed)
    print("Matched:", matched)
    print()
    
    # Test 1: Both think and answer tags present
    test_input_full = """
<think>
First, I need to analyze the problem.
Let me break it down step by step.
</think>

<answer>
(B)
</answer>
"""
    print("Test 1: Full format with both think and answer tags:")
    print(test_input_full)
    extracted = extract_answer(test_input_full)
    print("Extracted Result:")
    print(extracted)
    print()
    
    # Test 2: Only answer tag (backward compatibility)
    test_input_answer_only = """
Let me think about this problem.

The answer is <answer>(B)</answer>

I believe option B is correct.
"""
    print("Test 2: Only answer tag (backward compatibility):")
    print(test_input_answer_only)
    extracted = extract_answer(test_input_answer_only)
    print("Extracted Answer:", extracted)
    print()
    
    # Test 3: Only think tag
    test_input_think_only = """
<think>
Let me reason through this carefully.
Step 1: ...
Step 2: ...
</think>

Additional text outside tags.
"""
    print("Test 3: Only think tag:")
    print(test_input_think_only)
    extracted = extract_answer(test_input_think_only)
    print("Extracted Reasoning:", extracted)
    print()
    
    # Test 4: No tags
    test_input_no_tags = """
This is some text without any tags.
"""
    print("Test 4: No tags:")
    print(test_input_no_tags)
    extracted = extract_answer(test_input_no_tags)
    print("Extracted (empty):", repr(extracted))
    print()
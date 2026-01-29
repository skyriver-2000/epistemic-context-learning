import os
import json
import random
import argparse
import traceback

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from string import ascii_uppercase
from time import time
from tqdm import tqdm
from typing import Any, Dict, List, Optional

from datasets import Dataset, load_from_disk

from ECL.utils import set_seed
from ECL.utils.eval_utils import (EvalConfig, CONFIDENCE_PROTOCOLS, API_MODELS,
                                  RAW_PROMPT_TEMPLATE, RAW_PROMPT_TEMPLATE_LLAMA, RAW_PROMPT_TEMPLATE_QWEN3_STRONG_FORMAT, RAW_PROMPT_TEMPLATE_DS,
                                  build_raw_prompt, build_stage1_prompt, build_stage2_prompt,
                                  build_raw_prompt_llama, build_stage1_prompt_llama, build_stage2_prompt_llama,
                                  build_raw_prompt_qwen3, build_stage1_prompt_qwen3, build_stage2_prompt_qwen3,
                                  build_raw_prompt_ds, build_stage1_prompt_ds, build_stage2_prompt_ds,
                                  build_complete_prompt, build_complete_prompt_llama, build_complete_prompt_qwen3, build_complete_prompt_ds,
                                  extract_answer, extract_answer_llama, extract_answer_qwen3, extract_answer_ds,
                                  get_llm_client, generate_llm_chat, generate_llm_chat_two_stage,
                                  get_option_content)
from ECL.utils.logging_utils import setup_logger

# Set up logging
logger = setup_logger()
# Set seed for reproducibility
set_seed(42)

N_AGENTS = 6
MAX_WORKERS = int(os.environ.get('MAX_WORKERS_NUM', 32))  # Default max workers for thread pool


class EvalManager:
    def __init__(self, args: argparse.Namespace):
        self.ans_map = {i: letter for i, letter in enumerate(ascii_uppercase)}
        self.model_names = args.models  # Qwen/Qwen3-0.6B
        self.clients = {
            model: get_llm_client(model, ip=ip, port_number=port)
            for model, ip, port in zip(self.model_names, args.ips, args.port_numbers)
        }
        self.save_root = args.save_root
        self.dataset_path = args.dataset_path
        self.mode = args.mode
        self.tag = args.tag
        self.tag_peer = args.tag_peer
        self.tag_prompt = args.tag_prompt
        self.testing = args.testing
        self.temperature = args.temperature
        self.add_current_reason = args.current_reason
        self.add_history_reason = args.history_reason
        self.revert_test_identity = args.revert_test_identity
        self.decouple_belief_list = args.decouple_belief
        self.data_type = args.data_type
    
    def is_llama_model(self, model_name: str) -> bool:
        """Check if the model is a Llama model based on model name."""
        return "llama" in model_name.lower()
    
    def is_qwen3_model(self, model_name: str) -> bool:
        """Check if the model is a Qwen3 model based on model name."""
        return "qwen3" in model_name.lower()
    
    def is_large_model(self, model_name: str) -> bool:
        """Check if the model is a DS (DeepSeek) model based on model name."""
        return model_name in API_MODELS

    def save_results(self, config: EvalConfig, outputs: Dict, acc: int, failed_idx: set):
        results_path = os.path.join(self.save_root, "_".join(config.model.split('/')[1:]), config.fname)
        try:
            with open(results_path, 'w', encoding="utf-8") as f:
                json.dump({
                    'config': config.__dict__,
                    'outputs': outputs,
                    'correct_num': acc,
                    'failed_idx': list(failed_idx),
                }, f, indent=2)
        except IOError as e:
            logger.error(f"Failed to save results to {results_path}: {e}")

    def build_prompts(self, config: EvalConfig, eval_data: Dataset, raw_results: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
        # Note: Check DS before Llama because deepseek-distill-llama contains both keywords
        is_large = self.is_large_model(config.model)
        is_llama = self.is_llama_model(config.model)
        is_qwen3 = self.is_qwen3_model(config.model)
        
        if config.tag == 'RAW':
            if is_large:
                formatted_prompts = [{"single": RAW_PROMPT_TEMPLATE_DS.format(current=data)} for data in eval_data['formatted_question']]
            elif is_llama:
                formatted_prompts = [{"single": RAW_PROMPT_TEMPLATE_LLAMA.format(current=data)} for data in eval_data['formatted_question']]
            elif is_qwen3:
                formatted_prompts = [{"single": RAW_PROMPT_TEMPLATE_QWEN3_STRONG_FORMAT.format(current=data)} for data in eval_data['formatted_question']]
            else:
                formatted_prompts = [{"single": RAW_PROMPT_TEMPLATE.format(current=data)} for data in eval_data['formatted_question']]
        
        elif config.TAG_PROMPT == "AG":
            formatted_prompts = []
            for idx, sample in enumerate(eval_data):
                if is_large:
                    prompt = build_complete_prompt_ds(
                        sample, config.HR, config.CR, config.RTI,
                        config.TAG_PEER,
                        config.TAG_PROMPT.replace("NS",""),
                        data_type=self.data_type
                    )
                elif is_llama:
                    prompt = build_complete_prompt_llama(
                        sample, config.HR, config.CR, config.RTI,
                        config.TAG_PEER,
                        config.TAG_PROMPT.replace("NS",""),
                        data_type=self.data_type
                    )
                elif is_qwen3:
                    prompt = build_complete_prompt_qwen3(
                        sample, config.HR, config.CR, config.RTI,
                        config.TAG_PEER,
                        config.TAG_PROMPT.replace("NS",""),
                        data_type=self.data_type
                    )
                else:
                    prompt = build_complete_prompt(
                        sample, config.HR, config.CR, config.RTI,
                        config.TAG_PEER,
                        config.TAG_PROMPT.replace("NS",""),
                        data_type=self.data_type
                    )
                formatted_prompts.append(prompt)
         
        else:
            formatted_prompts = []
            
            for idx, sample in enumerate(eval_data):
                if is_large:
                    # Build stage 1 prompt for DS (summary generation)
                    stage1_prompt = build_stage1_prompt_ds(
                        sample, config.HR,
                        config.TAG_PEER,
                        config.TAG_PROMPT.replace("NS","")
                    )
                    
                    # Build stage 2 prompt template for DS (with placeholder for output_stage1 and optionally $decoupled_belief)
                    stage2_prompt = build_stage2_prompt_ds(
                        sample, config.HR, config.CR, config.RTI,
                        config.TAG_PEER,
                        decouple_belief=config.DB,
                        data_type=self.data_type
                    )
                elif is_llama:
                    # Build stage 1 prompt for Llama (summary generation)
                    stage1_prompt = build_stage1_prompt_llama(
                        sample, config.HR,
                        config.TAG_PEER,
                        config.TAG_PROMPT.replace("NS","")
                    )
                    
                    # Build stage 2 prompt template for Llama (with placeholder for output_stage1 and optionally $decoupled_belief)
                    stage2_prompt = build_stage2_prompt_llama(
                        sample, config.HR, config.CR, config.RTI,
                        config.TAG_PEER,
                        decouple_belief=config.DB,
                        data_type=self.data_type
                    )
                elif is_qwen3:
                    # Build stage 1 prompt for Qwen3 (summary generation)
                    stage1_prompt = build_stage1_prompt_qwen3(
                        sample, config.HR,
                        config.TAG_PEER,
                        config.TAG_PROMPT.replace("NS","")
                    )
                    
                    # Build stage 2 prompt template for Qwen3 (with placeholder for output_stage1 and optionally $decoupled_belief)
                    stage2_prompt = build_stage2_prompt_qwen3(
                        sample, config.HR, config.CR, config.RTI,
                        config.TAG_PEER,
                        decouple_belief=config.DB,
                        data_type=self.data_type
                    )
                else:
                    # Build stage 1 prompt (summary generation)
                    stage1_prompt = build_stage1_prompt(
                        sample, config.HR,
                        config.TAG_PEER,
                        config.TAG_PROMPT.replace("NS","")
                    )
                    
                    # Build stage 2 prompt template (with placeholder for output_stage1 and optionally $decoupled_belief)
                    stage2_prompt = build_stage2_prompt(
                        sample, config.HR, config.CR, config.RTI,
                        config.TAG_PEER,
                        decouple_belief=config.DB,
                        data_type=self.data_type
                    )

                formatted_prompts.append({
                    "stage_1": stage1_prompt,
                    "stage_2": stage2_prompt
                })
        
        return formatted_prompts

    
    def evaluate_single_config(self, config: EvalConfig, eval_data: Dataset, raw_results: Optional[Dict[str, Any]] = None, is_failed_example_loop: bool = False) -> Dict:
        try:        
            outputs = defaultdict(lambda: [None for _ in range(len(eval_data))])
            idx_list = range(len(eval_data))
            futures = {}  # Initialize futures dictionary
            
            # Determine which examples to go over
            if is_failed_example_loop:

                with open(f'{self.save_root}/{config.fname}','r') as f:
                    results = json.load(f)
                
                # Load up `outputs` with the results from the completed examples
                outputs.update(results['outputs'])

                idx_list = results['failed_idx'] 
                logger.info('Going over these examples:', idx_list)
            
            formatted_prompts = self.build_prompts(config, eval_data, raw_results)    
            failed_idx = set()
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                for idx in idx_list:
                    futures[executor.submit(self.process_single_example, idx, formatted_prompts, eval_data, config, failed_idx, raw_results)] = idx
                
                for cnt, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc=f"Processing {config.model}-{config.mode}-{config.tag}")):
                    idx = futures[future]
                    result_dict = future.result()
                    for key, val in result_dict.items():
                        outputs[key][idx] = val
                    
                    if cnt % 100 == 0 or cnt + 1 == len(idx_list):
                        logger.info(f'=== PROGRESS: {cnt + 1}/{len(idx_list)} ===')
                        
                        # Compute metrics
                        acc = sum([int(is_correct) for is_correct in outputs['is_correct'] if is_correct is not None])
                        logger.info(f'Acc (%): {acc / len(idx_list) * 100:.2f}')
                        logger.info(f'Num failed: {len(failed_idx)}')
                        
                        self.save_results(config, outputs, acc, failed_idx)
            
            # save final results again
            self.save_results(config, outputs, acc, failed_idx)
                
                            
        except KeyboardInterrupt:
            if 'futures' in locals():
                for t in futures:
                    t.cancel()
        except Exception as e:
            logger.error(traceback.format_exc())
            if 'futures' in locals():
                for t in futures:
                    t.cancel()

        return outputs
        
    
    def process_single_example(self, idx: int, formatted_prompts: List[Dict[str, str]], eval_data: Dataset, config: EvalConfig, failed_idx: set, raw_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a single example with both raw and protocol-based evaluation"""
        if self.is_qwen3_model(config.model):
            if "4b" or "8b" in config.model.lower():
                gen_kwargs = {"chat_template_kwargs": {"enable_thinking": False}, "max_tokens": 6144}
            else:
                gen_kwargs = {}
        gen_kwargs = {"reasoning": {"enabled": True, "effort": "high"}} if self.is_large_model(config.model) else gen_kwargs
        
        # Check if this is RAW mode or two-stage mode
        if config.tag == 'RAW':
            # Original single-stage generation for RAW mode
            responses = generate_llm_chat(
                self.clients[config.model],
                config.model,
                formatted_prompts[idx]["single"],
                {
                    "n": 8,
                    **gen_kwargs
                },
                max_retries=3
            )
            final_prompt = formatted_prompts[idx]["single"]
        elif config.TAG_PROMPT == "AG":
            # Single-stage generation for AG prompt mode
            responses = generate_llm_chat(
                self.clients[config.model],
                config.model,
                formatted_prompts[idx],
                {
                    "n": 8,
                    **gen_kwargs
                },
                max_retries=3
            )
            final_prompt = formatted_prompts[idx]
        else:
            # Two-stage generation for protocol mode with different system prompts
            
            # If decouple_belief is enabled, first generate raw response
            decoupled_belief = ""
            if config.DB:
                # Note: Check DS before Llama because deepseek-distill-llama contains both keywords
                is_large = self.is_large_model(config.model)
                is_llama = self.is_llama_model(config.model)
                is_qwen3 = self.is_qwen3_model(config.model)
                try:
                    # Build raw prompt
                    if is_large:
                        raw_prompt = build_raw_prompt_ds(eval_data[idx])
                    elif is_llama:
                        raw_prompt = build_raw_prompt_llama(eval_data[idx])
                    elif is_qwen3:
                        raw_prompt = build_raw_prompt_qwen3(eval_data[idx])
                    else:
                        raw_prompt = build_raw_prompt(eval_data[idx])
                    
                    # Generate one raw completion (n=1)
                    raw_responses = generate_llm_chat(
                        self.clients[config.model],
                        config.model,
                        raw_prompt,
                        {
                            "n": 1,
                            **gen_kwargs
                        },
                        max_retries=1
                    )
                    
                    # Extract belief from raw response
                    raw_text = raw_responses[0] if isinstance(raw_responses, list) and raw_responses else ""
                    extract_func = extract_answer_ds if is_large else (extract_answer_llama if is_llama else (extract_answer_qwen3 if is_qwen3 else extract_answer))
                    decoupled_belief = extract_func(raw_text)
                except Exception as e:
                    logger.warning(f"Failed to generate or extract raw belief for idx {idx}: {e}")
                    decoupled_belief = ""
            
            # Prepare stage2 prompt with decoupled_belief placeholder replaced
            stage2_prompt_with_belief = formatted_prompts[idx]["stage_2"]
            if config.DB and "$decoupled_belief" in stage2_prompt_with_belief:
                stage2_prompt_with_belief = stage2_prompt_with_belief.replace("$decoupled_belief", decoupled_belief)
            
            response_stage1, responses = generate_llm_chat_two_stage(
                self.clients[config.model],
                config.model,
                formatted_prompts[idx]["stage_1"],
                stage2_prompt_with_belief,
                gen_kwargs_stage1={
                    "n": 1,
                    **gen_kwargs
                },
                gen_kwargs_stage2={
                    "n": 8,
                    **gen_kwargs
                },
                max_retries=3
            )
            
            # Extract the summary from stage 1 response
            # Note: Check DS before Llama because deepseek-distill-llama contains both keywords
            is_large = self.is_large_model(config.model)
            is_llama = self.is_llama_model(config.model)
            is_qwen3 = self.is_qwen3_model(config.model)
            try:
                if is_large:
                    # For DS format, extract summary after </think>
                    if '</think>' in response_stage1:
                        parts = response_stage1.split('</think>')
                        output_stage1 = parts[-1].strip()
                    else:
                        output_stage1 = response_stage1
                elif is_llama:
                    # For Llama format, extract summary after #### Summary
                    if '#### Summary' in response_stage1:
                        parts = response_stage1.split('#### Summary')
                        output_stage1 = parts[-1].strip()
                    else:
                        output_stage1 = response_stage1
                elif is_qwen3:
                    if '[/think]' in response_stage1:
                        parts = response_stage1.split('[/think]')
                        output_stage1 = parts[-1].strip()
                    else:
                        output_stage1 = response_stage1
                else:
                    # For normal format, extract summary between <summary> tags
                    summary_start = response_stage1.find("<summary>")
                    summary_end = response_stage1.find("</summary>")
                    if summary_start != -1 and summary_end != -1:
                        output_stage1 = response_stage1[summary_start + 9:summary_end].strip()
                    else:
                        output_stage1 = response_stage1
            except Exception as e:
                logger.warning(f"Failed to extract summary from stage 1 response for idx {idx}: {e}")
                output_stage1 = response_stage1
            
            # Store the final prompt that was used (with summary and belief filled in)
            final_prompt = stage2_prompt_with_belief.replace("{output_stage1}", output_stage1)
        
        
        # Use appropriate extract function based on model type
        # Note: Check DS before Llama because deepseek-distill-llama contains both keywords
        is_large = self.is_large_model(config.model)
        is_llama = self.is_llama_model(config.model)
        is_qwen3 = self.is_qwen3_model(config.model)
        extract_func = extract_answer_ds if is_large else (extract_answer_llama if is_llama else (extract_answer_qwen3 if is_qwen3 else extract_answer))
        preds = [extract_func(response) for response in responses]
        
        # find uncertainty
        num_options = len(eval_data[idx]['wrong_options']) + 1
        pred_counts = {self.ans_map[i]: 0 for i in range(num_options)}
        for pred in preds:
            if pred in pred_counts:
                pred_counts[pred] += 1
            else:
                # Catch failures
                logger.warning(f"Invalid {idx}th prediction ##{pred}## for options {pred_counts.keys()}")
        
        majority_pred = max(pred_counts.items(), key=lambda x: x[1])[0]
        majority_pred_content = get_option_content(majority_pred, eval_data[idx]['wrong_options'], eval_data[idx]['gt_option'])
        is_correct = int(majority_pred_content == eval_data[idx]['gt_option'])
        
        # edge case: all predictions are wrongly formatted
        if all(count == 0 for count in pred_counts.values()):
            failed_idx.add(idx)
            response = responses[0]
            majority_pred_content = "Error"
            is_correct = 0
        else:
            response = responses[preds.index(majority_pred)]
        
        return {
            'model': config.model,
            'init_input': formatted_prompts[idx]["stage_1"] if config.tag != 'RAW' and config.TAG_PROMPT != "AG" else "",
            'init_response': response_stage1 if config.tag != 'RAW' and config.TAG_PROMPT != "AG" else "",
            'input': final_prompt,
            'response': response,
            'pred_counts': pred_counts,
            'y_pred': majority_pred_content,
            'y_true': eval_data[idx]['gt_option'],
            'is_correct': is_correct,
        }
    
    
    def load_eval_data(self, dataset_path: str) -> Dataset:
        dataset = load_from_disk(dataset_path, keep_in_memory=True)

        # If in testing mode, keep only a small subset (first 5 samples)
        if self.testing:
            try:
                dataset = dataset.select(random.sample(range(len(dataset)), 100))
            except Exception:
                # Fallback to basic slicing for iterable/other dataset types
                dataset = dataset[:100]

        return dataset
    
    
    def estimate_raw_results(self, eval_data: Dataset) -> List[Dict]:
        # estimate raw results
        raw_configs: List[EvalConfig] = [
            EvalConfig(
                model=model,
                mode=self.mode,
                tag="RAW"
            )
            for model in self.model_names
        ]
        
        all_raw_results = []
        for raw_cfg in raw_configs:
            cache_path = os.path.join(self.save_root, "_".join(raw_cfg.model.split('/')[1:]), raw_cfg.fname)
            
            if os.path.exists(cache_path):
                logger.info(f"Skipping {raw_cfg.model} because it already exists")
                try:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        cached_ouputs = json.load(f)['outputs']
                        all_raw_results.append({
                            'model': cached_ouputs['model'],
                            'y_pred': cached_ouputs['y_pred'],
                        })
                    
                    logger.info(f"Loaded cached raw results for {raw_cfg.model} "
                                f"from {cache_path}")
                except Exception as e:
                    logger.warning(f"Failed to load cache {cache_path}: {e}. "
                                   f"Re-computing raw results…")
                continue

            try:
                logger.info(f"Estimating raw accuracy for model "
                            f"{raw_cfg.model} …")
                cached_ouputs = self.evaluate_single_config(raw_cfg, eval_data, is_failed_example_loop=False)
                all_raw_results.append({
                    'model': cached_ouputs['model'],
                    'y_pred': cached_ouputs['y_pred'],
                })
                logger.info(f"Cached raw results for {raw_cfg.model} at "
                            f"{cache_path}")
            except Exception as e:
                logger.error(f"Raw estimation failed for model "
                             f"{raw_cfg.model}: {e}")
                logger.error(traceback.format_exc())
                
        return all_raw_results


    def run(self, configs_to_resolve: List[str] = None) -> None:
        """
        1. load eval data
        2. format examples and estimate raw results
        3. setup eval configs
        4. format examples and run eval
        5. run reflection
        6. save results
        """
        
        start_time = time()
        
        # load eval data
        eval_data = self.load_eval_data(self.dataset_path)
        
        # estimate raw results
        all_raw_results = self.estimate_raw_results(eval_data)
        
        # create evaluation configs
        eval_configs: List[EvalConfig] = []
        if configs_to_resolve:
            is_failed_example_loop = True
            logger.warning('CONFIGS TO RESOLVE FOR FAILED CASES')
            for con in configs_to_resolve:
                newcon = EvalConfig()
                with open(con,'r') as f:
                    newcon.__dict__ = json.load(f)["config"]
                eval_configs.append(newcon)
        else:
            is_failed_example_loop = False
            for model, current_reason, history_reason, revert_test_identity, tag_peer, tag_prompt, decouple_belief, raw_results in \
                zip(self.model_names, self.add_current_reason, self.add_history_reason, self.revert_test_identity, self.tag_peer, self.tag_prompt, self.decouple_belief_list, all_raw_results):
                if model == raw_results['model'][0]:
                    eval_configs.append(
                        EvalConfig(model=model, mode=self.mode, tag=self.tag,
                                   current_reason=current_reason,
                                   history_reason=history_reason,
                                   revert_test_identity=revert_test_identity,
                                   tag_peer=tag_peer,
                                   tag_prompt=tag_prompt,
                                   decouple_belief=decouple_belief)
                    )
                else:
                    raise ValueError(f"Model {model} not found in all_raw_results")
             
        # run evaluation
        for config, raw_results in zip(eval_configs, all_raw_results):
            
            logger.info('\n\n\nNew config')
            logger.info(config.__dict__)

            cache_path = os.path.join(self.save_root, "_".join(config.model.split('/')[1:]), config.fname)
            if os.path.exists(cache_path):
                logger.info(f"Skipping {config.model} because it already exists")
                try:
                    with open(cache_path, 'r', encoding="utf-8") as f:
                        outputs = json.load(f)['outputs']
                    logger.info(f"Loaded cached protocol outputs for {config.model} from {cache_path}")
                except Exception as e:
                    logger.error(f"Failed to load cached protocol outputs for {config.model}: {e}")
                    logger.error(traceback.format_exc())
            else:
                try:
                    logger.info(f"Starting evaluation for {config.model} in {config.mode} mode...")
                    outputs = self.evaluate_single_config(config, eval_data, raw_results, is_failed_example_loop)
                    logger.info(f"Eval on {config.model} completed in {round(time() - start_time)} seconds")
                except Exception as e:
                    logger.info(f"Error processing {config.model}: {str(e)}")
                    logger.error(traceback.format_exc())
                
        logger.info(f"Evaluation completed in {round(time() - start_time)} seconds")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str, required=True, nargs='+', help='List of Model names, e.g. gpt-3.5-turbo gpt-4o')
    parser.add_argument('--ips', type=str, required=True, nargs='+', help='List of Model ips, e.g. 127.0.0.1 127.0.0.2')
    parser.add_argument('--port_numbers', type=int, required=True, nargs='+', help='List of Model port numbers, e.g. 8080 8081')
    parser.add_argument('--temperature', default=0.7, type=float, help='Sampling temperature for LLM generation')
    parser.add_argument('--save_root', type=str, required=True, help='Path to save results')
    parser.add_argument('--dataset_path', default='data/final_test', type=str, help='dataset name, e.g. bbh other')
    parser.add_argument('--mode', default=['normal'], type=str, choices=['normal', 'empowered', 'reflection'], help='Mode(s) of LLM: normal, empowered, reflection')
    parser.add_argument('--testing', action='store_true', help='Run on small subset of data for testing')
    parser.add_argument('--tag', type=str, default='first', help='Tag for the evaluation')
    parser.add_argument('--tag_peer', type=str, required=True, nargs='+', help='List of numbers of peers and reliable ones, e.g. 6_3')
    parser.add_argument('--tag_prompt', type=str, required=True, nargs='+', help='List of prompt tags, e.g. JP')
    parser.add_argument('--current_reason', type=int, required=True, nargs='+', help='List of whether to add current round reasoning')
    parser.add_argument('--history_reason', type=int, required=True, nargs='+', help='List of whether to add history peer reasoning')
    parser.add_argument('--revert_test_identity', type=int, required=True, nargs='+', help='List of whether to revert the peer identity at test time')
    parser.add_argument('--decouple_belief', type=int, required=True, nargs='+', help='List of whether to run raw prompt first and insert extracted belief into stage2 prompt (0 or 1)')
    parser.add_argument('--data_type', type=str, default='adv', choices=['adv', 'nat'], help='Type of data to evaluate on: adv (adversarial) or nat (natural)')
    args = parser.parse_args()
    
    # check if there is any model named the same
    # if len(set(args.models)) != len(args.models):
    #     raise ValueError('There are models with the same name')
    
    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root, exist_ok=True)
    
    # use this to retry examples that previously failed
    # List paths to the json files for the results you want to retry
    configs_to_resolve = []
    
    manager = EvalManager(args)
    manager.run(configs_to_resolve)
    
    logger.info("Execution completed.")

if __name__ == '__main__':
    main()
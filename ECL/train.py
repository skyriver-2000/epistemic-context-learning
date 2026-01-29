import random
from accelerate.state import PartialState
from transformers.trainer_utils import get_last_checkpoint, set_seed
from trl import GRPOConfig, ModelConfig, get_peft_config
from trl.trainer.utils import empty_cache

from ECL.trainer.grpo_trainer_two_stage import GRPOTrainerTwoStage
from ECL.rewards import (
    summary_format_reward_normal, format_reward_normal, accuracy_reward_normal, 
    get_peer_recognition_reward_func_stage1,
    summary_format_reward_llama, format_reward_llama, accuracy_reward_llama,
    get_peer_recognition_reward_func_stage1_llama,
    summary_format_reward_qwen3, format_reward_qwen3, accuracy_reward_qwen3,
    get_peer_recognition_reward_func_stage1_qwen3,
    summary_format_reward_ds, format_reward_ds, accuracy_reward_ds,
    get_peer_recognition_reward_func_stage1_ds
)
from ECL.utils.arguments import H4ArgumentParser, ScriptArguments
from ECL.utils.logging_utils import setup_logger
from ECL.utils.train_utils import get_datasets, load_model_and_tokenizer
from ECL.utils.eval_utils import (
    SYSTEM_PROMPT_STAGE1,
    SYSTEM_PROMPT_STAGE2,
    SYSTEM_PROMPT_NORMAL,
    SYSTEM_PROMPT_STAGE1_LLAMA,
    SYSTEM_PROMPT_STAGE2_LLAMA,
    SYSTEM_PROMPT_NORMAL_LLAMA,
    SYSTEM_PROMPT_STAGE1_QWEN3,
    SYSTEM_PROMPT_STAGE2_QWEN3,
    SYSTEM_PROMPT_NORMAL_QWEN3,
    SYSTEM_PROMPT_STAGE1_DS,
    SYSTEM_PROMPT_STAGE2_DS,
    SYSTEM_PROMPT_NORMAL_DS,
    build_raw_prompt,
    build_ag_prompt,
    build_stage1_prompt,
    build_stage2_prompt,
    build_raw_prompt_llama,
    build_ag_prompt_llama,
    build_stage1_prompt_llama,
    build_stage2_prompt_llama,
    build_raw_prompt_qwen3,
    build_ag_prompt_qwen3,
    build_stage1_prompt_qwen3,
    build_stage2_prompt_qwen3,
    build_raw_prompt_ds,
    build_ag_prompt_ds,
    build_stage1_prompt_ds,
    build_stage2_prompt_ds
)

REWARD_FUNCS = []

SYSTEM_PROMPT_MAP = {
    "SYSTEM_PROMPT_NORMAL": {
        "default": SYSTEM_PROMPT_NORMAL,
        "stage1": SYSTEM_PROMPT_STAGE1,
        "stage2": SYSTEM_PROMPT_STAGE2
    }
}

SYSTEM_PROMPT_MAP_LLAMA = {
    "SYSTEM_PROMPT_NORMAL": {
        "default": SYSTEM_PROMPT_NORMAL_LLAMA,
        "stage1": SYSTEM_PROMPT_STAGE1_LLAMA,
        "stage2": SYSTEM_PROMPT_STAGE2_LLAMA
    }
}

SYSTEM_PROMPT_MAP_QWEN3 = {
    "SYSTEM_PROMPT_NORMAL": {
        "default": SYSTEM_PROMPT_NORMAL_QWEN3,
        "stage1": SYSTEM_PROMPT_STAGE1_QWEN3,
        "stage2": SYSTEM_PROMPT_STAGE2_QWEN3
    }
}

SYSTEM_PROMPT_MAP_DS = {
    "SYSTEM_PROMPT_NORMAL": {
        "default": SYSTEM_PROMPT_NORMAL_DS,
        "stage1": SYSTEM_PROMPT_STAGE1_DS,
        "stage2": SYSTEM_PROMPT_STAGE2_DS
    }
}

def main():
    parser = H4ArgumentParser((ScriptArguments, ModelConfig, GRPOConfig))
    script_args, model_args, training_args = parser.parse()

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logger = setup_logger(training_args, script_args)
    
    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.bf16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {script_args}")
    logger.info(f"Training/evaluation parameters {training_args}")
    
    ########################################
    # Model & Tokenizer & Reward functions
    ########################################
    logger.info("*** Loading pretrained model and tokenizer ***")
    
    model, model_kwargs, tokenizer = load_model_and_tokenizer(script_args, model_args, training_args)

    # check if it is debating grpo or normal grpo
    # also check if it is using Llama-specific format (#### separator) or Qwen3-specific format (no tags)
    # Note: Check deepseek before llama because deepseek-distill-llama contains both keywords
    global REWARD_FUNCS
    is_ds_format = "deepseek" in model_args.model_name_or_path.lower()
    is_llama_format = "llama" in model_args.model_name_or_path.lower()
    is_qwen3_format = "qwen3" in model_args.model_name_or_path.lower()
    
    decouple_internal_belief = "-DB" in training_args.output_dir
    data_type = "nat" if "_natural" in training_args.output_dir else "adv"
    print("Data Type:", data_type)
    
    if "-OR" in training_args.output_dir:
        if is_qwen3_format:
            REWARD_FUNCS_STAGE1 = [summary_format_reward_qwen3]
            REWARD_FUNCS = [format_reward_qwen3, accuracy_reward_qwen3]
        elif is_ds_format:
            REWARD_FUNCS_STAGE1 = [summary_format_reward_ds]
            REWARD_FUNCS = [format_reward_ds, accuracy_reward_ds]
        elif is_llama_format:
            REWARD_FUNCS_STAGE1 = [summary_format_reward_llama]
            REWARD_FUNCS = [format_reward_llama, accuracy_reward_llama]
        else:
            REWARD_FUNCS_STAGE1 = [summary_format_reward_normal]
            REWARD_FUNCS = [format_reward_normal, accuracy_reward_normal]
        training_args.reward_weights_stage1 = [1.0]
        training_args.reward_weights = [1.0, 1.0]
    elif "-PRR" in training_args.output_dir:
        if is_qwen3_format:
            peer_recognition_reward_qwen3 = get_peer_recognition_reward_func_stage1_qwen3(
                script_args.add_history_reasoning,
                script_args.tag_peer
            )
            REWARD_FUNCS_STAGE1 = [summary_format_reward_qwen3, peer_recognition_reward_qwen3]
            REWARD_FUNCS = [format_reward_qwen3, accuracy_reward_qwen3]
        elif is_ds_format:
            peer_recognition_reward_ds = get_peer_recognition_reward_func_stage1_ds(
                script_args.add_history_reasoning,
                script_args.tag_peer
            )
            REWARD_FUNCS_STAGE1 = [summary_format_reward_ds, peer_recognition_reward_ds]
            REWARD_FUNCS = [format_reward_ds, accuracy_reward_ds]
        elif is_llama_format:
            peer_recognition_reward_llama = get_peer_recognition_reward_func_stage1_llama(
                script_args.add_history_reasoning,
                script_args.tag_peer
            )
            REWARD_FUNCS_STAGE1 = [summary_format_reward_llama, peer_recognition_reward_llama]
            REWARD_FUNCS = [format_reward_llama, accuracy_reward_llama]
        else:
            peer_recognition_reward_normal = get_peer_recognition_reward_func_stage1(
                script_args.add_history_reasoning,
                script_args.tag_peer
            )
            REWARD_FUNCS_STAGE1 = [summary_format_reward_normal, peer_recognition_reward_normal]
            REWARD_FUNCS = [format_reward_normal, accuracy_reward_normal]
        training_args.reward_weights_stage1 = [1.0, 1.0]
        training_args.reward_weights = [1.0, 1.0]
    else:
        raise ValueError(f"Invalid output directory with reward not defined: {training_args.output_dir}")
    
    ################
    # Dataset
    ################
    logger.info("*** Loading datasets ***") 
   
    raw_datasets = get_datasets(
        script_args,
        splits=script_args.dataset_splits,
        configs=script_args.dataset_configs,
        columns_to_keep=None,
    )
    
    # split the dataset into train and test if requires evaluation
    if training_args.do_eval and "test" not in raw_datasets:
        raw_datasets = raw_datasets.train_test_split(test_size=0.1)
    
    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )

    if training_args.debug:
        for key in raw_datasets:
            raw_datasets[key] = raw_datasets[key].select(range(30))
    
    # Handle system prompt if needed
    with PartialState().main_process_first():
        if script_args.system_prompt in SYSTEM_PROMPT_MAP:
            logger.info(f"Using system prompt: {script_args.system_prompt}")
            # Automatically detect if model is Llama family, Qwen3 family, or DS family based on model name
            # Note: Check DS before Llama because deepseek-distill-llama contains both keywords
            is_qwen3 = is_qwen3_format  # Use the same detection as reward selection
            is_ds = is_ds_format  # Use the same detection as reward selection
            is_llama = is_llama_format  # Use the same detection as reward selection
            # Select appropriate prompt map based on model type
            if is_qwen3:
                prompt_map = SYSTEM_PROMPT_MAP_QWEN3
            elif is_ds:
                prompt_map = SYSTEM_PROMPT_MAP_DS
            elif is_llama:
                prompt_map = SYSTEM_PROMPT_MAP_LLAMA
            else:
                prompt_map = SYSTEM_PROMPT_MAP
                
            if script_args.tag_prompt == "AG":
                if is_qwen3:
                    raw_datasets = raw_datasets.map(
                        lambda x: {"prompt": [
                            {"role": "system", "content": prompt_map[script_args.system_prompt]["default"]},
                            {"role": "user", "content": build_ag_prompt_qwen3(x, script_args.add_history_reasoning, script_args.add_current_reasoning, revert_test_identity=0, peer_tag=script_args.tag_peer, decouple_belief=decouple_internal_belief, data_type=data_type)}
                        ]}, num_proc=script_args.preprocessing_num_workers, load_from_cache_file=False, keep_in_memory=True)
                elif is_ds:
                    raw_datasets = raw_datasets.map(
                        lambda x: {"prompt": [
                            {"role": "system", "content": prompt_map[script_args.system_prompt]["default"]},
                            {"role": "user", "content": build_ag_prompt_ds(x, script_args.add_history_reasoning, script_args.add_current_reasoning, revert_test_identity=0, peer_tag=script_args.tag_peer, decouple_belief=decouple_internal_belief, data_type=data_type)}
                        ]}, num_proc=script_args.preprocessing_num_workers, load_from_cache_file=False, keep_in_memory=True)
                elif is_llama:
                    raw_datasets = raw_datasets.map(
                        lambda x: {"prompt": [
                            {"role": "system", "content": prompt_map[script_args.system_prompt]["default"]},
                            {"role": "user", "content": build_ag_prompt_llama(x, script_args.add_history_reasoning, script_args.add_current_reasoning, revert_test_identity=0, peer_tag=script_args.tag_peer, decouple_belief=decouple_internal_belief, data_type=data_type)}
                        ]}, num_proc=script_args.preprocessing_num_workers, load_from_cache_file=False, keep_in_memory=True)
                else:
                    raw_datasets = raw_datasets.map(
                        lambda x: {"prompt": [
                            {"role": "system", "content": prompt_map[script_args.system_prompt]["default"]},
                            {"role": "user", "content": build_ag_prompt(x, script_args.add_history_reasoning, script_args.add_current_reasoning, revert_test_identity=0, peer_tag=script_args.tag_peer, decouple_belief=decouple_internal_belief, data_type=data_type)}
                        ]}, num_proc=script_args.preprocessing_num_workers, load_from_cache_file=False, keep_in_memory=True)
            elif script_args.add_peers:
                if is_qwen3:
                    # Use Qwen3-specific prompt builders
                    raw_datasets = raw_datasets.map(
                        lambda x: {
                            "prompt": [
                                {"role": "system", "content": prompt_map[script_args.system_prompt]["stage1"]},
                                {"role": "user", "content": build_stage1_prompt_qwen3(
                                    x,
                                    script_args.add_history_reasoning,
                                    peer_tag=script_args.tag_peer,
                                    prompt_tag=script_args.tag_prompt
                                )}
                            ],
                            "prompt_stage2": [
                                {"role": "system", "content": prompt_map[script_args.system_prompt]["stage2"]},
                                {"role": "user", "content": build_stage2_prompt_qwen3(
                                    x,
                                    script_args.add_history_reasoning,
                                    script_args.add_current_reasoning,
                                    revert_test_identity=0,
                                    peer_tag=script_args.tag_peer,
                                    decouple_belief=decouple_internal_belief,
                                    data_type=data_type
                                )}
                            ],
                            "prompt_raw": [
                                {"role": "system", "content": prompt_map[script_args.system_prompt]["default"]},
                                {"role": "user", "content": build_raw_prompt_qwen3(x)}
                            ]
                        }, num_proc=script_args.preprocessing_num_workers, load_from_cache_file=False, keep_in_memory=True)
                    # print(raw_datasets['train'][0]['prompt'][1]['content'])
                    # exit(-1)
                elif is_ds:
                    # Use DS-specific prompt builders
                    raw_datasets = raw_datasets.map(
                        lambda x: {
                            "prompt": [
                                {"role": "system", "content": prompt_map[script_args.system_prompt]["stage1"]},
                                {"role": "user", "content": build_stage1_prompt_ds(
                                    x,
                                    script_args.add_history_reasoning,
                                    peer_tag=script_args.tag_peer,
                                    prompt_tag=script_args.tag_prompt
                                )}
                            ],
                            "prompt_stage2": [
                                {"role": "system", "content": prompt_map[script_args.system_prompt]["stage2"]},
                                {"role": "user", "content": build_stage2_prompt_ds(
                                    x,
                                    script_args.add_history_reasoning,
                                    script_args.add_current_reasoning,
                                    revert_test_identity=0,
                                    peer_tag=script_args.tag_peer,
                                    decouple_belief=decouple_internal_belief,
                                    data_type=data_type
                                )}
                            ],
                            "prompt_raw": [
                                {"role": "system", "content": prompt_map[script_args.system_prompt]["default"]},
                                {"role": "user", "content": build_raw_prompt_ds(x)}
                            ]
                        }, num_proc=script_args.preprocessing_num_workers, load_from_cache_file=False, keep_in_memory=True)
                elif is_llama:
                    # Use Llama-specific prompt builders
                    raw_datasets = raw_datasets.map(
                        lambda x: {
                            "prompt": [
                                {"role": "system", "content": prompt_map[script_args.system_prompt]["stage1"]},
                                {"role": "user", "content": build_stage1_prompt_llama(
                                    x,
                                    script_args.add_history_reasoning,
                                    peer_tag=script_args.tag_peer,
                                    prompt_tag=script_args.tag_prompt
                                )}
                            ],
                            "prompt_stage2": [
                                {"role": "system", "content": prompt_map[script_args.system_prompt]["stage2"]},
                                {"role": "user", "content": build_stage2_prompt_llama(
                                    x,
                                    script_args.add_history_reasoning,
                                    script_args.add_current_reasoning,
                                    revert_test_identity=0,
                                    peer_tag=script_args.tag_peer,
                                    decouple_belief=decouple_internal_belief,
                                    data_type=data_type
                                )}
                            ],
                            "prompt_raw": [
                                {"role": "system", "content": prompt_map[script_args.system_prompt]["default"]},
                                {"role": "user", "content": build_raw_prompt_llama(x)}
                            ]
                        }, num_proc=script_args.preprocessing_num_workers, load_from_cache_file=False, keep_in_memory=True)
                else:
                    # Use normal prompt builders
                    raw_datasets = raw_datasets.map(
                        lambda x: {
                            "prompt": [
                                {"role": "system", "content": prompt_map[script_args.system_prompt]["stage1"]},
                                {"role": "user", "content": build_stage1_prompt(
                                    x,
                                    script_args.add_history_reasoning,
                                    peer_tag=script_args.tag_peer,
                                    prompt_tag=script_args.tag_prompt
                                )}
                            ],
                            "prompt_stage2": [
                                {"role": "system", "content": prompt_map[script_args.system_prompt]["stage2"]},
                                {"role": "user", "content": build_stage2_prompt(
                                    x,
                                    script_args.add_history_reasoning,
                                    script_args.add_current_reasoning,
                                    revert_test_identity=0,
                                    peer_tag=script_args.tag_peer,
                                    decouple_belief=decouple_internal_belief,
                                    data_type=data_type
                                )}
                            ],
                            "prompt_raw": [
                                {"role": "system", "content": prompt_map[script_args.system_prompt]["default"]},
                                {"role": "user", "content": build_raw_prompt(x)}
                            ]
                        }, num_proc=script_args.preprocessing_num_workers, load_from_cache_file=False, keep_in_memory=True)
            else:
                if is_qwen3:
                    raw_datasets = raw_datasets.map(
                        lambda x: {"prompt": [
                            {"role": "system", "content": prompt_map[script_args.system_prompt]["default"]},
                            {"role": "user", "content": build_raw_prompt_qwen3(x)}
                        ]}, num_proc=script_args.preprocessing_num_workers, load_from_cache_file=False, keep_in_memory=True)
                elif is_ds:
                    raw_datasets = raw_datasets.map(
                        lambda x: {"prompt": [
                            {"role": "system", "content": prompt_map[script_args.system_prompt]["default"]},
                            {"role": "user", "content": build_raw_prompt_ds(x)}
                        ]}, num_proc=script_args.preprocessing_num_workers, load_from_cache_file=False, keep_in_memory=True)
                elif is_llama:
                    raw_datasets = raw_datasets.map(
                        lambda x: {"prompt": [
                            {"role": "system", "content": prompt_map[script_args.system_prompt]["default"]},
                            {"role": "user", "content": build_raw_prompt_llama(x)}
                        ]}, num_proc=script_args.preprocessing_num_workers, load_from_cache_file=False, keep_in_memory=True)
                else:
                    raw_datasets = raw_datasets.map(
                        lambda x: {"prompt": [
                            {"role": "system", "content": prompt_map[script_args.system_prompt]["default"]},
                            {"role": "user", "content": build_raw_prompt(x)}
                        ]}, num_proc=script_args.preprocessing_num_workers, load_from_cache_file=False, keep_in_memory=True)
        else:
            raise ValueError(f"Invalid system prompt: {script_args.system_prompt}")
            
    # Log a few random samples from the training set:
    if PartialState().is_main_process:
        for index in random.sample(range(len(raw_datasets["train"])), 2):
            logger.info(f"Prompt sample {index} (stage 1) of the raw training set:\n\n{raw_datasets['train'][index]['prompt']}")
            if "prompt_stage2" in raw_datasets['train'][index]:
                logger.info(f"Prompt sample {index} (stage 2) of the raw training set:\n\n{raw_datasets['train'][index]['prompt_stage2']}")
  
    train_dataset = raw_datasets.get("train", None)
    eval_dataset = raw_datasets.get("test", None)

    ################
    # Instantiate GRPO trainer
    ################
    if training_args.model_init_kwargs is None:
        training_args.model_init_kwargs = model_kwargs
    else:
        training_args.model_init_kwargs.update(model_kwargs)
    
    trainer = GRPOTrainerTwoStage(
        model,
        args=training_args,
        is_llama_format=is_llama_format,
        is_qwen3_format=is_qwen3_format,
        is_ds_format=is_ds_format,
        decouple_internal_belief=decouple_internal_belief,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        reward_funcs=REWARD_FUNCS,
        reward_funcs_stage1=REWARD_FUNCS_STAGE1,
        peft_config=get_peft_config(model_args)
    )


    ###############
    # Training loop
    ###############
    logger.info("*** Training ***")
    checkpoint = None
    # Check for last checkpoint
    if training_args.resume_from_checkpoint is not None:
        checkpoint = get_last_checkpoint(training_args.output_dir) if isinstance(training_args.resume_from_checkpoint, bool) else training_args.resume_from_checkpoint
        if checkpoint is not None:
            logger.warning(f"Checkpoint detected, resuming training at {checkpoint=}.")
        else:
            logger.error(f"Failed to load last checkpoint at {checkpoint=}. Start training from scratch")
    
    train_result = trainer.train(resume_from_checkpoint=checkpoint)        
    metrics = train_result.metrics
    metrics["train_samples"] = len(raw_datasets["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    logger.info("*** Training complete ***")
    
    trainer.accelerator.wait_for_everyone()

    # Explicitly destroy DeepSpeed engine
    if hasattr(trainer.model, 'destroy'):
        trainer.model.destroy()

    # Cleanup distributed training
    import torch.distributed as dist
    if dist.is_initialized():
        dist.barrier()  # Ensure all processes synchronize
        dist.destroy_process_group()

    # Cleanup DeepSpeed
    if trainer.deepspeed:
        trainer.deepspeed = None
    
    
if __name__ == "__main__":
    main()
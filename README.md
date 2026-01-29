# Epistemic Context Learning

Codes and data for our paper [Epistemic Context Learning: Building Trust the Right Way in LLM-Based Multi-Agent Systems](https://arxiv.org/abs/2601.XXXXX)

### :mag: Overview

This repository contains three main components:
* [`ECL/`](ECL/): Python source codes for ECL training and evaluation.
* [`recipes/`](recipes/): Training configurations for ECL. See [Configurations](#hammer-configurations).
* [`scripts/`](scripts/): Bash scripts to launch training and evaluation. See [Running Experiments](#rocket-running-experiments).

### :airplane: Getting Started

1. Unzip the proprocessed training and evaluation datasets
```bash
unzip ecl-data.zip -d final_data
```

2. Install dependencies in your Python environment
```bash
conda create -n ecl python==3.11.5
conda activate ecl
pip install -r requirements.txt
```

3. Configure API Keys (**NOTE:** We use OpenRouter for consistent API usage. See [Customized API Usage](#wrench-customized-api-usage) if you want to use separate API calls)
```bash
export OPENROUTER_API_KEY="your_key"
```

### :rocket: Running Experiments

To start training jobs, put the list of configurations in `CONFIGS` of [`scripts/train.sh`](scripts/train.sh) and run
```bash
bash scripts/train.sh
```
Refer to [Training Configurations](#training-configurations) on how to write customized training config files.

To start evaluation jobs, set variables in [`scripts/eval.sh`](scripts/eval.sh) and run
```bash
bash scripts/eval.sh
```
Refer to [Variables in Evaluation Script](#variables-in-evaluation-script) on how to set the variables.

### :hammer: Customized Experiments

#### Training Configurations

Config files needed to reproduce the main evaluation results and analytic results have been stored in [`recipes/train_configs`](recipes/train_configs/). Here we introduce the meanings of key variables:
* `dataset_mixer`: The dataset you want to use or mix. By default we do not mix datasets, for single datasets you can choose among:
<div style="margin-left: auto; margin-right: auto; align: center">

| Value | Used Dataset |
| :--- | :--- |
| `"final_data/gpqa_formatted": 1.0` | GPQA (Adversarial) |
| `"final_data/gpqa_natural": 1.0` | GPQA (Natural) |
| `"final_data/mmlu_pro": 1.0` | MMLU-Pro (Adversarial) |
| `"final_data/mmlu_pro_natural": 1.0` | MMLU-Pro (Natural) |
</div>

* `tag_peer`: The number of peers and history rounds. You can choose among:
<div style="margin-left: auto; margin-right: auto; align: center">

| Value | Used Prompt |
| :--- | :--- |
| `2_1` | 2 peers, 5 rounds of history |
| `3_1` | 3 peers, 5 rounds of history |
| `4_1` | 4 peers, 5 rounds of history (default setting) |
| `4_1_2` | 4 peers, 2 rounds of history |
| `4_1_8` | 4 peers, 8 rounds of history |
</div>

* `tag_prompt`: The prompt you want to use. You can choose among:
<div style="margin-left: auto; margin-right: auto; align: center">

| Value | Used Prompt |
| :--- | :--- |
| `AG` | History-agnostic aggregator (`AG`) |
| `NS` | Default SA & ECL prompt (`ECL (I)`) |
| `JP` | Explicit peer recognition (`ECL (E)`) |
</div>

* `add_peers`: `MA context` will be used when set as `true`, `SA context` when set as `false`.
* `add_current_reasoning`: `MA-Reasoning` context will be used when set as `true`, `MA-Outcome` when set as `false`. **Valid only when `add_peers` set as `true`**.
* `output_dir`: This variable serves several purposes, please specify in forms of `SAVE_ROOT/DATASET/CONFIG_NAME`
    - `SAVE_ROOT` only specifies the path for saving model checkpoints and does not affect the setting.
    - `DATASET` determines whether natural or adversarial task setting is used. Adding `_natural` suffix like `gpqa_natural` will select the natural setting, otherwise the adversarial setting.
    - `CONFIG_NAME` determines the rewards in RL and whether the `DB` trick is used.
        + Including `-DB` will involve the `DB` trick.
        + Only include `-PRR` when you train models for `ECL (E)` so that our auxiliary PRR will be used in stage 1, and include `-OR` otherwise.

#### Variables in Evaluation Script

* `VLLM_MODELS`: The model (checkpoint) you want to deploy via `vLLM`. If you use API models, skip this field.
* `VLLM_IPS`: The IP address where you want to deploy model.
* `VLLM_PORT_NUMBERS`: The port number where you want to deploy model.
* `VLLM_CUDA_DEVICES`: The GPU device(s) you use to deploy the model.
* `MAX_LENGTH`: The server-side max length for vLLM.
* `EVAL_MODELS`: The model you want to evaluate, corresponding to `VLLM_MODELS`.
* `EVAL_DATASETS`: The dataset (test split) you want to use for evaluation.
* `EVAL_CURRENT_REASON`: Set as `true` to use `MA-Reasoning` context or `false` to use `MA-Outcome`.
* `EVAL_REVERT_IDENTITY`: Set as `0` for default, `1` for `Flip` analysis and `2` for `All-W` analysis.
* `EVAL_IPS`: IP address of the model you want to evaluate, corresponding to `VLLM_IPS`.
* `EVAL_PORT_NUMBERS`: Port number of the model you want to evaluate, corresponding to `VLLM_PORT_NUMBERS`.
* `EVAL_TAG_PEER`: The number of peers and history rounds. See `tag_peer` in [Training Configurations](#training-configurations).
* `EVAL_TAG_PROMPT`: The prompt you want to use. See `tag_prompt` in [Training Configurations](#training-configurations).
* `EVAL_DECOUPLE_BELIEF`: Whether to use the `DB` trick in evaluation.
* `SAVE_ROOT`: The folder you want to save the evaluation result files and logs.

**NOTE:**
1. If you want to run multiple evaluation jobs in parallel, just specify them as a list and make sure model names, ips, and port numbers are aligned correctly.
2. If you want to run different settings for the same model in one job, specify them as a space-separated string.

Refer to [`scripts/eval.sh`](scripts/eval.sh) for some examples.

### :wrench: Customized API Usage

In [`ECL/utils/eval_utils.py`](ECL/utils/eval_utils.py), we implement API calls for all models via OpenRouter for simplicity. If you need to call APIs in a different way, modify the [`get_llm_client`](ECL/utils/eval_utils.py#L533), [`generate_llm_chat`](ECL/utils/eval_utils.py#L556) and [`generate_llm_chat_two_stage`](ECL/utils/eval_utils.py#L581) functions and define your own ways of API usage, and setup specific `API_KEYs` in your environment.

### :speech_balloon: Contact & Citation

If you have any issues or questions regarding our ECL project, do not hesitate to contact: [Ruiwen Zhou](mailto:zhouruiwen@u.nus.edu).

If you found this work helpful, please consider citing it using the following:

```
@article{zhou2026ecl,
  title={Epistemic Context Learning: Building Trust the Right Way in LLM-Based Multi-Agent Systems},
  author={Zhou, Ruiwen and Song, Maojia and Wu, Xiaobao and Cheng, Sitao and Yin, Xunjian and Xie, Yuxi and Hao, Zhuoqun and Hua, Wenyue and Pan, Liangming and Poria, Soujanya and Kan, Min-Yen},
  journal={arXiv preprint arXiv:2601.XXXXX},
  year={2026}
}
```
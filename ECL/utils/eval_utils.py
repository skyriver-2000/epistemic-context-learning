import json
import os
import random
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from time import sleep
from copy import deepcopy
from typing import List, Dict, Tuple, Callable
import json
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI

from ECL.utils import set_seed

set_seed(42)


OPENAI_MODELS = ['openai/gpt-5-mini', 'openai/gpt-5', 'openai/gpt-5.2']
GOOGLE_MODELS = ['google/gemini-3-flash-preview', 'google/gemini-3-pro-preview']
DEEPSEEK_MODELS = ['deepseek/deepseek-v3.2']
AZURE_OPENAI_MODELS = ['gpt4o', "o4-mini"]
API_MODELS = OPENAI_MODELS + GOOGLE_MODELS + AZURE_OPENAI_MODELS + DEEPSEEK_MODELS
DEFAULT_MAX_RETRIES = 3

HISTORY_PROTOCOLS = ['0', '25', '50', '75', '100']
CURRENT_PROTOCOLS = ['SUPPORT', 'OPPOSEHARD', 'OPPOSEEASY']
RESPONSE_PROTOCOLS = ['WRONG', 'CORRECT']
CONFIDENCE_PROTOCOLS = ['LOW', 'HIGH']


SENTENCES = ["The best answer is \"{text}\"", "I'd like to choose \"{text}\"", "I'm going with \"{text}\"", "I'd select \"{text}\" as the best answer", "My choice would be \"{text}\"", "I think \"{text}\" is the most suitable option", "I believe \"{text}\" is the right answer",
    "I'm leaning towards \"{text}\" as the best choice", "I'd opt for \"{text}\" in this case", "I'd say \"{text}\" is the correct response", "My preference would be \"{text}\"", "I'd settle on \"{text}\" as the best answer", "I'd vote for \"{text}\" as the most appropriate choice",
    "I'm inclined to choose \"{text}\"", "I'd endorse \"{text}\" as the top pick", "I'd consider \"{text}\" as the most accurate answer", "I'd side with \"{text}\" as the best response", "I'd favor \"{text}\" as the most fitting option", "I'd stand by \"{text}\" as the correct answer",
    "I'd affirm \"{text}\" as the best selection", "I'd vouch for \"{text}\" as the most precise answer"]

OTHER_AGENTS = ['Mary', 'John', 'George', 'Tom', 'Tony', 'Jack']

SYSTEM_PROMPT_DEBATE = """You are a thoughtful AI assistant. \
Before responding, engage in a multi-turn internal debate within <think>...</think>. \
This debate is based on prior context and your own initiative—it explores possible questions, angles, or uncertainties, not necessarily responding to the user yet. \
Each line begins with a distinct, adjective-labeled voice (e.g., Curious voice:, Skeptical voice:), and the voices build on each other across multiple turns. \
After the internal debate, respond to the user's instruction within <answer>...</answer>. \
Respond strictly in the following format:
```
<think>
(Distinct, adjective-tagged voices in a meaningful debate)
</think>

<answer>
(Formal response to the user's instruction)
</answer>
```"""

SYSTEM_PROMPT_NORMAL = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think>\nreasoning process here\n</think>\n\n<answer>\nanswer here\n</answer>"
)

SYSTEM_PROMPT_STAGE1 = (
    "A conversation between User and Assistant. The user provides a history of QA from peers, and the Assistant summerizes "
    # "key information."
    "key information, especially whether each peer has correctly answered each previous question and their overall accuracy. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the summary. "
    "The reasoning process and summary are enclosed within <think> </think> and <summary> </summary> tags, i.e., "
    "<think>\nreasoning process here\n</think>\n\n<summary>\nsummary here\n</summary>"
)

SYSTEM_PROMPT_STAGE2 = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think>\nreasoning process here\n</think>\n\n<answer>\nanswer here\n</answer>"
)

# Llama-specific system prompts with #### separator
SYSTEM_PROMPT_NORMAL_LLAMA = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The response should be formatted as:\n"
    "#### Reasoning\n"
    "reasoning process here\n\n"
    "#### Answer\n"
    "answer here"
)

SYSTEM_PROMPT_STAGE1_LLAMA = (
    "A conversation between User and Assistant. The user provides a history of QA from peers, and the Assistant summerizes "
    "key information, especially whether each peer has correctly answered each previous question and their overall accuracy. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the summary. "
    "The response should be formatted as:\n"
    "#### Reasoning\n"
    "reasoning process here\n\n"
    "#### Summary\n"
    "summary here"
)

SYSTEM_PROMPT_STAGE2_LLAMA = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The response should be formatted as:\n"
    "#### Reasoning\n"
    "reasoning process here\n\n"
    "#### Answer\n"
    "answer here"
)

# Qwen3-specific system prompts - no tags for answer/summary
SYSTEM_PROMPT_NORMAL_QWEN3 = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind within [think] [/think] tags, and then provides the answer "
    "directly after without any tags, i.e., "
    "[think]\nreasoning process here\n[/think]\n\nanswer here"
)

SYSTEM_PROMPT_STAGE1_QWEN3 = (
    "A conversation between User and Assistant. The user provides a history of QA from peers, and the Assistant summerizes "
    "key information, especially whether each peer has correctly answered each previous question and their overall accuracy. "
    "The assistant first thinks about the reasoning process in the mind within [think] [/think] tags, and then provides "
    "the summary directly after without any tags, i.e., "
    "[think]\nreasoning process here\n[/think]\n\nsummary here"
)

SYSTEM_PROMPT_STAGE2_QWEN3 = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind within [think] [/think] tags, and then provides the answer "
    "directly after without any tags, i.e., "
    "[think]\nreasoning process here\n[/think]\n\nanswer here"
)

# DS (DeepSeek)-specific system prompts - uses <think></think> tags but no tags for answer/summary
SYSTEM_PROMPT_NORMAL_DS = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind within <think> </think> tags, and then provides the answer "
    "directly after without any tags, i.e., "
    "<think>\nreasoning process here\n</think>\n\nanswer here"
)

SYSTEM_PROMPT_STAGE1_DS = (
    "A conversation between User and Assistant. The user provides a history of QA from peers, and the Assistant summerizes "
    "key information, especially whether each peer has correctly answered each previous question and their overall accuracy. "
    "The assistant first thinks about the reasoning process in the mind within <think> </think> tags, and then provides "
    "the summary directly after without any tags, i.e., "
    "<think>\nreasoning process here\n</think>\n\nsummary here"
)

SYSTEM_PROMPT_STAGE2_DS = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind within <think> </think> tags, and then provides the answer "
    "directly after without any tags, i.e., "
    "<think>\nreasoning process here\n</think>\n\nanswer here"
)

FORMAT_STR = "Q: {question}\n\nAnswer choices:\n{options}"

PROMPT_TEMPLATE = """You are invited to participate in a Q&A. There are {number} other players invited along with you. Here are the previous Q&A history:

---- begin of history ----
{history}
---- end of history ----

Here are the current Q&A content:

{current}

Please give your answer in the following format, and output answer strictly as "The best answer is: \"(X) the content of the answer\"" between <answer> and </answer>:
<think>
Your reasoning process here.
</think>

<answer>
The best answer is: \"(X) the content of the answer\"
</answer>"""

PROMPT_TEMPLATE_JUDGE_PEER = """You are invited to participate in a Q&A. There are {number} other players invited along with you. Here are the previous Q&A history:

---- begin of history ----
{history}
---- end of history ----

Here are the current Q&A content:

{current}

In your reasoning process you should first analyze the trustworthiness of each peer by considering their responses to each question in the history and the current question. Then, identify the most reliable peer among them in your answer.
Please give your answer in the following format, and output answer strictly as "The most reliable peer agent: the name of the peer" and "The best answer is: \"(X) the content of the answer\"" in two separate lines between <answer> and </answer>:
<think>
Your reasoning process here.
</think>

<answer>
The most reliable peer agent: the name of the peer
The best answer is: \"(X) the content of the answer\"
</answer>"""

PROMPT_TEMPLATE_STAGE1 = """You are invited to participate in a Q&A. There are {number} other players invited along with you. Here are the previous Q&A history:

---- begin of history ----
{history}
---- end of history ----

Please give your response in the following format, and output the summary strictly between <summary> and </summary>:
<think>
Your reasoning process here.
</think>

<summary>
Your summary here.
</summary>"""

PROMPT_TEMPLATE_STAGE1_JUDGE_PEER = """You are invited to participate in a Q&A. There are {number} other players invited along with you. Here are the previous Q&A history:

---- begin of history ----
{history}
---- end of history ----

Please analyze the responses of each player in the history and identify which player is the most trustworthy based on their accuracy in answering previous questions.
Please give your response in the following format, and output the summary strictly between <summary> and </summary>, where PLAYER_NAME is the name of the most trustworthy player:
<think>
Your reasoning process here.
</think>

<summary>
The most trustworthy agent is: PLAYER_NAME
</summary>"""

PROMPT_TEMPLATE_STAGE2 = """You are invited to participate in a Q&A. There are {number} other players invited along with you.

From the previous Q&A history, you have summarized the following key information:

{output_stage1}

Here are the current Q&A content:

{current}

Please give your answer in the following format, and output answer strictly as "The best answer is: \"(X) the content of the answer\"" between <answer> and </answer>:
<think>
Your reasoning process here.
</think>

<answer>
The best answer is: \"(X) the content of the answer\"
</answer>"""

RAW_PROMPT_TEMPLATE = """You are invited to participate in a Q&A. Here are the current Q&A content:

{current}

Please give your answer in the following format, and output answer strictly as "The best answer is: \"(X) the content of the answer\"" between <answer> and </answer>:
<think>
Your reasoning process here.
</think>

<answer>
The best answer is: \"(X) the content of the answer\"
</answer>"""

# Llama-specific prompt templates with #### separator
PROMPT_TEMPLATE_LLAMA = """You are invited to participate in a Q&A. There are {number} other players invited along with you. Here are the previous Q&A history:

---- begin of history ----
{history}
---- end of history ----

Here are the current Q&A content:

{current}

Please give your answer in the following format:
#### Reasoning
Your reasoning process here.

#### Answer
The best answer is: "(X) the content of the answer"
"""

PROMPT_TEMPLATE_STAGE1_LLAMA = """You are invited to participate in a Q&A. There are {number} other players invited along with you. Here are the previous Q&A history:

---- begin of history ----
{history}
---- end of history ----

Please give your response in the following format:
#### Reasoning
Your reasoning process here.

#### Summary
Your summary here.
"""

PROMPT_TEMPLATE_STAGE1_JUDGE_PEER_LLAMA = """You are invited to participate in a Q&A. There are {number} other players invited along with you. Here are the previous Q&A history:

---- begin of history ----
{history}
---- end of history ----

Please analyze the responses of each player in the history and identify which player is the most trustworthy based on their accuracy in answering previous questions.
Please give your response in the following format, where PLAYER_NAME is the name of the most trustworthy player:
#### Reasoning
Your reasoning process here.

#### Summary
The most trustworthy agent is: PLAYER_NAME
"""

PROMPT_TEMPLATE_STAGE2_LLAMA = """You are invited to participate in a Q&A. There are {number} other players invited along with you.

From the previous Q&A history, you have summarized the following key information:

{output_stage1}

Here are the current Q&A content:

{current}

Please give your answer in the following format:
#### Reasoning
Your reasoning process here.

#### Answer
The best answer is: "(X) the content of the answer"
"""

RAW_PROMPT_TEMPLATE_LLAMA = """You are invited to participate in a Q&A. Here are the current Q&A content:

{current}

Please give your answer in the following format:
#### Reasoning
Your reasoning process here.

#### Answer
The best answer is: "(X) the content of the answer"
"""

# DS-Distilled prompt templates
PROMPT_TEMPLATE_DS = """You are invited to participate in a Q&A. There are {number} other players invited along with you. Here are the previous Q&A history:

---- begin of history ----
{history}
---- end of history ----

Here are the current Q&A content:

{current}

Please give your answer in the following format:
<think>
Your reasoning process here.
</think>

The best answer is: "(X) the content of the answer"
"""

PROMPT_TEMPLATE_STAGE1_DS = """You are invited to participate in a Q&A. There are {number} other players invited along with you. Here are the previous Q&A history:

---- begin of history ----
{history}
---- end of history ----

Please give your response in the following format:
<think>
Your reasoning process here.
</think>

Your summary here.
"""

PROMPT_TEMPLATE_STAGE1_JUDGE_PEER_DS = """You are invited to participate in a Q&A. There are {number} other players invited along with you. Here are the previous Q&A history:

---- begin of history ----
{history}
---- end of history ----

Please analyze the responses of each player in the history and identify which player is the most trustworthy based on their accuracy in answering previous questions.
Please give your response in the following format, where PLAYER_NAME is the name of the most trustworthy player:
<think>
Your reasoning process here.
</think>

The most trustworthy agent is: PLAYER_NAME
"""

PROMPT_TEMPLATE_STAGE2_DS = """You are invited to participate in a Q&A. There are {number} other players invited along with you.

From the previous Q&A history, you have summarized the following key information:

{output_stage1}

Here are the current Q&A content:

{current}

Please give your answer in the following format:
<think>
Your reasoning process here.
</think>

The best answer is: "(X) the content of the answer"
"""

RAW_PROMPT_TEMPLATE_DS = """You are invited to participate in a Q&A. Here are the current Q&A content:

{current}

Please give your answer in the following format:
<think>
Your reasoning process here.
</think>

The best answer is: "(X) the content of the answer"
"""

# Qwen3-specific prompt templates - no tags for answer/summary
PROMPT_TEMPLATE_QWEN3 = """You are invited to participate in a Q&A. There are {number} other players invited along with you. Here are the previous Q&A history:

---- begin of history ----
{history}
---- end of history ----

Here are the current Q&A content:

{current}

Please give your answer in the following format:
[think]
Your reasoning process here.
[/think]

The best answer is: "(X) the content of the answer"
"""

PROMPT_TEMPLATE_STAGE1_QWEN3 = """You are invited to participate in a Q&A. There are {number} other players invited along with you. Here are the previous Q&A history:

---- begin of history ----
{history}
---- end of history ----

Please give your response in the following format:
[think]
Your reasoning process here.
[/think]

Your summary here.
"""

PROMPT_TEMPLATE_STAGE1_JUDGE_PEER_QWEN3 = """You are invited to participate in a Q&A. There are {number} other players invited along with you. Here are the previous Q&A history:

---- begin of history ----
{history}
---- end of history ----

Please analyze the responses of each player in the history and identify which player is the most trustworthy based on their accuracy in answering previous questions.
Please give your response in the following format, where PLAYER_NAME is the name of the most trustworthy player:
[think]
Your reasoning process here.
[/think]

The most trustworthy agent is: PLAYER_NAME
"""

PROMPT_TEMPLATE_STAGE2_QWEN3 = """You are invited to participate in a Q&A. There are {number} other players invited along with you.

From the previous Q&A history, you have summarized the following key information:

{output_stage1}

Here are the current Q&A content:

{current}

Please give your answer in the following format, and output answer strictly as "The best answer is: \"(X) the content of the answer\"" after [/think]:
[think]
Your reasoning process here.
[/think]

The best answer is: "(X) the content of the answer"
"""

RAW_PROMPT_TEMPLATE_QWEN3 = """You are invited to participate in a Q&A. Here are the current Q&A content:

{current}

Please give your answer in the following format:
[think]
Your reasoning process here.
[/think]

The best answer is: "(X) the content of the answer"
"""

RAW_PROMPT_TEMPLATE_QWEN3_STRONG_FORMAT = """You are invited to participate in a Q&A. Here are the current Q&A content:

{current}

Please give your answer in the following format, and output answer strictly as "The best answer is: \"(X) the content of the answer\"" after [/think]:
[think]
Your reasoning process here.
[/think]

The best answer is: "(X) the content of the answer"
"""


def add_retries(f: Callable):
    def wrap(*args, **kwargs):
        max_retries = kwargs.get("max_retries", DEFAULT_MAX_RETRIES)
        num_retries = 0
        while True:
            try:
                result = f(*args, **kwargs)
                return result
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except KeyError:
                raise KeyError
            except Exception as e:
                print("Error: ", traceback.format_exc(), "\nRetrying in ", num_retries * 2, "seconds")
                if num_retries == max_retries:
                    traceback.print_exc()
                    return {"completion": traceback.format_exc()}
                num_retries += 1
                sleep(num_retries * 2)
    return wrap


def get_llm_client(model_name_or_path: str, **model_init_kwargs: Dict):
    if model_name_or_path in API_MODELS:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            timeout=1200
        )
        return client
    else:
        try:
            openai_api_key = "EMPTY"
            openai_api_base = f"http://{model_init_kwargs.pop('ip', None)}:{model_init_kwargs.pop('port_number', None)}/v1"
            client = OpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base,
                timeout=1200
            )
            return client
        except:
            raise ValueError("Unknown model")


@add_retries
def generate_llm_chat(client, model_name_or_path: str, user_prompt: str, gen_kwargs: Dict, max_retries: int = 3) -> List[str]:
    # Check if this is a Llama or Qwen3 model
    is_ds = "deepseek" in model_name_or_path.lower() or \
        model_name_or_path in API_MODELS
    is_llama = "llama" in model_name_or_path.lower()
    is_qwen3 = "qwen3" in model_name_or_path.lower()
    
    # Select appropriate system prompts based on model type
    system_prompt = SYSTEM_PROMPT_NORMAL_DS if is_ds else (SYSTEM_PROMPT_NORMAL_LLAMA if is_llama else (SYSTEM_PROMPT_NORMAL_QWEN3 if is_qwen3 else SYSTEM_PROMPT_NORMAL))
    
    messages = (
        [{"role": "system", "content": system_prompt}] +
        [{'role': 'user', 'content': user_prompt}]
    )
    
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model_name_or_path,
        extra_body=gen_kwargs
    )
    responses = [choice.message.content for choice in chat_completion.choices]
    return responses


@add_retries
def generate_llm_chat_two_stage(client, model_name_or_path: str, stage1_prompt: str, stage2_prompt_template: str, gen_kwargs_stage1: Dict, gen_kwargs_stage2: Dict, max_retries: int = 3) -> Tuple[str, List[str]]:
    """
    Two-stage generation with different system prompts for each stage.
    
    Args:
        client: LLM client
        model_name_or_path: Model name or path
        stage1_prompt: User prompt for stage 1 (summary generation)
        stage2_prompt_template: User prompt template for stage 2 with {output_stage1} placeholder
        gen_kwargs_stage1: Generation kwargs for stage 1
        gen_kwargs_stage2: Generation kwargs for stage 2
        max_retries: Maximum number of retries
        
    Returns:
        Tuple of (stage1_response, stage2_responses_list)
    """
    # Check if this is a Llama or Qwen3 model
    is_ds = "deepseek" in model_name_or_path.lower() or \
        model_name_or_path in API_MODELS
    is_llama = "llama" in model_name_or_path.lower()
    is_qwen3 = "qwen3" in model_name_or_path.lower()
    
    # Select appropriate system prompts based on model type
    stage1_system_prompt = SYSTEM_PROMPT_STAGE1_DS if is_ds else (SYSTEM_PROMPT_STAGE1_LLAMA if is_llama else (SYSTEM_PROMPT_STAGE1_QWEN3 if is_qwen3 else SYSTEM_PROMPT_STAGE1))
    stage2_system_prompt = SYSTEM_PROMPT_STAGE2_DS if is_ds else (SYSTEM_PROMPT_STAGE2_LLAMA if is_llama else (SYSTEM_PROMPT_STAGE2_QWEN3 if is_qwen3 else SYSTEM_PROMPT_STAGE2))
    
    # Stage 1: Generate summary with appropriate SYSTEM_PROMPT_STAGE1
    messages_stage1 = (
        [{"role": "system", "content": stage1_system_prompt}] +
        [{'role': 'user', 'content': stage1_prompt}]
    )
    
    chat_completion_stage1 = client.chat.completions.create(
        messages=messages_stage1,
        model=model_name_or_path,
        extra_body=gen_kwargs_stage1
    )
    response_stage1 = chat_completion_stage1.choices[0].message.content
    
    # Extract summary from stage 1 response
    try:
        if is_ds:
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
            # For Qwen3 format, extract summary after [/think]
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
                # If tags not found, use the entire response
                output_stage1 = response_stage1
    except Exception:
        output_stage1 = response_stage1

    # Stage 2: Generate final answer with appropriate SYSTEM_PROMPT_STAGE2
    stage2_prompt = stage2_prompt_template.replace("{output_stage1}", output_stage1)
    
    messages_stage2 = (
        [{"role": "system", "content": stage2_system_prompt}] +
        [{'role': 'user', 'content': stage2_prompt}]
    )
    
    chat_completion_stage2 = client.chat.completions.create(
        messages=messages_stage2,
        model=model_name_or_path,
        extra_body=gen_kwargs_stage2
    )
    responses_stage2 = [choice.message.content for choice in chat_completion_stage2.choices]
    
    return response_stage1, responses_stage2


def build_example_adv(sample: Dict, history_key: bool = False, add_current_reasoning: bool = True, revert_test_identity: int = 0, tag: str = "", decouple_belief: bool = False) -> str:
    """
    Build the example prompt based on the sample and protocol.
    Only support 1 correct agent at present.
    """
    agents = sample[history_key]["agents"]
    reliable_agent = sample[history_key]["reliable_agent"]
    n_agents = len(agents)
    agent_behaviors = np.zeros((n_agents, ))
    if revert_test_identity == 0:
        agent_behaviors[reliable_agent] += 1
    elif revert_test_identity == 1:
        reliable = reliable_agent if isinstance(reliable_agent, list) else [reliable_agent]
        irreliable = list(set(range(n_agents)) - set(reliable))
        if len(reliable) > len(irreliable):
            agent_behaviors[irreliable] = 1    # set irreliable agents reliable
            other_reliable = random.sample(reliable, len(reliable) - len(irreliable))
            agent_behaviors[other_reliable] = 1    # set some (originally) reliable agents reliable
        else:
            reliable_idx = random.sample(irreliable, len(reliable))
            agent_behaviors[reliable_idx] = 1    # set some (originally) irreliable agents reliable
    elif revert_test_identity == 2:
        pass    # keep all agents irreliable
    output_strings = random.sample(SENTENCES, n_agents)
    agent_strings = []
    
    # permute `gt_reasoning`, `wrong_options`, and `gt_reasoning`
    gt_reasonings = deepcopy(sample["gt_reasoning"]) if isinstance(sample["gt_reasoning"], list) else [sample["gt_reasoning"]]
    wrong_options = deepcopy(sample["wrong_options"])
    wrong_reasonings = deepcopy(sample["wrong_reasoning"])
    num_reasonings = len(wrong_reasonings) if isinstance(wrong_reasonings[0], list) else 1
    wrong_options = wrong_options * num_reasonings
    wrong_reasonings = sum(wrong_reasonings, []) if isinstance(wrong_reasonings[0], list) else wrong_reasonings
    
    # Filter out too long reasonings
    # Calculate per-agent budget: assuming 32k max tokens, reserve ~8k for prompt/format/question
    # Each agent should use at most (24000 / n_agents) tokens ≈ (18000 / n_agents) words
    # To be safe, use (16000 / n_agents) words per agent
    max_words_per_agent = max(500, 16000 // max(1, n_agents))  # at least 500 words
    max_char_per_agent = max_words_per_agent * 5  # approx. 5 characters per word

    valid_wrong_response_idx = [i for i, r in enumerate(wrong_reasonings) if len(r.split()) < max_words_per_agent]
    valid_wrong_response_idx = [i for i in valid_wrong_response_idx if len(wrong_reasonings[i]) < max_char_per_agent]
    wrong_options = [wrong_options[i] for i in valid_wrong_response_idx]
    wrong_reasonings = [wrong_reasonings[i] for i in valid_wrong_response_idx]
    gt_reasonings = [r for r in gt_reasonings if len(r.split()) < max_words_per_agent]
    gt_reasonings = [r for r in gt_reasonings if len(r) < max_char_per_agent]

    random.shuffle(gt_reasonings)
    order = np.random.permutation(len(wrong_options))
    wrong_options = [wrong_options[i] for i in order]
    wrong_reasonings = [wrong_reasonings[i] for i in order]

    for agent, correct, outp_string in zip(agents, agent_behaviors, output_strings):
        if correct:
            option, reason = sample["gt_option"], gt_reasonings.pop(0)
        else:
            option, reason = wrong_options.pop(0), wrong_reasonings.pop(0)
        reasoning_text = (reason + "\n\n") if add_current_reasoning else ""
        agent_strings.append(
            agent + ": " + reasoning_text + outp_string.format(text=option)
        )
    agent_strings = "\n\n".join(agent_strings)
    if decouple_belief:
        example_sample_str = f"{sample['formatted_question']}\n\nYour own belief: $decoupled_belief\n\n{agent_strings}"
    else:
        example_sample_str = f"{sample['formatted_question']}\n\n{agent_strings}"
    return example_sample_str

def build_example_natural(sample: Dict, history_key: bool = False, add_current_reasoning: bool = True, revert_test_identity: int = 0, tag: str = "", decouple_belief: bool = False) -> str:
    """
    Build the example prompt based on the sample and protocol.
    Only support 1 correct agent at present.
    """
    agents = sample[history_key]["agents"]
    reliable_agent = sample[history_key]["reliable_agent"]
    wrong_agent_order = sample[history_key].get("wrong_agent_order", None)
    n_agents = len(agents)
    agent_behaviors = np.zeros((n_agents, ))
    if revert_test_identity == 0:
        agent_behaviors[reliable_agent] += 1
    elif revert_test_identity == 1:
        reliable = reliable_agent if isinstance(reliable_agent, list) else [reliable_agent]
        irreliable = list(set(range(n_agents)) - set(reliable))
        if len(reliable) > len(irreliable):
            agent_behaviors[irreliable] = 1    # set irreliable agents reliable
            other_reliable = random.sample(reliable, len(reliable) - len(irreliable))
            agent_behaviors[other_reliable] = 1    # set some (originally) reliable agents reliable
        else:
            reliable_idx = random.sample(irreliable, len(reliable))
            agent_behaviors[reliable_idx] = 1    # set some (originally) irreliable agents reliable
    elif revert_test_identity == 2:
        pass    # keep all agents irreliable
    output_strings = random.sample(SENTENCES, n_agents)
    agent_strings = []
    
    # permute `gt_reasoning`, `wrong_options`, and `gt_reasoning`
    gt_options = deepcopy(sample["pseudo_gt_option"])
    gt_reasonings = deepcopy(sample["gt_reasoning"])
    wrong_options = deepcopy(sample["pseudo_wrong_options"])
    wrong_reasonings = deepcopy(sample["wrong_reasoning"])
        
    # Shuffle wrong agent order
    if wrong_agent_order is not None:
        wrong_options = [wrong_options[i] for i in wrong_agent_order]
        wrong_reasonings = [wrong_reasonings[i] for i in wrong_agent_order]
    
    # Filter out too long reasonings
    # Calculate per-agent budget: assuming 32k max tokens, reserve ~8k for prompt/format/question
    # Each agent should use at most (24000 / n_agents) tokens ≈ (18000 / n_agents) words
    # To be safe, use (16000 / n_agents) words per agent
    max_words_per_agent = max(500, 16000 // max(1, n_agents))  # at least 500 words
    max_char_per_agent = max_words_per_agent * 4  # approx. 4 characters per word

    # Remove overlength reasonings
    valid_gt_response_idx = [i for i, r in enumerate(gt_reasonings) if len(r) < max_char_per_agent and len(r.split()) < max_words_per_agent]
    if len(valid_gt_response_idx) == 0:
        valid_gt_response_idx = [np.argmin([len(r) for r in gt_reasonings])]
    gt_options = [gt_options[i] for i in valid_gt_response_idx]
    gt_reasonings = [gt_reasonings[i] for i in valid_gt_response_idx]
    for i in range(len(wrong_options)):
        valid_wrong_response_idx = [j for j, r in enumerate(wrong_reasonings[i]) if len(r) < max_char_per_agent and len(r.split()) < max_words_per_agent]
        if len(valid_wrong_response_idx) == 0:
            valid_wrong_response_idx = [np.argmin([len(r) for r in wrong_reasonings[i]])]
        wrong_options[i] = [wrong_options[i][j] for j in valid_wrong_response_idx]
        wrong_reasonings[i] = [wrong_reasonings[i][j] for j in valid_wrong_response_idx]

    # Shuffle options and reasoning
    order = np.random.permutation(len(gt_reasonings))
    gt_options = [gt_options[i] for i in order]
    gt_reasonings = [gt_reasonings[i] for i in order]

    for i in range(len(wrong_options)):
        order = np.random.permutation(len(wrong_options[i]))
        wrong_options[i] = [wrong_options[i][j] for j in order]
        wrong_reasonings[i] = [wrong_reasonings[i][j] for j in order]

    wrong_agent_cnt = 0
    for agent, correct, outp_string in zip(agents, agent_behaviors, output_strings):
        if correct:
            option, reason = gt_options[0], gt_reasonings[0]
        else:
            option, reason = wrong_options[wrong_agent_cnt][0], wrong_reasonings[wrong_agent_cnt][0]
            wrong_agent_cnt += 1
        reasoning_text = (reason + "\n\n") if add_current_reasoning else ""
        agent_strings.append(
            agent + ": " + reasoning_text + outp_string.format(text=option)
        )
    agent_strings = "\n\n".join(agent_strings)
    if decouple_belief:
        example_sample_str = f"{sample['formatted_question']}\n\nYour own belief: $decoupled_belief\n\n{agent_strings}"
    else:
        example_sample_str = f"{sample['formatted_question']}\n\n{agent_strings}"
    return example_sample_str

def build_raw_prompt(sample: Dict):
    prompt = RAW_PROMPT_TEMPLATE.format(current=sample['formatted_question'])
    return prompt

def build_complete_prompt(sample: Dict, add_history_reasoning: bool, add_current_reasoning: bool, revert_test_identity: int = 0, peer_tag: str = "", prompt_tag: str = "", decouple_belief: bool = False, data_type: str = "adv"):
    key = "history_with_reason" if add_history_reasoning else "history"
    if peer_tag != "":
        key = key + "_" + peer_tag
    assert data_type in ["adv", "nat"], "data_type must be 'adv' or 'nat'"
    build_example_fn = build_example_natural if data_type == "nat" else build_example_adv
    current = build_example_fn(sample, key, add_current_reasoning, revert_test_identity, peer_tag, decouple_belief)
    if "AG" in prompt_tag:
        prompt = RAW_PROMPT_TEMPLATE.format(current=current)
    else:
        prompt = PROMPT_TEMPLATE.format(
            number=len(sample[key]["agents"]),
            history=sample[key]["text"],
            current=current
        )
    return prompt

def build_ag_prompt(sample: Dict, add_history_reasoning: bool, add_current_reasoning: bool, revert_test_identity: int = 0, peer_tag: str = "", decouple_belief: bool = False, data_type: str = "adv"):
    key = "history_with_reason" if add_history_reasoning else "history"
    if peer_tag != "":
        key = key + "_" + peer_tag
    assert data_type in ["adv", "nat"], "data_type must be 'adv' or 'nat'"
    build_example_fn = build_example_natural if data_type == "nat" else build_example_adv
    current = build_example_fn(sample, key, add_current_reasoning, revert_test_identity, peer_tag, decouple_belief)
    prompt = RAW_PROMPT_TEMPLATE.format(current=current)
    return prompt

def build_stage1_prompt(sample: Dict, add_history_reasoning: bool, peer_tag: str = "", prompt_tag: str = ""):
    key = "history_with_reason" if add_history_reasoning else "history"
    if peer_tag != "":
        key = key + "_" + peer_tag
    template = PROMPT_TEMPLATE_STAGE1_JUDGE_PEER if "JP" in prompt_tag else PROMPT_TEMPLATE_STAGE1
    prompt = template.format(
        number=len(sample[key]["agents"]),
        history=sample[key]["text"]
    )
    return prompt

def build_stage2_prompt(sample: Dict, add_history_reasoning: bool, add_current_reasoning: bool, revert_test_identity: int = 0, peer_tag: str = "", decouple_belief: bool = False, data_type: str = "adv"):
    key = "history_with_reason" if add_history_reasoning else "history"
    if peer_tag != "":
        key = key + "_" + peer_tag
    assert data_type in ["adv", "nat"], "data_type must be 'adv' or 'nat'"
    build_example_fn = build_example_natural if data_type == "nat" else build_example_adv
    current = build_example_fn(sample, key, add_current_reasoning, revert_test_identity, peer_tag, decouple_belief)
    prompt = PROMPT_TEMPLATE_STAGE2.replace("{current}", current)
    prompt = prompt.replace("{number}", str(len(sample[key]["agents"])))
    return prompt


# Llama-specific build functions
def build_raw_prompt_llama(sample: Dict):
    prompt = RAW_PROMPT_TEMPLATE_LLAMA.format(current=sample['formatted_question'])
    return prompt

def build_complete_prompt_llama(sample: Dict, add_history_reasoning: bool, add_current_reasoning: bool, revert_test_identity: int = 0, peer_tag: str = "", prompt_tag: str = "", decouple_belief: bool = False, data_type: str = "adv"):
    key = "history_with_reason" if add_history_reasoning else "history"
    if peer_tag != "":
        key = key + "_" + peer_tag
    assert data_type in ["adv", "nat"], "data_type must be 'adv' or 'nat'"
    build_example_fn = build_example_natural if data_type == "nat" else build_example_adv
    current = build_example_fn(sample, key, add_current_reasoning, revert_test_identity, peer_tag, decouple_belief)
    if "AG" in prompt_tag:
        prompt = RAW_PROMPT_TEMPLATE_LLAMA.format(current=current)
    else:
        prompt = PROMPT_TEMPLATE_LLAMA.format(
            number=len(sample[key]["agents"]),
            history=sample[key]["text"],
            current=current
        )
    return prompt

def build_ag_prompt_llama(sample: Dict, add_history_reasoning: bool, add_current_reasoning: bool, revert_test_identity: int = 0, peer_tag: str = "", decouple_belief: bool = False, data_type: str = "adv"):
    key = "history_with_reason" if add_history_reasoning else "history"
    if peer_tag != "":
        key = key + "_" + peer_tag
    assert data_type in ["adv", "nat"], "data_type must be 'adv' or 'nat'"
    build_example_fn = build_example_natural if data_type == "nat" else build_example_adv
    current = build_example_fn(sample, key, add_current_reasoning, revert_test_identity, peer_tag, decouple_belief)
    prompt = RAW_PROMPT_TEMPLATE_LLAMA.format(current=current)
    return prompt

def build_stage1_prompt_llama(sample: Dict, add_history_reasoning: bool, peer_tag: str = "", prompt_tag: str = ""):
    key = "history_with_reason" if add_history_reasoning else "history"
    if peer_tag != "":
        key = key + "_" + peer_tag
    template = PROMPT_TEMPLATE_STAGE1_JUDGE_PEER_LLAMA if "JP" in prompt_tag else PROMPT_TEMPLATE_STAGE1_LLAMA
    prompt = template.format(
        number=len(sample[key]["agents"]),
        history=sample[key]["text"]
    )
    return prompt


def build_stage2_prompt_llama(sample: Dict, add_history_reasoning: bool, add_current_reasoning: bool, revert_test_identity: int = 0, peer_tag: str = "", decouple_belief: bool = False, data_type: str = "adv"):
    key = "history_with_reason" if add_history_reasoning else "history"
    if peer_tag != "":
        key = key + "_" + peer_tag
    assert data_type in ["adv", "nat"], "data_type must be 'adv' or 'nat'"
    build_example_fn = build_example_natural if data_type == "nat" else build_example_adv
    current = build_example_fn(sample, key, add_current_reasoning, revert_test_identity, peer_tag, decouple_belief)
    prompt = PROMPT_TEMPLATE_STAGE2_LLAMA.replace("{current}", current)
    prompt = prompt.replace("{number}", str(len(sample[key]["agents"])))
    return prompt


# Qwen3-specific build functions
def build_raw_prompt_qwen3(sample: Dict):
    prompt = RAW_PROMPT_TEMPLATE_QWEN3_STRONG_FORMAT.format(current=sample['formatted_question'])
    return prompt

def build_complete_prompt_qwen3(sample: Dict, add_history_reasoning: bool, add_current_reasoning: bool, revert_test_identity: int = 0, peer_tag: str = "", prompt_tag: str = "", decouple_belief: bool = False, data_type: str = "adv"):
    key = "history_with_reason" if add_history_reasoning else "history"
    if peer_tag != "":
        key = key + "_" + peer_tag
    assert data_type in ["adv", "nat"], "data_type must be 'adv' or 'nat'"
    build_example_fn = build_example_natural if data_type == "nat" else build_example_adv
    current = build_example_fn(sample, key, add_current_reasoning, revert_test_identity, peer_tag, decouple_belief)
    if "AG" in prompt_tag:
        prompt = RAW_PROMPT_TEMPLATE_QWEN3_STRONG_FORMAT.format(current=current)
    else:
        prompt = PROMPT_TEMPLATE_QWEN3.format(
            number=len(sample[key]["agents"]),
            history=sample[key]["text"],
            current=current
        )
    return prompt

def build_ag_prompt_qwen3(sample: Dict, add_history_reasoning: bool, add_current_reasoning: bool, revert_test_identity: int = 0, peer_tag: str = "", decouple_belief: bool = False, data_type: str = "adv"):
    key = "history_with_reason" if add_history_reasoning else "history"
    if peer_tag != "":
        key = key + "_" + peer_tag
    assert data_type in ["adv", "nat"], "data_type must be 'adv' or 'nat'"
    build_example_fn = build_example_natural if data_type == "nat" else build_example_adv
    current = build_example_fn(sample, key, add_current_reasoning, revert_test_identity, peer_tag, decouple_belief)
    prompt = RAW_PROMPT_TEMPLATE_QWEN3_STRONG_FORMAT.format(current=current)
    return prompt

def build_stage1_prompt_qwen3(sample: Dict, add_history_reasoning: bool, peer_tag: str = "", prompt_tag: str = ""):
    key = "history_with_reason" if add_history_reasoning else "history"
    if peer_tag != "":
        key = key + "_" + peer_tag
    template = PROMPT_TEMPLATE_STAGE1_JUDGE_PEER_QWEN3 if "JP" in prompt_tag else PROMPT_TEMPLATE_STAGE1_QWEN3
    prompt = template.format(
        number=len(sample[key]["agents"]),
        history=sample[key]["text"]
    )
    return prompt


def build_stage2_prompt_qwen3(sample: Dict, add_history_reasoning: bool, add_current_reasoning: bool, revert_test_identity: int = 0, peer_tag: str = "", decouple_belief: bool = False, data_type: str = "adv"):
    key = "history_with_reason" if add_history_reasoning else "history"
    if peer_tag != "":
        key = key + "_" + peer_tag
    assert data_type in ["adv", "nat"], "data_type must be 'adv' or 'nat'"
    build_example_fn = build_example_natural if data_type == "nat" else build_example_adv
    current = build_example_fn(sample, key, add_current_reasoning, revert_test_identity, peer_tag, decouple_belief)
    prompt = PROMPT_TEMPLATE_STAGE2_QWEN3.replace("{current}", current)
    prompt = prompt.replace("{number}", str(len(sample[key]["agents"])))
    return prompt


# DS (DeepSeek)-specific build functions
def build_raw_prompt_ds(sample: Dict):
    prompt = RAW_PROMPT_TEMPLATE_DS.format(current=sample['formatted_question'])
    return prompt

def build_complete_prompt_ds(sample: Dict, add_history_reasoning: bool, add_current_reasoning: bool, revert_test_identity: int = 0, peer_tag: str = "", prompt_tag: str = "", decouple_belief: bool = False, data_type: str = "adv"):
    key = "history_with_reason" if add_history_reasoning else "history"
    if peer_tag != "":
        key = key + "_" + peer_tag
    assert data_type in ["adv", "nat"], "data_type must be 'adv' or 'nat'"
    build_example_fn = build_example_natural if data_type == "nat" else build_example_adv
    current = build_example_fn(sample, key, add_current_reasoning, revert_test_identity, peer_tag, decouple_belief)
    if "AG" in prompt_tag:
        prompt = RAW_PROMPT_TEMPLATE_DS.format(current=current)
    else:
        prompt = PROMPT_TEMPLATE_DS.format(
            number=len(sample[key]["agents"]),
            history=sample[key]["text"],
            current=current
        )
    return prompt

def build_ag_prompt_ds(sample: Dict, add_history_reasoning: bool, add_current_reasoning: bool, revert_test_identity: int = 0, peer_tag: str = "", decouple_belief: bool = False, data_type: str = "adv"):
    key = "history_with_reason" if add_history_reasoning else "history"
    if peer_tag != "":
        key = key + "_" + peer_tag
    assert data_type in ["adv", "nat"], "data_type must be 'adv' or 'nat'"
    build_example_fn = build_example_natural if data_type == "nat" else build_example_adv
    current = build_example_fn(sample, key, add_current_reasoning, revert_test_identity, peer_tag, decouple_belief)
    prompt = RAW_PROMPT_TEMPLATE_DS.format(current=current)
    return prompt

def build_stage1_prompt_ds(sample: Dict, add_history_reasoning: bool, peer_tag: str = "", prompt_tag: str = ""):
    key = "history_with_reason" if add_history_reasoning else "history"
    if peer_tag != "":
        key = key + "_" + peer_tag
    template = PROMPT_TEMPLATE_STAGE1_JUDGE_PEER_DS if "JP" in prompt_tag else PROMPT_TEMPLATE_STAGE1_DS
    prompt = template.format(
        number=len(sample[key]["agents"]),
        history=sample[key]["text"]
    )
    return prompt

def build_stage2_prompt_ds(sample: Dict, add_history_reasoning: bool, add_current_reasoning: bool, revert_test_identity: int = 0, peer_tag: str = "", decouple_belief: bool = False, data_type: str = "adv"):
    key = "history_with_reason" if add_history_reasoning else "history"
    if peer_tag != "":
        key = key + "_" + peer_tag
    assert data_type in ["adv", "nat"], "data_type must be 'adv' or 'nat'"
    build_example_fn = build_example_natural if data_type == "nat" else build_example_adv
    current = build_example_fn(sample, key, add_current_reasoning, revert_test_identity, peer_tag, decouple_belief)
    prompt = PROMPT_TEMPLATE_STAGE2_DS.replace("{current}", current)
    prompt = prompt.replace("{number}", str(len(sample[key]["agents"])))
    return prompt


def extract_answer(model_answer: str) -> str:
    """Extract the answer from the model's response."""
    try:
        tmp = model_answer.split('is: "(')
        if len(tmp) == 1:
            tmp = model_answer.split('is: \'(')
        if len(tmp) == 1:
            tmp = model_answer.split('is: `(')
        if len(tmp) == 1:
            tmp = model_answer.split('is: (')
        if len(tmp) == 1:
            tmp = model_answer.split('is (')
        if len(tmp) == 1:
            tmp = model_answer.split('is: "$(')
        if len(tmp) == 1:
            tmp = model_answer.split('is: $(')
        assert len(tmp) > 1, f"didn't output trigger: {model_answer}"
        assert tmp[-1][1] == ')', f"didn't output letter for choice: {model_answer}"
        pred = tmp[-1][0]
        return pred
    except Exception:
        return traceback.format_exc()


def extract_answer_llama(model_answer: str) -> str:
    """Extract the answer from the model's response for Llama format with #### Answer separator."""
    try:
        # Look for #### Answer section
        if '#### Answer' in model_answer:
            # Split by #### Answer and get everything after it
            parts = model_answer.split('#### Answer')
            answer_part = parts[-1].strip()
        else:
            return traceback.format_exc()
        
        # Extract option letter using the same logic as extract_answer
        tmp = answer_part.split('is: "(')
        if len(tmp) == 1:
            tmp = answer_part.split('is: \'(')
        if len(tmp) == 1:
            tmp = answer_part.split('is: `(')
        if len(tmp) == 1:
            tmp = answer_part.split('is: (')
        if len(tmp) == 1:
            tmp = answer_part.split('is (')
        if len(tmp) == 1:
            tmp = answer_part.split('is: "$(')
        if len(tmp) == 1:
            tmp = answer_part.split('is: $(')
        assert len(tmp) > 1, f"didn't output trigger: {answer_part}"
        assert tmp[-1][1] == ')', f"didn't output letter for choice: {answer_part}"
        pred = tmp[-1][0]
        return pred
    except Exception:
        return traceback.format_exc()
    
def extract_answer_qwen3(model_answer: str) -> str:
    """Extract the answer from the model's response for Qwen3 format without tags."""
    try:
        # Look for #### Answer section
        if '[/think]' in model_answer:
            # Split by #### Answer and get everything after it
            parts = model_answer.split('[/think]')
            answer_part = parts[-1].strip()
        else:
            return traceback.format_exc()
        
        # Extract option letter using the same logic as extract_answer
        tmp = answer_part.split('is: "(')
        if len(tmp) == 1:
            tmp = answer_part.split('is: \'(')
        if len(tmp) == 1:
            tmp = answer_part.split('is: `(')
        if len(tmp) == 1:
            tmp = answer_part.split('is: (')
        if len(tmp) == 1:
            tmp = answer_part.split('is (')
        if len(tmp) == 1:
            tmp = answer_part.split('is: "$(')
        if len(tmp) == 1:
            tmp = answer_part.split('is: $(')
        assert len(tmp) > 1, f"didn't output trigger: {answer_part}"
        assert tmp[-1][1] == ')', f"didn't output letter for choice: {answer_part}"
        pred = tmp[-1][0]
        return pred
    except Exception:
        return traceback.format_exc()


def extract_answer_ds(model_answer: str) -> str:
    """Extract the answer from the model's response for DS format without tags."""
    try:
        # Look for </think> tag
        if '</think>' in model_answer:
            # Split by </think> and get everything after it
            parts = model_answer.split('</think>')
            answer_part = parts[-1].strip()
        else:
            answer_part = model_answer
        
        # Extract option letter using the same logic as extract_answer
        tmp = answer_part.split('is: "(')
        if len(tmp) == 1:
            tmp = answer_part.split('is: \'(')
        if len(tmp) == 1:
            tmp = answer_part.split('is: `(')
        if len(tmp) == 1:
            tmp = answer_part.split('is: (')
        if len(tmp) == 1:
            tmp = answer_part.split('is (')
        if len(tmp) == 1:
            tmp = answer_part.split('is: "$(')
        if len(tmp) == 1:
            tmp = answer_part.split('is: $(')
        # GPT-5.2 compatibility
        if len(tmp) == 1:
            tmp = model_answer.split('is: **"(')
        if len(tmp) == 1:
            tmp = model_answer.split('is: **(')
        assert len(tmp) > 1, f"didn't output trigger: {answer_part}"
        assert tmp[-1][1] == ')', f"didn't output letter for choice: {answer_part}"
        pred = tmp[-1][0]
        return pred
    except Exception:
        return traceback.format_exc()
    
    
def get_option_content(pred: str, wrong_options: List[str], gt_option: str) -> str:
    """Get the content of the option."""
    for option in wrong_options:
        if option.startswith(f"({pred})"):
            return option
            
    if gt_option.startswith(f"({pred})"):
        return gt_option
        
    print(f"Cannot find corresponding option for prediction '{pred}'")
    return "Error"


def load_eval_data(file_path: str) -> Dict:
    """Loads data from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading {file_path}: {e}")
        return None


def calculate_accuracies(data: Dict) -> float:
    """Calculates overall accuracy."""
    correct_list = data.get("is_correct", [])
    overall_acc = np.mean(correct_list) if correct_list else 0
    
    return overall_acc


class EvalConfig:
    """Evaluation configuration."""
    def __init__(self, model: str, mode: List, tag: str, current_reason: int = None, history_reason: int = None, revert_test_identity: int = None, tag_peer: str = None, tag_prompt: str = None, decouple_belief: int = None, **kwargs):
        self.model = model
        self.mode = mode
        self.tag = tag
        if current_reason is not None:
            self.CR = current_reason
        if history_reason is not None:
            self.HR = history_reason
        if revert_test_identity is not None:
            self.RTI = revert_test_identity
        if tag_peer is not None:
            self.TAG_PEER = tag_peer
        if tag_prompt is not None:
            self.TAG_PROMPT = tag_prompt
        if decouple_belief is not None:
            self.DB = decouple_belief

        self.fname = str(self)+'.json'
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        """String representation of the evaluation configuration."""
        base_str =  "_".join(self.model.split('/')[1:]) + "-" + self.mode + "-" + self.tag
        for k, v in sorted(self.__dict__.items()):
            if k == "fname" or k == "model" or k == "mode" or k == "tag":
                continue
            base_str = base_str + "_" + k.replace("_", "") + "_" + str(v).replace("-", "").replace('.json','')
        base_str = base_str.replace("_RTI_0", "")
        base_str = base_str.replace("_TAGPEER_4_1", "")
        base_str = base_str.replace("_TAGPROMPT_NS", "")
        base_str = base_str.replace("_DB_0", "")
        return base_str
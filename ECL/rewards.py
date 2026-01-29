# Utility functions to compute Exact Match and Citation Scores to reward
import re
import traceback

PATTERN = r"^<think>(.*?)</think>\n\n<answer>(.*?)</answer>\s*\Z"
SUMMARY_PATTERN = r"^<think>(.*?)</think>\n\n<summary>(.*?)</summary>\s*\Z"

# Llama-specific patterns with #### Reasoning and #### Answer/Summary separator
PATTERN_LLAMA = r"^.*?#### Reasoning\s+(.*?)#### Answer\s+(.*?)\s*\Z"
SUMMARY_PATTERN_LLAMA = r"^.*?#### Reasoning\s+(.*?)#### Summary\s+(.*?)\s*\Z"

# Qwen3-specific patterns - no tags for answer/summary, just [/think]\n\n followed by content
# Must NOT have <answer> or <summary> tags after [/think]
PATTERN_QWEN3 = r"^\[think\](.*?)\[/think\]\n\n(?!<answer>|<summary>)(.*?)\s*\Z"
SUMMARY_PATTERN_QWEN3 = r"^\[think\](.*?)\[/think\]\n\n(?!<answer>|<summary>)(.*?)\s*\Z"

# DS (DeepSeek)-specific patterns - uses <think></think> tags but no tags for answer/summary
# Similar to Qwen3 but with angle brackets instead of square brackets
PATTERN_DS = r"^<think>(.*?)</think>\n\n(?!<answer>|<summary>)(.*?)\s*\Z"
SUMMARY_PATTERN_DS = r"^<think>(.*?)</think>\n\n(?!<answer>|<summary>)(.*?)\s*\Z"

def is_conversational(messages):
    if isinstance(messages, list):
        message = messages[0]
        # Each message must a list of dictionaries with keys "role" and "content"
        if isinstance(message, dict) and "role" in message and "content" in message:
            return True
    return False

def get_option_char(model_answer):
    try:
        tmp=model_answer.split('is: "(')
        if len(tmp) == 1:
            tmp = model_answer.split('is: (')
        if len(tmp) == 1:
            tmp = model_answer.split('is (')
        assert len(tmp) > 1, "model didn't output trigger"
        assert tmp[-1][1] == ')', "didnt output letter for choice"
        pred = tmp[-1][0]
        return pred
    except Exception as e:
        return traceback.format_exc()


def format_reward_normal(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    if is_conversational(completions[0]):
        completions = [completion[0]["content"] for completion in completions]
    # pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    matches = [re.match(PATTERN, content, re.DOTALL | re.MULTILINE) for content in completions]
    rewards_list = [1.0 if match else 0.0 for match in matches]
    return rewards_list


def summary_format_reward_normal(completions, **kwargs):
    """Reward function that checks if the completion has a specific summary format."""
    if is_conversational(completions[0]):
        completions = [completion[0]["content"] for completion in completions]
    matches = [re.match(SUMMARY_PATTERN, content, re.DOTALL | re.MULTILINE) for content in completions]
    rewards_list = [1.0 if match else 0.0 for match in matches]
    return rewards_list


# Llama-specific format reward functions
def format_reward_llama(completions, **kwargs):
    """Reward function that checks if the completion has #### separator format for Llama models."""
    if is_conversational(completions[0]):
        completions = [completion[0]["content"] for completion in completions]
    matches = [re.match(PATTERN_LLAMA, content, re.DOTALL | re.MULTILINE) for content in completions]
    rewards_list = [1.0 if match else 0.0 for match in matches]
    return rewards_list


def summary_format_reward_llama(completions, **kwargs):
    """Reward function that checks if the completion has #### separator format for Llama models (stage 1)."""
    if is_conversational(completions[0]):
        completions = [completion[0]["content"] for completion in completions]
    matches = [re.match(SUMMARY_PATTERN_LLAMA, content, re.DOTALL | re.MULTILINE) for content in completions]
    rewards_list = [1.0 if match else 0.0 for match in matches]
    return rewards_list


# Qwen3-specific format reward functions
def format_reward_qwen3(completions, **kwargs):
    """Reward function that checks if the completion has <think></think> format with answer directly after for Qwen3 models."""
    if is_conversational(completions[0]):
        completions = [completion[0]["content"] for completion in completions]
    matches = [re.match(PATTERN_QWEN3, content, re.DOTALL | re.MULTILINE) for content in completions]
    rewards_list = [1.0 if match else 0.0 for match in matches]
    return rewards_list


def summary_format_reward_qwen3(completions, **kwargs):
    """Reward function that checks if the completion has <think></think> format with summary directly after for Qwen3 models (stage 1)."""
    if is_conversational(completions[0]):
        completions = [completion[0]["content"] for completion in completions]
    matches = [re.match(SUMMARY_PATTERN_QWEN3, content, re.DOTALL | re.MULTILINE) for content in completions]
    rewards_list = [1.0 if match else 0.0 for match in matches]
    return rewards_list


# DS (DeepSeek)-specific format reward functions
def format_reward_ds(completions, **kwargs):
    """Reward function that checks if the completion has <think></think> format with answer directly after for DS models."""
    if is_conversational(completions[0]):
        completions = [completion[0]["content"] for completion in completions]
    matches = [re.match(PATTERN_DS, content, re.DOTALL | re.MULTILINE) for content in completions]
    rewards_list = [1.0 if match else 0.0 for match in matches]
    return rewards_list


def summary_format_reward_ds(completions, **kwargs):
    """Reward function that checks if the completion has <think></think> format with summary directly after for DS models (stage 1)."""
    if is_conversational(completions[0]):
        completions = [completion[0]["content"] for completion in completions]
    matches = [re.match(SUMMARY_PATTERN_DS, content, re.DOTALL | re.MULTILINE) for content in completions]
    rewards_list = [1.0 if match else 0.0 for match in matches]
    return rewards_list


def accuracy_reward_normal(completions, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    if is_conversational(completions[0]):
        completions = [completion[0]["content"] for completion in completions]
    rewards = []
    answers = [gt_option.strip() for gt_option in kwargs['gt_option']]
    
    for content, answer in zip(completions, answers):
        
        answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
        answer_parsed = answer_match.group(1).strip() if answer_match else ""
        pred_answer_option = get_option_char(answer_parsed) 
        
        if answer[1] == pred_answer_option:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
            
    return rewards


def accuracy_reward_llama(completions, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth for Llama models."""
    if is_conversational(completions[0]):
        completions = [completion[0]["content"] for completion in completions]
    rewards = []
    answers = [gt_option.strip() for gt_option in kwargs['gt_option']]
    
    for content, answer in zip(completions, answers):
        # Look for #### Answer section
        if '#### Answer' in content:
            parts = content.split('#### Answer')
            answer_parsed = parts[-1].strip()
        else:
            answer_parsed = ""
        
        pred_answer_option = get_option_char(answer_parsed) 
        
        if answer[1] == pred_answer_option:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
            
    return rewards


def accuracy_reward_qwen3(completions, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth for Qwen3 models."""
    if is_conversational(completions[0]):
        completions = [completion[0]["content"] for completion in completions]
    rewards = []
    answers = [gt_option.strip() for gt_option in kwargs['gt_option']]
    
    for content, answer in zip(completions, answers):
        # Extract content after [/think]\n\n (no <answer> tags for Qwen3)
        answer_match = re.search(r'\[/think\]\n\n(.*?)\s*\Z', content, re.DOTALL)
        answer_parsed = answer_match.group(1).strip() if answer_match else ""
        pred_answer_option = get_option_char(answer_parsed) 
        
        if answer[1] == pred_answer_option:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
            
    return rewards


def accuracy_reward_ds(completions, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth for DS models."""
    if is_conversational(completions[0]):
        completions = [completion[0]["content"] for completion in completions]
    rewards = []
    answers = [gt_option.strip() for gt_option in kwargs['gt_option']]
    
    for content, answer in zip(completions, answers):
        # Extract content after </think>\n\n (no <answer> tags for DS)
        answer_match = re.search(r'</think>\n\n(.*?)\s*\Z', content, re.DOTALL)
        answer_parsed = answer_match.group(1).strip() if answer_match else ""
        pred_answer_option = get_option_char(answer_parsed) 
        
        if answer[1] == pred_answer_option:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
            
    return rewards
    
def get_agent_name(model_answer):
    try:
        tmp=model_answer.split("Agent: ")
        if len(tmp) == 1:
            tmp = model_answer.split("agent: ")
        assert len(tmp) > 1, "model didn't output trigger"
        agent_name = tmp[-1].strip().split()[0]
        return agent_name
    except Exception as e:
        return traceback.format_exc()

def get_peer_recognition_reward_func(add_history_reasoning: bool, tag_peer: str):
    def peer_recognition_reward_normal(completions, **kwargs):
        key = 'history_with_reason' if add_history_reasoning else 'history'
        key = key + "_" + tag_peer if tag_peer != "" else key
        peer_names = [history_dict['agents'][history_dict['reliable_agent']].strip() for history_dict in kwargs[key]]
        
        if is_conversational(completions[0]):
            completions = [completion[0]["content"] for completion in completions]
        
        rewards = []
        for content, peer_name in zip(completions, peer_names):
            answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
            answer_parsed = answer_match.group(1).strip() if answer_match else ""
            pred_agent_name = get_agent_name(answer_parsed)

            if peer_name == pred_agent_name:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        return rewards
    return peer_recognition_reward_normal


def get_peer_recognition_reward_func_llama(add_history_reasoning: bool, tag_peer: str):
    """Get peer recognition reward function for Llama models with #### Answer separator."""
    def peer_recognition_reward_llama(completions, **kwargs):
        key = 'history_with_reason' if add_history_reasoning else 'history'
        key = key + "_" + tag_peer if tag_peer != "" else key
        peer_names = [history_dict['agents'][history_dict['reliable_agent']].strip() for history_dict in kwargs[key]]
        
        if is_conversational(completions[0]):
            completions = [completion[0]["content"] for completion in completions]
        
        rewards = []
        for content, peer_name in zip(completions, peer_names):
            # Look for #### Answer section
            if '#### Answer' in content:
                parts = content.split('#### Answer')
                answer_parsed = parts[-1].strip()
            else:
                answer_parsed = ""
            
            pred_agent_name = get_agent_name(answer_parsed)

            if peer_name == pred_agent_name:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        return rewards
    return peer_recognition_reward_llama


def get_peer_recognition_reward_func_stage1(add_history_reasoning: bool, tag_peer: str):
    """
    Stage 1 peer recognition reward function.
    Rewards model for correctly identifying the most trustworthy agent.
    Total reward = 1.0 if the predicted agent matches the reliable agent, 0.0 otherwise.
    
    Expected format in <summary>:
    The most trustworthy agent is: AGENT_NAME
    
    Args:
        add_history_reasoning: Whether to use history_with_reason key
        tag_peer: Peer tag string
    """
    def peer_recognition_reward_stage1(completions, **kwargs):
        key = 'history_with_reason' if add_history_reasoning else 'history'
        key = key + "_" + tag_peer if tag_peer != "" else key
        
        # Get ground truth: the reliable agent name
        history_dicts = kwargs[key]
        
        if is_conversational(completions[0]):
            completions = [completion[0]["content"] for completion in completions]
        
        rewards = []
        for content, history_dict in zip(completions, history_dicts):
            # Extract summary section
            summary_match = re.search(r'<summary>(.*?)</summary>', content, re.DOTALL)
            if not summary_match:
                rewards.append(0.0)
                continue
                
            summary_text = summary_match.group(1).strip()
            
            # Get all agent names and the reliable agent
            all_agents = history_dict['agents']
            reliable_agent_idx = history_dict['reliable_agent']
            reliable_agent_name = all_agents[reliable_agent_idx].strip()
            
            # Parse the summary to extract the predicted trustworthy agent
            # Expected format: "The most trustworthy agent is: AGENT_NAME"
            match = re.search(r'(?:The\s+most\s+trustworthy\s+agent\s+is|most\s+trustworthy\s+agent):\s*(.+?)(?:\n|$)', summary_text, re.IGNORECASE)
            
            if not match:
                rewards.append(0.0)
                continue
            
            pred_agent_name = match.group(1).strip()
            
            # Check if the predicted agent name matches the reliable agent
            # Support partial matching for robustness
            if pred_agent_name == reliable_agent_name or reliable_agent_name in pred_agent_name or pred_agent_name in reliable_agent_name:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        
        return rewards
    
    return peer_recognition_reward_stage1


def get_peer_recognition_reward_func_stage1_llama(add_history_reasoning: bool, tag_peer: str):
    """
    Stage 1 peer recognition reward function for Llama models with #### Summary separator.
    Rewards model for correctly identifying the most trustworthy agent.
    Total reward = 1.0 if the predicted agent matches the reliable agent, 0.0 otherwise.
    
    Expected format with #### Summary:
    #### Summary
    The most trustworthy agent is: AGENT_NAME
    
    Args:
        add_history_reasoning: Whether to use history_with_reason key
        tag_peer: Peer tag string
    """
    def peer_recognition_reward_stage1_llama(completions, **kwargs):
        key = 'history_with_reason' if add_history_reasoning else 'history'
        key = key + "_" + tag_peer if tag_peer != "" else key
        
        # Get ground truth: the reliable agent name
        history_dicts = kwargs[key]
        
        if is_conversational(completions[0]):
            completions = [completion[0]["content"] for completion in completions]
        
        rewards = []
        for content, history_dict in zip(completions, history_dicts):
            # Extract summary section for Llama format (after #### Summary)
            if '#### Summary' not in content:
                rewards.append(0.0)
                continue
            
            # Split by #### Summary and get everything after it
            parts = content.split('#### Summary')
            summary_text = parts[-1].strip()
            
            # Get all agent names and the reliable agent
            all_agents = history_dict['agents']
            reliable_agent_idx = history_dict['reliable_agent']
            reliable_agent_name = all_agents[reliable_agent_idx].strip()
            
            # Parse the summary to extract the predicted trustworthy agent
            # Expected format: "The most trustworthy agent is: AGENT_NAME"
            match = re.search(r'(?:The\s+most\s+trustworthy\s+agent\s+is|most\s+trustworthy\s+agent):\s*(.+?)(?:\n|$)', summary_text, re.IGNORECASE)
            
            if not match:
                rewards.append(0.0)
                continue
            
            pred_agent_name = match.group(1).strip()
            
            # Check if the predicted agent name matches the reliable agent
            # Support partial matching for robustness
            if pred_agent_name == reliable_agent_name or reliable_agent_name in pred_agent_name or pred_agent_name in reliable_agent_name:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        
        return rewards
    
    return peer_recognition_reward_stage1_llama


def get_peer_recognition_reward_func_stage1_qwen3(add_history_reasoning: bool, tag_peer: str):
    """
    Stage 1 peer recognition reward function for Qwen3 models.
    Rewards model for correctly identifying the most trustworthy agent.
    Total reward = 1.0 if the predicted agent matches the reliable agent, 0.0 otherwise.
    
    Expected format for Qwen3 (no summary tags):
    [think]
    reasoning...
    [/think]
    
    The most trustworthy agent is: AGENT_NAME
    
    Args:
        add_history_reasoning: Whether to use history_with_reason key
        tag_peer: Peer tag string
    """
    def peer_recognition_reward_stage1_qwen3(completions, **kwargs):
        key = 'history_with_reason' if add_history_reasoning else 'history'
        key = key + "_" + tag_peer if tag_peer != "" else key
        
        # Get ground truth: the reliable agent name
        history_dicts = kwargs[key]
        
        if is_conversational(completions[0]):
            completions = [completion[0]["content"] for completion in completions]
        
        rewards = []
        for content, history_dict in zip(completions, history_dicts):
            # Extract content after [/think]\n\n (no <summary> tags for Qwen3)
            summary_match = re.search(r'\[/think\]\n\n(.*?)\s*\Z', content, re.DOTALL)
            if not summary_match:
                rewards.append(0.0)
                continue
                
            summary_text = summary_match.group(1).strip()
            
            # Get all agent names and the reliable agent
            all_agents = history_dict['agents']
            reliable_agent_idx = history_dict['reliable_agent']
            try:
                reliable_agent_name = all_agents[reliable_agent_idx].strip()
            except:
                reliable_agent_name = all_agents[reliable_agent_idx[0]].strip()
            
            # Parse the summary to extract the predicted trustworthy agent
            # Expected format: "The most trustworthy agent is: AGENT_NAME"
            match = re.search(r'(?:The\s+most\s+trustworthy\s+agent\s+is|most\s+trustworthy\s+agent):\s*(.+?)(?:\n|$)', summary_text, re.IGNORECASE)
            
            if not match:
                rewards.append(0.0)
                continue
            
            pred_agent_name = match.group(1).strip()
            
            # Check if the predicted agent name matches the reliable agent
            # Support partial matching for robustness
            if pred_agent_name == reliable_agent_name or reliable_agent_name in pred_agent_name or pred_agent_name in reliable_agent_name:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        
        return rewards
    
    return peer_recognition_reward_stage1_qwen3


def get_peer_recognition_reward_func_stage1_ds(add_history_reasoning: bool, tag_peer: str):
    """
    Stage 1 peer recognition reward function for DS models.
    Rewards model for correctly identifying the most trustworthy agent.
    Total reward = 1.0 if the predicted agent matches the reliable agent, 0.0 otherwise.
    
    Expected format for DS (no summary tags):
    <think>
    reasoning...
    </think>
    
    The most trustworthy agent is: AGENT_NAME
    
    Args:
        add_history_reasoning: Whether to use history_with_reason key
        tag_peer: Peer tag string
    """
    def peer_recognition_reward_stage1_ds(completions, **kwargs):
        key = 'history_with_reason' if add_history_reasoning else 'history'
        key = key + "_" + tag_peer if tag_peer != "" else key
        
        # Get ground truth: the reliable agent name
        history_dicts = kwargs[key]
        
        if is_conversational(completions[0]):
            completions = [completion[0]["content"] for completion in completions]
        
        rewards = []
        for content, history_dict in zip(completions, history_dicts):
            # Extract content after </think>\n\n (no <summary> tags for DS)
            summary_match = re.search(r'</think>\n\n(.*?)\s*\Z', content, re.DOTALL)
            if not summary_match:
                rewards.append(0.0)
                continue
                
            summary_text = summary_match.group(1).strip()
            
            # Get all agent names and the reliable agent
            all_agents = history_dict['agents']
            reliable_agent_idx = history_dict['reliable_agent']
            try:
                reliable_agent_name = all_agents[reliable_agent_idx].strip()
            except:
                reliable_agent_name = all_agents[reliable_agent_idx[0]].strip()
            
            # Parse the summary to extract the predicted trustworthy agent
            # Expected format: "The most trustworthy agent is: AGENT_NAME"
            match = re.search(r'(?:The\s+most\s+trustworthy\s+agent\s+is|most\s+trustworthy\s+agent):\s*(.+?)(?:\n|$)', summary_text, re.IGNORECASE)
            
            if not match:
                rewards.append(0.0)
                continue
            
            pred_agent_name = match.group(1).strip()
            
            # Check if the predicted agent name matches the reliable agent
            # Support partial matching for robustness
            if pred_agent_name == reliable_agent_name or reliable_agent_name in pred_agent_name or pred_agent_name in reliable_agent_name:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        
        return rewards
    
    return peer_recognition_reward_stage1_ds

if __name__ == "__main__":
    output = """<think>\nxxxxxxx\n</think>\n\n<answer>\nYou: The best answer is: "(B) the content of the answer"\n</answer>"""
    
    print(format_reward_normal([output]))
    print(accuracy_reward_normal([output], gt_option=["(A) 730 cm cube"]))
    
    # Test get_peer_recognition_reward_func_stage1
    print("\n=== Testing get_peer_recognition_reward_func_stage1 ===")
    test_completion_correct = """<think>
Looking at the history, I can see:
- Mary answered all 5 questions correctly
- John got none correct
- George got none correct
Mary is clearly the most trustworthy.
</think>

<summary>
The most trustworthy agent is: Mary
</summary>"""
    
    test_completion_wrong = """<think>
Looking at the history...
I think John is the most trustworthy.
</think>

<summary>
The most trustworthy agent is: John
</summary>"""
    
    test_completion_no_match = """<think>
Wrong analysis...
</think>

<summary>
The most reliable agent is Tom.
</summary>"""
    
    test_history = {
        'agents': ['Mary', 'John', 'George'],
        'reliable_agent': 0  # Mary is 100% correct
    }
    
    reward_func = get_peer_recognition_reward_func_stage1(False, "")
    
    # Test correct recognition - should get 1.0
    rewards = reward_func([test_completion_correct], history=[test_history])
    print(f"Correct recognition reward: {rewards[0]} (expected: 1.0)")
    
    # Test wrong recognition - should get 0.0
    rewards = reward_func([test_completion_wrong], history=[test_history])
    print(f"Wrong recognition reward: {rewards[0]} (expected: 0.0)")
    
    # Test no match - should get 0.0
    rewards = reward_func([test_completion_no_match], history=[test_history])
    print(f"No match reward: {rewards[0]} (expected: 0.0)")
    
    # Test get_peer_recognition_reward_func_stage1_llama
    print("\n=== Testing get_peer_recognition_reward_func_stage1_llama ===")
    test_completion_correct_llama = """#### Reasoning
Looking at the history, I can see:
- Mary answered all 5 questions correctly
- John got none correct
- George got none correct
Mary is clearly the most trustworthy.

#### Summary
The most trustworthy agent is: Mary
"""
    
    test_completion_wrong_llama = """#### Reasoning
Looking at the history...
I think John is the most trustworthy.

#### Summary
The most trustworthy agent is: John
"""
    
    test_completion_no_match_llama = """#### Reasoning
Wrong analysis...

#### Summary
The most reliable agent is Tom.
"""
    
    reward_func_llama = get_peer_recognition_reward_func_stage1_llama(False, "")
    
    # Test correct recognition - should get 1.0
    rewards = reward_func_llama([test_completion_correct_llama], history=[test_history])
    print(f"Correct recognition reward (Llama): {rewards[0]} (expected: 1.0)")
    
    # Test wrong recognition - should get 0.0
    rewards = reward_func_llama([test_completion_wrong_llama], history=[test_history])
    print(f"Wrong recognition reward (Llama): {rewards[0]} (expected: 0.0)")
    
    # Test no match - should get 0.0
    rewards = reward_func_llama([test_completion_no_match_llama], history=[test_history])
    print(f"No match reward (Llama): {rewards[0]} (expected: 0.0)")
    
    # Test get_peer_recognition_reward_func_stage1_qwen3
    print("\n=== Testing get_peer_recognition_reward_func_stage1_qwen3 ===")
    test_completion_correct_qwen3 = """<think>
Looking at the history, I can see:
- Mary answered all 5 questions correctly
- John got none correct
- George got none correct
Mary is clearly the most trustworthy.
</think>

The most trustworthy agent is: Mary
"""
    
    test_completion_wrong_qwen3 = """<think>
Looking at the history...
I think John is the most trustworthy.
</think>

The most trustworthy agent is: John
"""
    
    test_completion_no_match_qwen3 = """<think>
Wrong analysis...
</think>

The most reliable agent is Tom.
"""
    
    reward_func_qwen3 = get_peer_recognition_reward_func_stage1_qwen3(False, "")
    
    # Test correct recognition - should get 1.0
    rewards = reward_func_qwen3([test_completion_correct_qwen3], history=[test_history])
    print(f"Correct recognition reward (Qwen3): {rewards[0]} (expected: 1.0)")
    
    # Test wrong recognition - should get 0.0
    rewards = reward_func_qwen3([test_completion_wrong_qwen3], history=[test_history])
    print(f"Wrong recognition reward (Qwen3): {rewards[0]} (expected: 0.0)")
    
    # Test no match - should get 0.0
    rewards = reward_func_qwen3([test_completion_no_match_qwen3], history=[test_history])
    print(f"No match reward (Qwen3): {rewards[0]} (expected: 0.0)")
    
    # pass
    # Test Extraction
    # Completion = "<think>\n Reasoning \n Trace \n</think>\n\n<answer>\nAnswer\n</answer>"
    # print(Completion)
    # print(format_reward(Completion))
    # print(count_tags(Completion))
    # print(split_reasoning(Completion))

#     Reasoning = """
# iauhbfio augbioah 

# hungry Voice: sidfhos

# Indulgent Voice: ...

# Disciplined Voice: ...

# Hungry Voice: ...
# """
#     voices = get_voices(Reasoning) 
#     print(voices)

#     uniq = get_unique_voices(voices)
#     uniq = group_similar_voices_spacy(voices)
#     print(uniq)

#     print(has_non_consecutive_turn(uniq, voices))


    # # Test Get Unique Voices
    # print(group_similar_voices_spacy([
    # "disciplined",
    # "self-control",
    # "gluttonous",
    # "lazy",
    # "responsible",
    # "indulgent",
    # "productive",
    # "hungry",
    # "hard working"
    # ]))

    # # Test turn checking fn
    # groups = [('A', 'B'), ('C', 'D', 'E')]
    # sequence = ['A', 'B', 'C', 'D', 'E', 'A', 'D', 'E']
    # print(has_non_consecutive_turn(groups, sequence), "Should be True")

    # groups = [('A', 'B'), ('C', 'D'), ('E', 'F')]
    # sequence = ['A', 'C', 'E', 'A'] 
    # print(has_non_consecutive_turn(groups, sequence), "Should be True") 

    # groups = [('A', 'B'), ('C', 'D'), ('E', 'F')]
    # sequence = ['A', 'C', 'A', 'E'] 
    # print(has_non_consecutive_turn(groups, sequence), "Should be True") 

    # groups = [('A', 'B'), ('C', 'D'), ('E', 'F')]
    # sequence = ['A', 'B', 'C', 'D', 'E', 'F']
    # print(has_non_consecutive_turn(groups, sequence), "Should be False") 


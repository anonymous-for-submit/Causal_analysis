# -*- coding: utf-8 -*-
import sys
sys.path.append('/home/zlyuaj/Causal/MetaGPT')
sys.path.append('/home/zlyuaj/Causal/MetaGPT/metagpt')
import asyncio
from pathlib import Path
import os
import json
import argparse
import random
from openai import OpenAI
from metagpt.logs import logger
from evaluate_result import evaluate_one, evaluate_one_codecontest, evaluate_one_MBPP, evaluate_one_CoderEval
from concurrent.futures import as_completed, ProcessPoolExecutor
from _utils import prompt_split_humaneval
from metagpt.actions.intervent import check_one_exist
# 导入原有代码中的函数
from software_causal_basedataset import startup, extract_code_from_repo, extract_plan_from_repo, format_plan

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, required=True,
                    help='Path to original results jsonl file')
parser.add_argument('--rerun', type=int, default=0)
parser.add_argument('--openai_api_key', type=str, default='')
parser.add_argument('--openai_base_url', type=str, default='')
parser.add_argument('--output_path', type=str, default='./output/')
parser.add_argument('--dataset', type=str, default='HumanEval')
parser.add_argument('--output_file_name', type=str, default='test')
parser.add_argument('--workspace', type=str, default='workspace_baseDataset')
parser.add_argument('--num_generate', type=int, default=10)
parser.add_argument('--parallel', type=int, default=1)
parser.add_argument('--model', type=str, default='gpt-35-turbo')
parser.add_argument('--run_generate', type=int, default=1)
parser.add_argument('--run_evaluate', type=int, default=1)
parser.add_argument('--MBPP_test_case_num', type=int, default=1)
parser.add_argument('--eval_start_index', type=int, default=-1)
parser.add_argument('--recover', type=int, default=0)
parser.add_argument('--begin_idx', type=int, default=0)
parser.add_argument('--end_idx', type=int, default=1000)
parser.add_argument('--do_intervent', type=int, default=0)
parser.add_argument('--levels', type=str, default='')
parser.add_argument('--original_path', type=str, default='aaa')


parser.add_argument('--add_monitor', type=int, default=0)
parser.add_argument('--repair_plan', type=int, default=0)
parser.add_argument('--repair_code', type=int, default=0)
parser.add_argument('--run_multi_gen', type=int, default=0)

args = parser.parse_args()

# 初始化OpenAI客户端用于prompt改写
client_4o_mini = OpenAI(
        api_key = "sk-FRQxdGxCMDSPoogN0SgdGGm4IEfv3uMjUFTtgepRNC7bnxO8",  # 此处的key需要自己通过官方购买 或者通过其他渠道获取
        base_url = "https://api.chatanywhere.tech/v1" # 中转地址
        )

client_ds_qwen= OpenAI(api_key= 'EMPTY', base_url= 'http://127.0.0.1:8000/v1/')

def modify_prompt(original_prompt):
    """
    Use GPT-4o-mini to modify the prompt to be similar but different
    Goal: Simulate a new problem that an LLM might incorrectly interpret the original problem as
    """
    system_prompt = """You are a professional requirement modification assistant. Your task is to rewrite the given programming problem requirement and make it expressing the very opposite meaning while keeping the overall structure similar.

Return ONLY the rewritten prompt (including signature if there is in original requirement) with no additional explanations or comments."""

    try:
        if '4o' in args.model:
            client = client_4o_mini
        else:
            client=client_ds_qwen

        model='gpt-4o-mini-ca'
        if  '4o' in  args.model:
            model = 'gpt-4o-mini-ca'
        if 'deepseek' in  args.model:
            model = 'deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct'
        if 'qwen' in args.model:
            model = 'qwen/Qwen2.5-14B-Instruct'  # Replace with the actual model name for qwen
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Please rewrite the following programming problem requirement:\n\n{original_prompt}"}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        modified_prompt = response.choices[0].message.content.strip()
        return modified_prompt
    except Exception as e:
        logger.error(f"Error modifying prompt: {e}")
        return None

import random

def modify_prompt_new(original_prompt):
    """
    Use LLM to modify the prompt to create adversarial examples
    Goal: Create a subtly different problem that LLMs might confuse with the original
    """
    
    # 定义多种错误理解模式
    error_patterns = [
        {
            "name": "opposite_logic",
            "instruction": """Rewrite the problem with OPPOSITE logic/goal while keeping similar wording:
- If finding maximum, change to minimum (but keep word "find the best")
- If sorting ascending, change to descending (but use ambiguous terms)
- If checking existence, change to counting occurrences
- If returning first occurrence, change to last occurrence
- If filtering/keeping elements, change to removing them
Example: "find largest even number" → "find smallest odd number" but phrase as "find the most suitable number from even positions\""""
        },
        {
            "name": "boundary_flip",
            "instruction": """Keep the main task but flip boundary conditions:
- Change inclusive to exclusive (>= to >)
- Change "at least" to "at most"
- Change "up to" to "starting from"
- Swap upper/lower bounds
- Change "all" to "any" or vice versa
Example: "return elements >= 5" → "return elements < 5" but describe as "filter based on threshold 5\""""
        },
        {
            "name": "operation_swap",
            "instruction": """Keep problem structure but swap the core operation:
- Sum → Product
- Concatenate → Split
- Append → Prepend
- Add → Subtract
- Multiply → Divide
- AND → OR (in boolean logic)
Example: "sum all elements" → "multiply all elements" but describe as "combine all elements\""""
        },
        {
            "name": "index_confusion",
            "instruction": """Create confusion about indexing/position:
- Change 0-indexed to 1-indexed (or vice versa)
- Change "first N" to "last N"
- Change "every Nth" to "every N+1th"
- Swap "before" and "after"
- Change "index i" to "value i"
Example: "return first 3 elements" → "return last 3 elements" but say "return the 3 primary elements\""""
        },
        {
            "name": "data_structure_shift",
            "instruction": """Maintain similar problem but change expected data structure handling:
- List to Set (lose order/duplicates)
- String to List of chars
- Dict keys to Dict values
- Nested to Flat (or vice versa)
- Tuple to List (immutable to mutable)
Example: "return list of unique elements in order" → "return set of elements" described as "return collection of elements\""""
        },
        {
            "name": "off_by_one",
            "instruction": """Introduce subtle off-by-one errors:
- Change "length" to "length-1" or "length+1"
- Change range(n) to range(n+1) or range(n-1)
- Change "include endpoint" to "exclude endpoint"
- Shift slice boundaries by 1
Example: "process N items" → "process N-1 items" but describe as "process the items up to N\""""
        },
        {
            "name": "semantic_opposite",
            "instruction": """Keep function name/signature but invert semantic meaning:
- "is_valid" checking for invalid instead
- "remove_duplicates" keeping only duplicates
- "find_missing" finding present elements
- "count_true" counting false
Example: function "is_palindrome" → check if NOT palindrome but keep same name in description"""
        }
    ]
    
    # 随机选择一个错误模式
    pattern = random.choice(error_patterns)
    
    system_prompt = f"""You are creating adversarial test cases for code generation models. 

Your task: Rewrite the programming problem to create a SUBTLY DIFFERENT version that looks similar but requires different logic.

CRITICAL REQUIREMENTS:
1. **Apply this modification pattern**: {pattern['instruction']}

2. **Keep surface similarity**:
   - Maintain similar function signature/name (only change if necessary)
   - Use similar vocabulary and phrasing
   - Keep the same general problem domain
   - Preserve the overall structure

3. **Modify examples consistently**:
   - Change input/output examples to match the NEW (modified) requirement
   - Examples should demonstrate the MODIFIED behavior
   - Keep example format similar to original

4. **Make it subtle**:
   - Avoid explicitly stating "opposite" or "different"
   - Use ambiguous terms that could be misinterpreted
   - The modification should be easy to overlook

5. **Output format**:
   - Return ONLY the rewritten prompt
   - Include modified function signature (if present)
   - Include modified examples
   - NO explanations, NO markdown, NO preamble

The goal is to create a problem that a code model might confuse with the original, leading to incorrect code."""

    user_prompt = f"""Original problem:
{original_prompt}

Rewrite this problem applying the modification pattern. Make it look similar but require different logic."""

    try:
        # 根据模型选择client
        if '4o' in args.model:
            client = client_4o_mini
            model = 'gpt-4o-mini-ca'
        elif 'deepseek' in args.model:
            client = client_ds_qwen
            model = 'deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct'
        elif 'qwen' in args.model:
            client = client_ds_qwen
            model = 'qwen/Qwen2.5-14B-Instruct'
        else:
            client = client_4o_mini
            model = 'gpt-4o-mini-ca'
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.8,  # 提高温度增加多样性
            max_tokens=2000
        )
        
        modified_prompt = response.choices[0].message.content.strip()
        
        # 清理可能的markdown格式
        modified_prompt = modified_prompt.replace('```python', '').replace('```', '').strip()
        
        return modified_prompt
        
    except Exception as e:
        logger.error(f"Error modifying prompt: {e}")
        return None

import random

def modify_prompt_agg(original_prompt):
    """
    Aggressively modify the prompt to create adversarial examples
    Goal: Create problems that are fundamentally different but superficially similar
    """
    
    # 更激进的错误模式，可以组合使用
    aggressive_patterns = [
        {
            "name": "complete_logic_inversion",
            "instruction": """COMPLETELY INVERT the problem logic at multiple levels:
- Reverse the main goal (find→remove, check→ignore, validate→invalidate)
- Flip ALL comparison operators (>, <, ==, !=)
- Invert return conditions (return true cases → return false cases)
- Reverse iteration order (forward→backward, left→right)
- Change success criteria to failure criteria
Example: "return list of valid passwords (length >= 8, has digit)" → "return list of invalid passwords (length < 8 OR no digit)" but phrase as "filter passwords based on requirements: length 8, must contain numbers\""""
        },
        {
            "name": "multi_dimensional_flip",
            "instruction": """Change multiple aspects simultaneously:
- Swap operation AND data structure (sum list → multiply set elements)
- Change direction AND boundary (ascending order, include duplicates → descending order, exclude duplicates)
- Invert logic AND shift index (first even number → last odd number, 0-indexed → 1-indexed)
- Change type AND operation (string concatenation → list intersection)
Example: "find first 5 even numbers in ascending order" → "find last 3 odd numbers in descending order, 1-indexed" described as "locate the primary 3 numbers with specific properties\""""
        },
        {
            "name": "core_algorithm_replacement",
            "instruction": """Replace the fundamental algorithm while keeping description vague:
- Binary search → Linear search (or vice versa)
- Dynamic programming → Greedy algorithm
- DFS → BFS (or vice versa)
- Iterative → Recursive (with different base case)
- Sort-based solution → Hash-based solution
- Two-pointer → Sliding window (with different window behavior)
Example: "find pair that sums to target (using two pointers)" → "find all triplets that sum to target" described as "find combinations that meet the sum criteria\""""
        },
        {
            "name": "edge_case_as_main_case",
            "instruction": """Make edge cases the primary requirement, ignore main cases:
- Handle only empty inputs (ignore non-empty)
- Process only single-element cases (ignore multiple elements)
- Focus on null/None values (ignore valid values)
- Handle only duplicates (ignore unique elements)
- Process only boundary values (ignore middle range)
Example: "process list of numbers" → "return empty list if input has 2+ elements, otherwise return input" described as "process and return the optimized list\""""
        },
        {
            "name": "silent_precondition_change",
            "instruction": """Keep task same but change hidden assumptions:
- Assume sorted input when original doesn't (or vice versa)
- Assume unique elements when original allows duplicates
- Assume positive numbers when original allows negative
- Assume ASCII when original allows Unicode
- Change array bounds without stating clearly
Example: "count occurrences" → "count occurrences assuming pre-sorted input in descending order" described as "count frequency of elements in the collection\""""
        },
        {
            "name": "catastrophic_off_by_one_cascade",
            "instruction": """Introduce multiple compounding off-by-one errors:
- Start index: 0 → 1
- End index: n → n-1
- Loop count: n iterations → n+1 iterations
- Array access: arr[i] → arr[i+1] or arr[i-1]
- Length calculation: len(arr) → len(arr)-1
- Range: range(n) → range(1, n+1)
Example: "process first N elements (0 to N-1)" → "process elements from index 1 to N" described as "process the N primary elements\""""
        },
        {
            "name": "semantic_trojan",
            "instruction": """Keep function name but completely change internal behavior:
- is_prime() → checks if composite
- max() → returns min
- remove_duplicates() → keeps only duplicates, removes unique
- sort_ascending() → sorts descending
- find_first() → finds last
- count_true() → counts false
- is_empty() → checks if full
Keep the description ambiguous enough to match the original name.
Example: function signature "def is_palindrome(s)" → check if string is NOT a palindrome, described as "checks the palindrome property of string\""""
        },
        {
            "name": "hidden_negation",
            "instruction": """Add logical negation in subtle ways:
- "include X" → "include everything EXCEPT X"
- "elements that satisfy" → "elements that DON'T satisfy"
- "while condition" → "while NOT condition"
- "if valid" → "if invalid"
- Use double negatives in description to confuse
Example: "return non-empty strings" → "return empty strings" described as "return strings that are not invalid (non-empty violates this)\""""
        },
        {
            "name": "parameter_meaning_swap",
            "instruction": """Keep parameter names but swap their semantic meaning:
- Parameter "target" now means "value to avoid"
- Parameter "max_size" now means "min_size"
- Parameter "include_zeros" now means "exclude_zeros"
- Parameter "ascending" now means "descending"
- Return value meaning reversed (True→False semantics)
Example: "find_element(arr, target, include_duplicates=True)" → find all EXCEPT target, include_duplicates=True means EXCLUDE duplicates"""
        },
        {
            "name": "multi_step_corruption",
            "instruction": """Break multi-step problems by corrupting one step:
- If problem is: parse → filter → transform → aggregate
- Change to: parse → WRONG_filter → transform → aggregate
- Or: parse → filter → WRONG_transform → aggregate
- Make the corruption subtle in description
Example: "parse CSV, filter positive numbers, sum them" → "parse CSV, filter negative numbers, sum them" described as "parse and aggregate numerical data based on criteria\""""
        }
    ]
    
    # 随机选择1-2个模式组合使用（更激进）
    num_patterns = random.choice([1, 2])
    selected_patterns = random.sample(aggressive_patterns, num_patterns)
    
    pattern_instructions = "\n\n".join([
        f"MODIFICATION PATTERN {i+1}: {p['instruction']}" 
        for i, p in enumerate(selected_patterns)
    ])
    
    system_prompt = f"""You are creating ADVERSARIAL test cases for code generation models. Your goal is to make models produce WRONG code.

MISSION: Create a problem that looks deceptively similar but requires FUNDAMENTALLY DIFFERENT logic.

{pattern_instructions}

MANDATORY REQUIREMENTS:

1. **AGGRESSIVE MODIFICATION**:
   - Apply ALL selected patterns above
   - The solution should be INCOMPATIBLE with the original
   - If original returns X, new should return NOT-X or opposite
   - Make changes that BREAK any attempt to reuse original logic

2. **DECEPTIVE SIMILARITY**:
   - Keep function name identical or nearly identical
   - Use similar terminology (but with different meaning)
   - Maintain similar sentence structure
   - Use ambiguous phrasing that hides the true requirement
   - NEVER explicitly say "opposite", "different", "inverted"

3. **EXAMPLES MUST MATCH NEW LOGIC**:
   - Provide 2-3 examples that clearly demonstrate the MODIFIED behavior
   - Examples should make the NEW requirement obvious (even if description is vague)
   - Input/output pairs must be INCONSISTENT with original problem
   - Make examples concrete and verifiable

4. **MAXIMIZE CONFUSION**:
   - Use words that could mean multiple things
   - Keep variable names similar to original
   - Maintain similar code structure hints
   - Add red herrings that point to original solution

5. **CRITICAL - VERIFICATION**:
   - Mentally test: Would original's correct solution fail on your new problem?
   - If answer is NO, modify MORE aggressively
   - The new problem should FAIL with original's logic

6. **OUTPUT FORMAT**:
   - Return ONLY the rewritten problem
   - Include function signature (with same or similar name)
   - Include 2-3 modified examples with explanations
   - NO meta-commentary, NO "here's the modified version", NO markdown code blocks

SEVERITY LEVEL: MAXIMUM - Make it nearly impossible for a model to accidentally solve correctly if it misunderstands.
"""

    user_prompt = f"""Original problem:
{original_prompt}

---

Rewrite this problem applying the modification patterns. 

REMEMBER: 
- The solution logic must be FUNDAMENTALLY INCOMPATIBLE with the original
- Examples must clearly show the NEW (modified) behavior
- Keep it deceptively similar in appearance
- Make it FAIL if someone uses the original solution approach

Generate the adversarial version now:"""

    try:
        # 根据模型选择client
        if '4o' in args.model:
            client = client_4o_mini
            model = 'gpt-4o-mini-ca'
        elif 'deepseek' in args.model:
            client = client_ds_qwen
            model = 'deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct'
        elif 'qwen' in args.model:
            client = client_ds_qwen
            model = 'qwen/Qwen2.5-14B-Instruct'
        else:
            client = client_4o_mini
            model = 'gpt-4o-mini-ca'
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.9,  # 提高到0.9增加变化
            max_tokens=2500,
            top_p=0.95  # 增加多样性
        )
        
        modified_prompt = response.choices[0].message.content.strip()
        
        # 清理格式
        modified_prompt = modified_prompt.replace('```python', '').replace('```', '').strip()
        
        # 移除可能的元注释
        lines = modified_prompt.split('\n')
        cleaned_lines = []
        skip_phrases = [
            'here is the modified',
            'here\'s the modified', 
            'modified version:',
            'rewritten version:',
            'adversarial version:',
            'new problem:'
        ]
        
        for line in lines:
            if not any(phrase in line.lower() for phrase in skip_phrases):
                cleaned_lines.append(line)
        
        modified_prompt = '\n'.join(cleaned_lines).strip()
        
        return modified_prompt
        
    except Exception as e:
        logger.error(f"Error modifying prompt: {e}")
        return None



# # 可选:添加验证函数来检查修改是否足够不同
# def validate_modification(original, modified):
#     """
#     Check if the modification is sufficiently different
#     Returns True if modification seems valid
#     """
#     if not modified or len(modified) < 20:
#         return False
    
#     # 检查是否只是简单复制
#     if original.strip() == modified.strip():
#         return False
    
#     # 可以添加更多验证逻辑
#     # 例如:检查关键词是否有变化
    
#     return True


def load_original_results(result_path):
    """加载原始结果文件，返回通过的任务列表"""
    loaded_dataset = []
    idx=0
    with open(result_path, 'r') as f:
        for line in f:
            task = json.loads(line)
            # 检查是否通过（pass字段或passed字段）
            if task.get('pass', False) or task.get('passed', False):
                loaded_dataset.append(idx)
            idx+=1
    return loaded_dataset

def generate_modified_code(task, method_name, args):
    """为修改后的prompt生成代码"""
    try:
        generate_ids = [i for i in range(args.num_generate)]
        
        if args.parallel == 1:
            with ProcessPoolExecutor() as executor:
                futures = []
                for cnt in generate_ids:
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    new_method_name = method_name + "_" + str(cnt)
                    idea = task['modified_prompt']
                    args_dict = vars(args)
                    future = executor.submit(startup, idea=idea, project_name=new_method_name, args=args_dict)
                    futures.append(future)
                
                for future in as_completed(futures):
                    future.result()
        else:
            for cnt in generate_ids:
                new_method_name = method_name + "_" + str(cnt)
                args_dict = vars(args)
                startup(idea=task['modified_prompt'], project_name=new_method_name, args=args_dict)
        
        return True
    except Exception as e:
        logger.error(f"Error generating code: {e}")
        return False

def evaluate_modified_task(task, idx, args):
    """评估修改后的任务是否在原数据集的evaluation上失败"""
    codes = []
    plans = []
    
    method_name = f"{args.dataset}_modified_{idx}"
    
    for cnt in range(args.num_generate):
        new_method_name = method_name + "_" + str(cnt)
        plan_file_path = f'/home/zlyuaj/Causal/MetaGPT/{args.workspace}/{new_method_name}'
        
        if not os.path.exists(plan_file_path):
            continue
        
        RA, RP, IA = extract_plan_from_repo(plan_file_path)
        code_file_path = f'/home/zlyuaj/Causal/MetaGPT/{args.workspace}/{new_method_name}/{new_method_name}'
        
        if not os.path.exists(code_file_path):
            continue
        
        code = extract_code_from_repo(code_file_path)
        if not code:
            continue
        
        plan = format_plan(RA, RP, IA)
        code = 'from typing import List\n' + code
        
        import re
        def remove_input_content(code):
            code = re.sub(r'input\([^)]*\)', 'input()', code)
            return code
        code = remove_input_content(code)
        
        codes.append(code)
        plans.append(plan)
    
    if not codes:
        return False, 0, None
    
    while len(codes) < args.num_generate:
        ran = random.randint(0, len(codes) - 1)
        codes.append(codes[ran])
        plans.append(plans[ran])
    
    # 使用原始的test来评估修改后的代码
    eval_task = task.copy()
    eval_task['completions'] = codes
    eval_task['plans'] = plans
    
    try:
        if 'human' in args.dataset.lower():
            score, passes, passAt1, passAt10 = evaluate_one(eval_task, args.num_generate)
        elif 'mbpp' in args.dataset.lower():
            score, passes, passAt1, passAt10 = evaluate_one_MBPP(eval_task, args.num_generate)
        elif 'codecontest' in args.dataset.lower():
            score, passes, passAt1, passAt10 = evaluate_one_codecontest(eval_task, args.num_generate)
        elif 'codereval' in args.dataset.lower():
            passAt10, passes = evaluate_one_CoderEval(eval_task, args.num_generate)
        else:
            score, passes, passAt1, passAt10 = evaluate_one(eval_task, args.num_generate)
        

        with open(plan_file_path + '/eval_result.txt','w+') as eval_f:
            eval_f.write(f'{passes}\n')
        return passAt10, passes, codes
    except Exception as e:
        logger.error(f"Error evaluating: {e}")
        return None, 0, codes
def main():
    # 创建输出目录
    initial_output_path = args.output_path
    if not os.path.exists(initial_output_path):
        os.makedirs(initial_output_path)
    
    args.output_path = os.path.join(initial_output_path, f'results-{args.output_file_name}/')
    x = 2
    while os.path.exists(args.output_path):
        args.output_path = os.path.join(initial_output_path, f'results-{args.output_file_name}_{x}/')
        x += 1
    os.makedirs(args.output_path)
    
    print(f"Output path: {args.output_path}")
    print(f"Args: {args}")
    
    # 加载原始通过的任务
    INPUTPATH = args.input_path
    loaded_dataset = []
    with open(INPUTPATH, 'r') as f:
        loaded_dataset = [json.loads(line) for line in f]
    
    original_path = f'/home/zlyuaj/Causal/MetaGPT/output/{args.dataset}/results-{args.dataset}_{args.model}_original_dataset/{args.dataset}.jsonl' 
    
    passed_idx_list = load_original_results(original_path)
    # 过滤索引范围
    loaded_dataset = [t for i, t in enumerate(loaded_dataset) if args.begin_idx <= i < args.end_idx]
    print(f"Processing {len(loaded_dataset)} tasks (index {args.begin_idx} to {args.end_idx})")
    
    # 输出文件
    output_file = os.path.join(args.output_path, f'{args.dataset}_modified.jsonl')
    summary_file = os.path.join(args.output_path, f'{args.dataset}_summary.json')

    original_result_path = f'/home/zlyuaj/Causal/MetaGPT/workspace_{args.model}_{args.dataset}_original_result'
    
    successful_modifications = []
    failed_modifications = []
    unsuccessful_idx_list = []  # 记录不成功的idx
    reran_case = []
    reran_case = [12,38,50]
    args.rerun=1
    with open(output_file, 'w') as f:
        for idx, task in enumerate(loaded_dataset):
            print(f"\n{'='*50}")
            print(f"Processing task {idx}/{len(loaded_dataset)}")
            print(f"{'='*50}")

            # if os.path.exists(f'/home/zlyuaj/Causal/MetaGPT/workspace_deepseek-coder_humaneval_modified_orignal/humaneval_modified_{idx}_0'):
            #     continue
            # if idx not in passed_idx_list:
            #     print('original result failed. Continue...')
            #     continue

            if args.rerun == 1 and idx not in reran_case:
                print('Not in reran_case. Continue...')
                continue
            
            # 最多重试3次
            max_attempts = 3
            success_flag = False
            
            for attempt in range(max_attempts):
                print(f"\n--- Attempt {attempt + 1}/{max_attempts} ---")
                
                # 1. 改写prompt
                print("Step 1: Modifying prompt.")
                modified_prompt= ''
                original_prompt = task.get('prompt', task.get('description', task.get('text', task.get('input', ''))))
                if args.rerun==1:
                    modified_prompt = modify_prompt_agg(original_prompt)
                else:
                    modified_prompt = modify_prompt_new(original_prompt)
                
                if not modified_prompt:
                    print(f"Failed to modify prompt for task {idx} (attempt {attempt + 1})")
                    if attempt == max_attempts - 1:
                        failed_modifications.append({'idx': idx, 'reason': 'prompt_modification_failed'})
                        unsuccessful_idx_list.append(idx)
                    continue
                
                print(f"Original prompt: {original_prompt[:100]}...")
                print(f"Modified prompt: {modified_prompt[:100]}...")
                
                task['modified_prompt'] = modified_prompt
                task['original_prompt'] = original_prompt
                
                # 2. 生成代码
                print("Step 2: Generating code for modified prompt...")
                method_name = f"{args.dataset}_modified_{idx}"
                code_success = generate_modified_code(task, method_name, args)
                
                if not code_success:
                    print(f"Failed to generate code for task {idx} (attempt {attempt + 1})")
                    if attempt == max_attempts - 1:
                        failed_modifications.append({'idx': idx, 'reason': 'code_generation_failed'})
                        unsuccessful_idx_list.append(idx)
                    continue
                
                # 3. 评估（使用原始的test）
                print("Step 3: Evaluating modified code with original tests...")
                passAt10, passes, codes = evaluate_modified_task(task, idx, args)
                
                task['modified_pass'] = passAt10
                task['modified_pass_num'] = passes
                task['modified_completions'] = codes
                
                # 4. 检查文件是否存在
                print("Step 4: Checking if required files exist...")
                files_exist = False
                try:
                    files_exist = check_one_exist(args.dataset, args.model, idx)
                except (FileNotFoundError, NotImplementedError) as e:
                    print(f"File check failed: {e}")
                    files_exist = False
                
                # 判断是否成功：eval结果为false 且 文件都存在
                if passAt10 is False and files_exist:
                    print(f"✓ Successfully created failing case for task {idx} with all files present")
                    result_entry = {
                        'idx': idx,
                        'original_pass': task.get('pass', True),
                        'modified_pass': passAt10,
                        'original_prompt': original_prompt,
                        'modified_prompt': modified_prompt,
                        'attempts': attempt + 1
                    }
                    successful_modifications.append(result_entry)
                    success_flag = True
                    break  # 成功，跳出重试循环
                else:
                    # 记录失败原因
                    fail_reason = []
                    if passAt10 is not False:
                        fail_reason.append(f"eval_still_passes(passAt10={passAt10})")
                    if not files_exist:
                        fail_reason.append("files_missing")
                    
                    print(f"✗ Attempt {attempt + 1} failed: {', '.join(fail_reason)}")
                    
                    if attempt == max_attempts - 1:
                        # 最后一次尝试仍然失败
                        result_entry = {
                            'idx': idx,
                            'original_pass': task.get('pass', True),
                            'modified_pass': passAt10,
                            'original_prompt': original_prompt,
                            'modified_prompt': modified_prompt,
                            'reason': ', '.join(fail_reason),
                            'attempts': max_attempts
                        }
                        failed_modifications.append(result_entry)
                        unsuccessful_idx_list.append(idx)
            
            # 保存到文件（保存最后一次的结果）
            if success_flag or attempt == max_attempts - 1:
                f.write(json.dumps(task, ensure_ascii=False) + '\n')
                f.flush()
    
    # 保存摘要
    summary = {
        'total_loaded_dataset': len(loaded_dataset),
        'successful_modifications': len(successful_modifications),
        'failed_modifications': len(failed_modifications),
        'unsuccessful_idx_list': unsuccessful_idx_list,  # 添加不成功的idx列表
        'success_rate': len(successful_modifications) / len(loaded_dataset) if loaded_dataset else 0,
        'successful_cases': successful_modifications,
        'failed_cases': failed_modifications
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*50}")
    print("Summary:")
    print(f"Total passed tasks processed: {len(loaded_dataset)}")
    print(f"Successful modifications (now failing with files): {len(successful_modifications)}")
    print(f"Failed modifications: {len(failed_modifications)}")
    print(f"Success rate: {summary['success_rate']:.2%}")
    print(f"\nUnsuccessful indices: {unsuccessful_idx_list}")
    print(f"Results saved to: {output_file}")
    print(f"Summary saved to: {summary_file}")


if __name__ == '__main__':
    main()

'''

"""You are a professional requirement modification assistant. Your task is to subtly rewrite the given programming problem requirement so that:

1. It appears very similar to the original problem, almost identical at first glance
2. But differs in key details that would cause the original correct solution to no longer work
3. This modification should simulate a misunderstanding that an LLM might make when interpreting the problem

Modification strategies can include:
- Change subtle requirements in input/output format
- Modify boundary conditions or special case handling
- Adjust the function's return value type or structure
- Alter problem constraints
- Fine-tune algorithmic requirements (e.g., sorting order, comparison rules, etc.)
- Change edge case definitions
- Modify the expected behavior for empty inputs or null values
- Adjust rounding/precision requirements
- Change uniqueness or ordering requirements

Important guidelines:
- Keep the overall problem structure and function signature similar
- Make changes that seem minor but are semantically significant
- The modification should be plausible as a misinterpretation, not completely different
- Maintain the same programming language and general approach

Return ONLY the rewritten prompt with no additional explanations or comments."""


for example 
originl requirement:
"from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n"

modified requirement:
"from typing import List\n\n\ndef has_true_elements(numbers: List[bool]) -> bool:\n    \"\"\" Given a list of input, judge whether there exist 'True' in the list\n >>> has_true_elements([True, False, False])\n    True\n    >>> has_true_elements([False, False])\n    False\n    \"\"\"\n"

'''
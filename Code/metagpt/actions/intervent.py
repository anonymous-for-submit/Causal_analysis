
{"Language": "en_us", "Programming Language": "Python", "Original Requirements": "Write a python function to identify non-prime numbers.\nExamples:\n- is_not_prime(2) == False\n- is_not_prime(10) == True", "Product Goals": ["Create an efficient and accurate algorithm for prime number identification", "Ensure the function is optimized for performance and scalability", "Develop a user-friendly interface for easy integration into other applications"], "User Stories": ["As a developer, I want to use the function to filter out non-prime numbers in a list", "As a user, I want the function to be fast and reliable so I can trust its results", "As a teacher, I want to use the function in educational materials to explain prime numbers"], "Competitive Analysis": ["Prime Number Checker: Simple and straightforward, but lacks advanced features", "Advanced Number Theory Tool: Comprehensive but complex and not user-friendly", "Number Sieve: Efficient but designed for finding prime numbers, not identifying non-prime numbers", "Python Math Library: Includes prime checking but not as a standalone function", "Custom Application: Specific to our needs but lacks the advanced features of our target product", "Online Prime Checker: User-friendly but may not be as reliable or fast as our target product"], "Competitive Quadrant Chart": "quadrantChart\n    title \"Product Performance and User Experience\"\n    x-axis \"Low Performance\" --> \"High Performance\"\n    y-axis \"Poor User Experience\" --> \"Excellent User Experience\"\n    quadrant-1 \"Re-evaluate\"\n    quadrant-2 \"Need to promote\"\n    quadrant-3 \"We should expand\"\n    quadrant-4 \"May be improved\"\n    \"Prime Number Checker\": [0.2, 0.3]\n    \"Advanced Number Theory Tool\": [0.4, 0.5]\n    \"Number Sieve\": [0.6, 0.7]\n    \"Python Math Library\": [0.3, 0.4]\n    \"Custom Application\": [0.7, 0.8]\n    \"Online Prime Checker\": [0.5, 0.6]\n    \"Our Target Product\": [0.8, 0.9]", "Requirement Analysis": "The primary requirement is to create a function that can identify non-prime numbers efficiently. This function should be able to handle a range of inputs and provide accurate results. The user stories highlight the need for the function to be fast, reliable, and easy to integrate into various applications. The competitive analysis shows that there is a need for a product that offers both high performance and a good user experience.", "Requirement Pool": [["P0", "The main code should be optimized for speed and accuracy in identifying non-prime numbers."], ["P0", "The function should be user-friendly and easy to integrate into other applications."], ["P1", "The product should include advanced features for handling complex prime number checks."], ["P1", "The user interface should be intuitive and provide clear feedback on the results."], ["P2", "The product should be compatible with a wide range of devices and platforms."]], "UI Design draft": "A simple and clean interface with input fields for numbers and a button to check if they are prime. The output should be clearly displayed below the input field, indicating whether the number is prime or not.", "Anything UNCLEAR": ""}

prd2change = {"Language":"chinese","Programming Language":"Java","Original Requirements":"Write a python function to identify non-prime numbers.","Product Goals":["Create a reliable function for identifying non-prime numbers","Ensure the function is efficient and handles large inputs","Provide clear documentation for users"],"User Stories":["As a user, I want to input a number and receive a response indicating if it is non-prime","As a user, I want to check a list of numbers for non-primality","As a user, I want to understand the logic behind the non-prime identification"],"Competitive Analysis":["NumPy: Efficient numerical operations but lacks specific non-prime identification","SymPy: Offers prime checking but is more complex than needed for simple tasks","math.isprime: Simple prime checking but does not directly identify non-prime numbers","Primality Test Libraries: Various libraries exist but may be overkill for basic needs","Custom Implementations: Many users create their own functions but may lack optimization"],"Competitive Quadrant Chart":"quadrantChart\n    title \"Reach and engagement of campaigns\"\n    x-axis \"Low Reach\" --> \"High Reach\"\n    y-axis \"Low Engagement\" --> \"High Engagement\"\n    quadrant-1 \"We should expand\"\n    quadrant-2 \"Need to promote\"\n    quadrant-3 \"Re-evaluate\"\n    quadrant-4 \"May be improved\"\n    \"Library A\": [0.3, 0.6]\n    \"Library B\": [0.45, 0.23]\n    \"Library C\": [0.57, 0.69]\n    \"Library D\": [0.78, 0.34]\n    \"Library E\": [0.40, 0.34]\n    \"Our Target Function\": [0.5, 0.6]","Requirement Analysis":"The function should efficiently determine non-prime numbers, ideally using a method that minimizes computational complexity. It should handle edge cases such as negative numbers and zero.","Requirement Pool":[["P0","Implement a function to check for non-prime numbers."],["P1","Optimize the function for performance with large inputs."],["P2","Provide unit tests to ensure accuracy of the function."]],"UI Design draft":"The function will be a simple command-line interface where users can input numbers and receive immediate feedback on their primality status.","Anything UNCLEAR":"Clarification on whether the function should handle only integers or also floating-point numbers."}

design2change = {"Implementation approach":"We will implement a simple function to identify non-prime numbers using basic mathematical checks. The function will handle edge cases such as negative numbers and zero, and will be optimized for performance to handle large inputs efficiently. We will use the built-in Python capabilities without relying on heavy external libraries, ensuring simplicity and reliability.","File list":["test.py"],"Data structures and interfaces":"def is_non_prime(n: int) -> bool:\n    \"\"\"\n    Check if a number is non-prime.\n    \n    Parameters:\n    n (int): The number to check.\n    \n    Returns:\n    bool: True if the number is non-prime, False if it is prime.\n    \"\"\"\n    # Handle edge cases\n    if n <= 1:\n        return True  # 0 and 1 are non-prime\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return True  # Found a divisor, hence non-prime\n    return False  # No divisors found, hence prime","Program call flow":"sequenceDiagram\n    participant User\n    participant Function\n    User->>Function: Call is_non_prime(n)\n    Function-->>User: Return True if non-prime, False if prime\n    Note over Function: Check if n <= 1\n    alt n <= 1\n        Function-->>User: Return True\n    else\n        Note over Function: Loop from 2 to sqrt(n)\n        loop Check divisibility\n            Function-->>Function: Check if n % i == 0\n            alt Found divisor\n                Function-->>User: Return True\n            end\n        end\n        Function-->>User: Return False","Anything UNCLEAR":"Clarification needed on whether the function should handle only integers or also floating-point numbers."}

task2change={"Required packages":["required packages are a,b,c,d,e,f,g"],"Required Other language third-party packages":["These are the required third-party packages: x,y,z"],"Logic Analysis":[["test.py","Contains one function with the same input/output format as user requirement"]],"File list":["test.py"],"Full API spec":"","Shared Knowledge":"","Anything UNCLEAR":"Clarification needed on whether the function should handle only integers or also floating-point numbers."}

# {"file_name": "mbpp_1_0", "requirements": "No required packages", "code": "## main.py\n\nimport math\n\ndef is_not_prime(number: int) -> bool:\n    \"\"\"\n    Determines if a number is non-prime.\n    \n    Args:\n        number (int): The number to check for non-primality.\n        \n    Returns:\n        bool: True if the number is non-prime, False otherwise.\n    \"\"\"\n    if number <= 1:\n        return True\n    if number <= 3:\n        return False\n    \n    if number % 2 == 0 or number % 3 == 0:\n        return True\n    \n    i = 5\n    while i * i <= number:\n        if number % i == 0 or number % (i + 2) == 0:\n            return True\n        i += 6\n    \n    return False\n", "prd": {"Language": "en_us", "Programming Language": "Python", "Original Requirements": "Write a python function to identify non-prime numbers.\nExamples:\n- is_not_prime(2) == False\n- is_not_prime(10) == True", "Product Goals": ["Create an efficient and accurate algorithm for prime number identification", "Ensure the function is optimized for performance and scalability", "Develop a user-friendly interface for easy integration into other applications"], "User Stories": ["As a developer, I want to use the function to filter out non-prime numbers in a list", "As a user, I want the function to be fast and reliable so I can trust its results", "As a teacher, I want to use the function in educational materials to explain prime numbers"], "Competitive Analysis": ["Prime Number Checker: Simple and straightforward, but lacks advanced features", "Advanced Number Theory Tool: Comprehensive but complex and not user-friendly", "Number Sieve: Efficient but designed for finding prime numbers, not identifying non-prime numbers", "Python Math Library: Includes prime checking but not as a standalone function", "Custom Application: Specific to our needs but lacks the advanced features of our target product", "Online Prime Checker: User-friendly but may not be as reliable or fast as our target product"], "Competitive Quadrant Chart": "quadrantChart\n    title \"Product Performance and User Experience\"\n    x-axis \"Low Performance\" --> \"High Performance\"\n    y-axis \"Poor User Experience\" --> \"Excellent User Experience\"\n    quadrant-1 \"Re-evaluate\"\n    quadrant-2 \"Need to promote\"\n    quadrant-3 \"We should expand\"\n    quadrant-4 \"May be improved\"\n    \"Prime Number Checker\": [0.2, 0.3]\n    \"Advanced Number Theory Tool\": [0.4, 0.5]\n    \"Number Sieve\": [0.6, 0.7]\n    \"Python Math Library\": [0.3, 0.4]\n    \"Custom Application\": [0.7, 0.8]\n    \"Online Prime Checker\": [0.5, 0.6]\n    \"Our Target Product\": [0.8, 0.9]", "Requirement Analysis": "The primary requirement is to create a function that can identify non-prime numbers efficiently. This function should be able to handle a range of inputs and provide accurate results. The user stories highlight the need for the function to be fast, reliable, and easy to integrate into various applications. The competitive analysis shows that there is a need for a product that offers both high performance and a good user experience.", "Requirement Pool": [["P0", "The main code should be optimized for speed and accuracy in identifying non-prime numbers."], ["P0", "The function should be user-friendly and easy to integrate into other applications."], ["P1", "The product should include advanced features for handling complex prime number checks."], ["P1", "The user interface should be intuitive and provide clear feedback on the results."], ["P2", "The product should be compatible with a wide range of devices and platforms."]], "UI Design draft": "A simple and clean interface with input fields for numbers and a button to check if they are prime. The output should be clearly displayed below the input field, indicating whether the number is prime or not.", "Anything UNCLEAR": ""}, "system_design": {"Implementation approach": "To create a function that identifies non-prime numbers efficiently, we will use a simple and optimized approach. We will leverage the fact that a non-prime number must have at least one divisor other than 1 and itself. We will implement this by checking divisibility from 2 up to the square root of the number. This method is efficient and avoids unnecessary checks.", "File list": ["main.py"], "Data structures and interfaces": "def is_not_prime(number: int) -> bool:  # The function takes an integer as input and returns a boolean indicating whether the number is non-prime.", "Program call flow": "sequenceDiagram\n    participant Developer\n    participant User\n    participant Teacher\n    Developer->>+main.py: import is_not_prime\n    Developer->>+main.py: result = is_not_prime(number)\n    Developer->>+User: Provide result\n    Teacher->>+main.py: import is_not_prime\n    Teacher->>+main.py: result = is_not_prime(number)\n    Teacher->>+Teacher: Provide result\n", "Anything UNCLEAR": "No unclear aspects mentioned."}, "task": {"Required packages": ["No required packages"], "Required Other language third-party packages": ["No third-party dependencies required"], "Logic Analysis": [["main.py", "Contains the function is_not_prime(number: int) -> bool: to identify if a number is non-prime."]], "File list": ["main.py"], "Full API spec": "", "Shared Knowledge": "", "Anything UNCLEAR": "No unclear aspects mentioned."}, "eval_result": null}



# input of product manager
# [{'role': 'system', 'content': 'You are a Project Manager, named Eve, your goal is break down tasks according to PRD/technical design, generate the task for completing the input funtion in one filedependencies to start with the prerequisite modules. the constraint is use same language as user requirement. '}, {'role': 'user', 'content': '\n## context\n{"Implementation approach":"We will implement a function named `has_close_elements` that iterates through the list of numbers and checks if any two numbers are closer than the specified threshold. The function will handle edge cases such as empty lists and lists with a single element. We will use the built-in capabilities of Python for efficient comparisons without the need for external libraries, as the requirements do not specify complex data structures or operations.","File list":["main.py"],"Data structures and interfaces":"The function signature is `def has_close_elements(numbers: List[float], threshold: float) -> bool:`. The input is a list of floats `numbers` and a float `threshold`. The output is a boolean value indicating whether any two numbers in the list are closer than the threshold. The function will return `False` for an empty list or a list with a single element.","Program call flow":"sequenceDiagram\\n    participant User\\n    participant Function\\n    User->>Function: has_close_elements([1.0, 2.0, 3.0], 0.5)\\n    Function-->>User: return False\\n    User->>Function: has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\\n    Function-->>User: return True\\n    User->>Function: has_close_elements([], 0.5)\\n    Function-->>User: return False\\n    User->>Function: has_close_elements([1.0], 0.5)\\n    Function-->>User: return False","Anything UNCLEAR":"Clarification needed on whether the function should handle only integers or also floating-point numbers."}\n\n-----\n\n## format example\n[CONTENT]\n{\n    "Required packages": [\n        "No required packages"\n    ],\n    "Required Other language third-party packages": [\n        "No third-party dependencies required"\n    ],\n    "Logic Analysis": [\n        [\n            "main.py",\n            "Contains one function with the same input/output format as user requirment"\n        ]\n    ],\n    "File list": [\n        "main.py"\n    ],\n    "Full API spec": "openapi: 3.0.0 ...",\n    "Shared Knowledge": "",\n    "Anything UNCLEAR": "Clarification needed on how to start and initialize third-party libraries."\n}\n[/CONTENT]\n\n## nodes: "<node>: <type>  # <instruction>"\n- Required packages: typing.Optional[typing.List[str]]  # Provide required packages from standard library of python if needed.\n- Required Other language third-party packages: typing.List[str]  # Please state that no other packages are provided\n- Logic Analysis: typing.List[typing.List[str]]  # Provide one main.py file to be implemented, make sure only use the standard libraryincluding dependency analysis and imports.\n- File list: typing.List[str]  # Only include main.py\n- Full API spec: <class \'str\'>  # Describe all APIs using OpenAPI 3.0 spec that may be used by both frontend and backend. If front-end and back-end communication is not required, leave it blank.\n- Shared Knowledge: <class \'str\'>  # Detail any shared knowledge, like common utility functions or configuration variables.\n- Anything UNCLEAR: <class \'str\'>  # Mention any unclear aspects in the project management context and try to clarify them.\n\n\n## constraint\nLanguage: Please use the same language as Human INPUT.\nFormat: output wrapped inside [CONTENT][/CONTENT] like format example, nothing else.\n\n## action\nFollow instructions of nodes, generate output and make sure it follows the format example.\n'}]


import copy

def clear_changed_contents():
    l=[prd2change,design2change,task2change]
    for item in l:
        for key in item.keys():
            if type(item[key])==str:
                item[key]=''
            elif type(item[key])==list:
                item[key]=[]
def modified_changed_contents(all_modified_prd, all_modified_design, all_modified_task,idx):
    global prd2change,design2change,task2change
    prd2change,design2change,task2change = copy.deepcopy(all_modified_prd[idx]), copy.deepcopy(all_modified_design[idx]), copy.deepcopy(all_modified_task[idx])
    
    for k in prd2change.keys():
        if k not in all_modified_prd[idx].keys():
            raise NotImplementedError
        prd2change[k] = all_modified_prd[idx][k]

    for k in design2change.keys():
        if k not in all_modified_design[idx].keys():
            raise NotImplementedError
        design2change[k] = all_modified_design[idx][k]
    
    for k in task2change.keys():
        if k not in all_modified_task[idx].keys():
            raise NotImplementedError
        task2change[k] = all_modified_task[idx][k]

    print('Modified contents at idx=',idx)
    print(design2change)

def print_prd():
    print(prd2change)


import os
import json

def read_json_file(file_path):
    """辅助函数：读取JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def read_internal_output(dataset, model, idx, base_path = ''):
    """
    读取指定dataset、model和idx对应的prd、design、task文件
    
    Args:
        dataset: 数据集名称
        model: 模型名称
        idx: 索引号

    
    Raises:
        FileNotFoundError: 当文件不存在时抛出异常
    """
    prd, design, task = '', '', ''
    
    # 构造基础路径
    if not base_path:
        base_path = f'/home/zlyuaj/Causal/MetaGPT/workspace_{model}_{dataset}_modified_orignal'
        folder_name = f'{dataset}_modified_{idx}_0'
    else:
        folder_name = f'{dataset}_{idx}_0'
    
    # 构造各个文档的路径
    prd_path = os.path.join(base_path, folder_name, 'docs', 'prd')
    design_path = os.path.join(base_path, folder_name, 'docs', 'system_design')
    task_path = os.path.join(base_path, folder_name, 'docs', 'task')
    
    # 读取prd文件
    if not os.path.exists(prd_path):
        raise FileNotFoundError(f'PRD路径不存在: {prd_path}')
    prd_files = [f for f in os.listdir(prd_path) if f.endswith('.json')]
    if not prd_files:
        raise FileNotFoundError(f'PRD文件不存在于: {prd_path}')
    prd = read_json_file(os.path.join(prd_path, prd_files[0]))
    
    # 读取design文件
    if not os.path.exists(design_path):
        raise FileNotFoundError(f'Design路径不存在: {design_path}')
    design_files = [f for f in os.listdir(design_path) if f.endswith('.json')]
    if not design_files:
        raise FileNotFoundError(f'Design文件不存在于: {design_path}')
    design = read_json_file(os.path.join(design_path, design_files[0]))
    
    # 读取task文件
    if not os.path.exists(task_path):
        raise FileNotFoundError(f'Task路径不存在: {task_path}')
    task_files = [f for f in os.listdir(task_path) if f.endswith('.json')]
    if not task_files:
        raise FileNotFoundError(f'Task文件不存在于: {task_path}')
    task = read_json_file(os.path.join(task_path, task_files[0]))
    

    prd2change, design2change, task2change = prd, design, task
    return prd, design, task


# type=task


datasets = ["humaneval", "mbpp", "codecontest", "CoderEval"]
models = ["gpt-4o-mini-ca", "deepseek-coder", "qwen"]





def check_all_exist(dataset, model):
    """
    检查指定dataset和model的workspace下所有文件夹中的prd、design、task文件是否存在
    
    Args:
        dataset: 数据集名称
        model: 模型名称
    
    Returns:
        dict: 包含检查结果的字典，格式为 {idx: {'prd': bool, 'design': bool, 'task': bool}}
    
    Raises:
        FileNotFoundError: 当workspace路径不存在时抛出异常
    """
    # 构造基础路径
    base_path = f'/home/zlyuaj/Causal/MetaGPT/workspace_{model}_{dataset}_modified_orignal_re'
    
    if not os.path.exists(base_path):
        raise FileNotFoundError(f'Workspace路径不存在: {base_path}')
    
    # 获取所有文件夹并排序
    directories = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    directories = sorted(directories, key=lambda x: int(x.split('_')[-2]))
    
    check_results = {}
    missing_files = []
    
    for directory in directories:
        # 提取idx
        idx = int(directory.split('_')[-2])
        
        # 构造各个文档的路径
        prd_path = os.path.join(base_path, directory, 'docs', 'prd')
        design_path = os.path.join(base_path, directory, 'docs', 'system_design')
        task_path = os.path.join(base_path, directory, 'docs', 'task')
        
        result = {'prd': False, 'design': False, 'task': False}
        
        # 检查prd文件
        if os.path.exists(prd_path):
            prd_files = [f for f in os.listdir(prd_path) if f.endswith('.json')]
            result['prd'] = len(prd_files) > 0
            if not result['prd']:
                missing_files.append(f'{directory}/docs/prd: 无JSON文件')
        else:
            missing_files.append(f'{directory}/docs/prd: 路径不存在')
        
        # 检查design文件
        if os.path.exists(design_path):
            design_files = [f for f in os.listdir(design_path) if f.endswith('.json')]
            result['design'] = len(design_files) > 0
            if not result['design']:
                missing_files.append(f'{directory}/docs/system_design: 无JSON文件')
        else:
            missing_files.append(f'{directory}/docs/system_design: 路径不存在')
        
        # 检查task文件
        if os.path.exists(task_path):
            task_files = [f for f in os.listdir(task_path) if f.endswith('.json')]
            result['task'] = len(task_files) > 0
            if not result['task']:
                missing_files.append(f'{directory}/docs/task: 无JSON文件')
        else:
            missing_files.append(f'{directory}/docs/task: 路径不存在')
        
        check_results[idx] = result
    
    # 打印检查结果
    print(f'\n检查 {model} - {dataset}:')
    print(f'总共检查了 {len(directories)} 个文件夹')
    
    all_exist = all(all(v.values()) for v in check_results.values())
    if all_exist:
        print('✓ 所有文件都存在')
    else:
        print(f'✗ 发现 {len(missing_files)} 个缺失文件:')
        for missing in missing_files:
            print(f'  - {missing}')
        raise NotImplementedError
    
    return check_results

def check_one_exist(dataset, model, idx):
    """
    检查指定dataset和model的workspace下所有文件夹中的prd、design、task文件是否存在
    
    Args:
        dataset: 数据集名称
        model: 模型名称
    
    Returns:
        dict: 包含检查结果的字典，格式为 {idx: {'prd': bool, 'design': bool, 'task': bool}}
    
    Raises:
        FileNotFoundError: 当workspace路径不存在时抛出异常
    """
    # 构造基础路径
    base_path = f'/home/zlyuaj/Causal/MetaGPT/workspace_{model}_{dataset}_modified_orignal'
    
    if not os.path.exists(base_path):
        raise FileNotFoundError(f'Workspace路径不存在: {base_path}')
    
    # 获取所有文件夹并排序
    directories = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    directories = sorted(directories, key=lambda x: int(x.split('_')[-2]))
    
    check_results = {}
    missing_files = []
    
    for directory in directories:
        # 提取idx
        cur_idx = int(directory.split('_')[-2])

        if cur_idx!=idx:
            continue
        
        # 构造各个文档的路径
        prd_path = os.path.join(base_path, directory, 'docs', 'prd')
        design_path = os.path.join(base_path, directory, 'docs', 'system_design')
        task_path = os.path.join(base_path, directory, 'docs', 'task')
        
        result = {'prd': False, 'design': False, 'task': False}
        
        # 检查prd文件
        if os.path.exists(prd_path):
            prd_files = [f for f in os.listdir(prd_path) if f.endswith('.json')]
            result['prd'] = len(prd_files) > 0
            if not result['prd']:
                missing_files.append(f'{directory}/docs/prd: 无JSON文件')
        else:
            missing_files.append(f'{directory}/docs/prd: 路径不存在')
        
        # 检查design文件
        if os.path.exists(design_path):
            design_files = [f for f in os.listdir(design_path) if f.endswith('.json')]
            result['design'] = len(design_files) > 0
            if not result['design']:
                missing_files.append(f'{directory}/docs/system_design: 无JSON文件')
        else:
            missing_files.append(f'{directory}/docs/system_design: 路径不存在')
        
        # 检查task文件
        if os.path.exists(task_path):
            task_files = [f for f in os.listdir(task_path) if f.endswith('.json')]
            result['task'] = len(task_files) > 0
            if not result['task']:
                missing_files.append(f'{directory}/docs/task: 无JSON文件')
        else:
            missing_files.append(f'{directory}/docs/task: 路径不存在')
        
        check_results[idx] = result
    
    # 打印检查结果
    print(f'\n检查 {model} - {dataset} - {idx}:')
    
    all_exist = all(all(v.values()) for v in check_results.values())
    if all_exist:
        print('✓ 所有文件都存在')
    else:
        print(f'✗ 发现 {len(missing_files)} 个缺失文件:')
        for missing in missing_files:
            print(f'  - {missing}')
        raise NotImplementedError
    
    return all_exist

def get_level_map():
    level_map = {
        'prd':prd2change,
        'design':design2change,
        'task':task2change
    }
    return level_map

import json

from pathlib import Path
from typing import Optional

from metagpt.actions import Action, ActionOutput
from metagpt.const import DATA_API_DESIGN_FILE_REPO, SEQ_FLOW_FILE_REPO
from metagpt.logs import logger
from metagpt.schema import Document, Documents, Message
from metagpt.utils.mermaid import mermaid_to_file


def get_content_from_context(context,key="Original Requirements"):
    try:
        data = json.loads(context)
        original_requirements=data[key]
        return original_requirements
    except:
        print('can not get req from context, try splitting...')
        # 查找原始需求的位置
        try:
            start_index = context.find(key) + len(key)
            end_index = context.find('",', start_index)

            # 提取原始需求内容
            original_requirements = context[start_index:end_index].strip().replace('\\n', '\n').replace('\\"', '"')

            return original_requirements
        except:
            logger.error('can not get original requirement')
            return ''

def intervent_node_prd(node, context, level, keys2change, args):
    level_map = get_level_map()
    if 'all_original_prd' in args.keys():
        o_prd = args['all_original_prd'][args['cur_id']]
        all_keys = o_prd.keys()
        changed_contents = []
        for key in all_keys:
            if key in keys2change:
                changed_content = level_map[level][key]
            else:
                changed_content = o_prd[key]
            # if not changed_content:
            #     logger.error(f'fail to get changed_content for key:{key}, using original content!')
            #     changed_content = get_content_from_context(context,key)
            changed_contents.append(changed_content)
    else:
        all_keys = keys2change
        changed_contents = []
        for key in keys2change:
            changed_content = level_map[level][key]
            changed_contents.append(changed_content)
    

    # logger.error(f'contentvalue:{changed_contents}')
    if level == 'design':
        node.refresh_instruct_content(keys=all_keys,changed_contents=changed_contents)
    if level == 'prd':
        node.refresh_instruct_content_prd(keys=all_keys,changed_contents=changed_contents)
    logger.error(f'after intervent, context:{node.instruct_content.model_dump_json()}')

    return node
    
def intervent_node_design(node, context, level, keys2change, args):
    level_map = get_level_map()
    # 无prd，只有design，要复原为原来的
    if 'prd' not in args['levels'] and 'all_original_design' in args.keys():
        o_design = args['all_original_design'][args['cur_id']]
        all_keys = o_design.keys()
        changed_contents = []
        for key in all_keys:
            if key in keys2change:
                changed_content = level_map[level][key]
            else:
                changed_content = o_design[key]
            # if not changed_content:
            #     logger.error(f'fail to get changed_content for key:{key}, using original content!')
            #     changed_content = get_content_from_context(context,key)
            changed_contents.append(changed_content)
        node.refresh_instruct_content(keys=all_keys,changed_contents=changed_contents)
    # 如果有design有prd，不复原
    # 也就是prd改变了，那么design就不能复原成原来的design，要保留已经生成的
    else:    
        changed_contents = []
        for key in keys2change:
            changed_content = level_map[level][key]
            # if not changed_content:
            #     logger.error(f'fail to get changed_content for key:{key}, using original content!')
            #     changed_content = get_content_from_context(context,key)
            changed_contents.append(changed_content)
        node.refresh_instruct_content(keys=keys2change,changed_contents=changed_contents)
    # logger.error(f'contentvalue:{changed_contents}')
    
    logger.error(f'after intervent, context:{node.instruct_content.model_dump_json()}')
    return node


def intervent_node_task(node, context, level, keys2change, args):
    level_map = get_level_map()
    # task只在没有prd和desigh的时候复原
    if 'prd' not in args['levels'] and 'design' not in args['levels'] and 'all_original_task' in args.keys():
        o_task = args['all_original_task'][args['cur_id']]
        all_keys = o_task.keys()
        changed_contents = []
        for key in all_keys:
            if key in keys2change:
                changed_content = level_map[level][key]
            else:
                changed_content = o_task[key]
            # if not changed_content:
            #     logger.error(f'fail to get changed_content for key:{key}, using original content!')
            #     changed_content = get_content_from_context(context,key)
            changed_contents.append(changed_content)
        node.refresh_instruct_content(keys=all_keys,changed_contents=changed_contents)
    else:
        changed_contents = []
        for key in keys2change:
            changed_content = level_map[level][key]
            # if not changed_content:
            #     logger.error(f'fail to get changed_content for key:{key}, using original content!')
            #     changed_content = get_content_from_context(context,key)
            changed_contents.append(changed_content)
        node.refresh_instruct_content(keys=keys2change,changed_contents=changed_contents)
    # logger.error(f'contentvalue:{changed_contents}')
    
    logger.error(f'after intervent, context:{node.instruct_content.model_dump_json()}')
    return node



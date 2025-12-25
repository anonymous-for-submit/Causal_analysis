# -*- coding: utf-8 -*-
import sys
sys.path.append('/home/zlyuaj/Causal/MetaGPT')
sys.path.append('/home/zlyuaj/Causal/MetaGPT/metagpt')
import asyncio
from pathlib import Path
import os
import copy
import json
import argparse
import tqdm
import numpy as np
import time
import random
from datasets import load_dataset, load_from_disk
from collections import defaultdict
from evaluate_result import evaluate_one, evaluate_one_codecontest, evaluate_one_MBPP, evaluate_one_CoderEval
from concurrent.futures import as_completed, ProcessPoolExecutor
from metagpt.actions.intervent import clear_changed_contents, modified_changed_contents, print_prd
from openai import OpenAI

# 参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--output_path', type=str, default='./output/')
parser.add_argument('--input_path', type=str, default='data/HumanEval_test_case_ET.jsonl')
parser.add_argument('--dataset', type=str, default='HumanEval')
parser.add_argument('--output_file_name', type=str, default='repair_test')
parser.add_argument('--workspace', type=str, default='workspace_repair')
parser.add_argument('--num_generate', type=int, default=1)
parser.add_argument('--parallel', type=int, default=1)
parser.add_argument('--model', type=str, default='gpt-4o-mini-ca')
parser.add_argument('--run_generate', type=int, default=1)
parser.add_argument('--run_evaluate', type=int, default=1)
parser.add_argument('--do_intervent', type=int, default=1)
parser.add_argument('--begin_idx', type=int, default=0)
parser.add_argument('--end_idx', type=int, default=1000)
parser.add_argument('--original_result_workspace', type=str, default='')
parser.add_argument('--original_result_path', type=str, default='')
parser.add_argument('--repair_feature_num', type=int, default=5)
parser.add_argument('--codereval_data_path', type=str, default='/home/zlyuaj/Causal/MetaGPT/data/CoderEval4Python.json')

parser.add_argument('--add_monitor', type=int, default=0)
parser.add_argument('--repair_plan', type=int, default=0)
parser.add_argument('--repair_code', type=int, default=0)
parser.add_argument('--run_multi_gen', type=int, default=0)
parser.add_argument('--repair_prompt_num', type=int, default=0)


args = parser.parse_args()

# 初始化OpenAI客户端用于调用GPT-4o-mini

client_4o_mini = OpenAI(
        api_key = "sk-FRQxdGxCMDSPoogN0SgdGGm4IEfv3uMjUFTtgepRNC7bnxO8",
        base_url = "https://api.chatanywhere.tech/v1"
        )
client=client_4o_mini
def generate_repo(
    idea,
    investment=3.0,
    n_round=5,
    code_review=True,
    run_tests=False,
    implement=True,
    project_name="",
    inc=False,
    project_path="",
    reqa_file="",
    max_auto_summarize_code=0,
    recover_path=None,
    args=None
    ):
    """Run the startup logic. Can be called from CLI or other Python scripts."""
    # return 
    from metagpt.config2 import config
    from metagpt.context import Context
    from metagpt.roles import (
        Architect,
        Engineer,
        ProductManager,
        ProjectManager,
        QaEngineer,
    )
    from metagpt.team import Team
    if config.agentops_api_key != "":
        agentops.init(config.agentops_api_key, tags=["software_company"])
    print('in generating repo')
    config.set_args(args=args)
    config.update_via_cli(project_path, project_name, inc, reqa_file, max_auto_summarize_code)
    ctx = Context(config=config,args=args)
    ctx.set_args(args)

    if not recover_path:
        # 建立公司，并招募员工
        company = Team(context=ctx)
        # 先找三个员工
        company.hire(
            [
                #再role的初始化函数里就做了llm生成
                ProductManager(args=args),
                Architect(args=args),
                ProjectManager(args=args),
            ]
        )

        if implement or code_review:
            company.hire([Engineer(args=args,n_borg=5, use_code_review=code_review)])

        if run_tests:
            company.hire([QaEngineer(args=args)])
    else:
        stg_path = Path(recover_path)
        if not stg_path.exists() or not str(stg_path).endswith("team"):
            raise FileNotFoundError(f"{recover_path} not exists or not endswith `team`")

        company = Team.deserialize(stg_path=stg_path, context=ctx)
        idea = company.idea
    # 做项目评估，仅从budget角度

    company.invest(investment)
    # 根据输入的idea进行软件开发
    
    company.run_project(idea)
    asyncio.run(company.run(args=args,n_round=n_round))

    # if config.agentops_api_key != "":
    #     agentops.end_session("Success")

    return ctx.repo


def startup(
    idea: str = 'write a python function to count 1-100',
    investment: float = 3.0,
    n_round: int = 5,
    code_review: bool = True,
    run_tests: bool = False,
    implement: bool = True,
    project_name: str = "",
    inc: bool = False,
    project_path: str = "",
    reqa_file: str ="",
    max_auto_summarize_code: int = 0,
    recover_path: str = None,
    init_config: bool = False,
    args = None,
    ):
    """Run a startup. Be a boss."""

    if idea is None:
        typer.echo("Missing argument 'IDEA'. Run 'metagpt --help' for more information.")
        raise typer.Exit()
    # print(idea)
    # print('coming to generating repo')
    # print(args)
    return generate_repo(
        idea,
        investment,
        n_round,
        code_review,
        run_tests,
        implement,
        project_name,
        inc,
        project_path,
        reqa_file,
        max_auto_summarize_code,
        recover_path,
        args=args,
    )


def extract_code_from_repo(file_path):
    files=os.listdir(file_path)
    num_py_files = len(files)
    if num_py_files==0:
        return ''
    # print(files)
    file_name = files[0]
    if 'main' in file_name and num_py_files>1:
        file_name = files[1]
    # print(file_name)
    sourse=''
    code=''
    if os.path.exists(file_path+ '/'+file_name) and os.path.isfile(file_path+ '/'+file_name) and file_name.endswith('.py'):
        with open(file_path+ '/'+file_name,'r') as f:
            code = f.read()
        return code
def extract_features_from_repo(file_path):
    prd_path = file_path +'/docs/prd'
    system_design_path = file_path +'/docs/system_design'
    task_path = file_path +'/docs/task'
    prd = ''
    system_design=''
    task=''
    if os.path.exists(prd_path) and os.listdir(prd_path):
        path = prd_path + '/'+os.listdir(prd_path)[0]
        with open(path,'r') as f:
            try:
                prd=json.load(f)
            except:
                prd = ''
    else:
        prd = ''

    if os.path.exists(system_design_path) and os.listdir(system_design_path):
        path = system_design_path + '/'+os.listdir(system_design_path)[0]
        with open(path,'r') as f:
            try:
                system_design=json.load(f)
            except:
                system_design = ''
    else:
        system_design = ''
    if os.path.exists(task_path) and os.listdir(task_path):
        path = task_path + '/'+os.listdir(task_path)[0]
        with open(path,'r') as f:
            try:
                task=json.load(f)
            except:
                task = ''
    else:
        task = ''
    return prd,system_design,task

def extract_plan_from_repo(file_path):
    prd_path = file_path +'/docs/prd'
    system_design_path = file_path +'/docs/system_design'
    plan = ''
    RequirementAnalysis,RequirementPool,ImplementationApproach = '','',''
    if not os.path.exists(prd_path) or not os.listdir(prd_path):
        RequirementAnalysis,RequirementPool='',''
    else:
        path = prd_path + '/'+os.listdir(prd_path)[0]
        with open(path,'r') as f:
            try:
                prd=json.load(f)
                RequirementAnalysis = prd['Requirement Analysis']
                RequirementPool = prd['Requirement Pool']
            except:
                pass
    if not os.path.exists(system_design_path) or not os.listdir(system_design_path):
        ImplementationApproach=''
    else:
        path = system_design_path + '/'+os.listdir(system_design_path)[0]
        with open(path,'r') as f:
            try:
                system_design=json.load(f)
                ImplementationApproach = system_design['Implementation approach']
            except:
                pass 
    return RequirementAnalysis,RequirementPool,ImplementationApproach
def delete_repo(file_path):
    import shutil
    shutil.rmtree(file_path) 

def format_plan(RA,RP,IA):
    plan = ''
    if RA:
        plan+=f'requirement analysis:\n{RA}\n'
    if RP:
        plan+='requirement pool:\n'
        for req in RP:
            plan+='- '+req[1]+'\n'
    plan+=IA+'\n'
    return plan
   
def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_data_from_directory(path):
    prd_list = []
    system_design_list = []
    task_list = []

    # 获取目录下的所有文件夹，并按名称排序
    directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    directories = sorted(directories, key=lambda x: int(x.split('_')[-2]))
    # print(directories)


    for directory in directories:
        # 构造每个需要读取的json文件的路径


        prd_path = os.path.join(path, directory, 'docs', 'prd')
        system_design_path = os.path.join(path, directory, 'docs', 'system_design')
        task_path = os.path.join(path, directory, 'docs', 'task')

        # 读取prd json文件
        prd_files = [f for f in os.listdir(prd_path) if f.endswith('.json')]
        if prd_files:
            prd_data = read_json_file(os.path.join(prd_path, prd_files[0]))
            prd_list.append(prd_data)
        else:
            print('no prd file in ', prd_path)
            # raise FileNotFoundError(f'No prd file found in {prd_path}')
            prd_list.append({})

        # 读取system_design json文件
        system_design_files = [f for f in os.listdir(system_design_path) if f.endswith('.json')]
        if system_design_files:
            system_design_data = read_json_file(os.path.join(system_design_path, system_design_files[0]))
            system_design_list.append(system_design_data)
        else:
            print('no system_design file in ', system_design_path)
            # raise FileNotFoundError(f'No system_design file found in {system_design_path}')
            system_design_list.append({})
        # 读取task json文件
        task_files = [f for f in os.listdir(task_path) if f.endswith('.json')]
        if task_files:
            task_data = read_json_file(os.path.join(task_path, task_files[0]))
            task_list.append(task_data)
        else:
            print('no task file in ', task_path)
            # raise FileNotFoundError(f'No task file found in {task_path}')
            task_list.append({})

    return prd_list, system_design_list, task_list
def get_data_from_jsonl(path):
    prd_list = []
    system_design_list = []
    task_list = []

    # 构造jsonl文件路径
    jsonl_file_path = path + ".jsonl"

    if not os.path.exists(jsonl_file_path):
        raise FileNotFoundError(f"No JSONL file found at {jsonl_file_path}")

    data_list = []

    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 解析每一行的JSON对象
            data = json.loads(line)
            data_list.append(data)

    # 根据 file_name 中的数字进行排序
    data_list.sort(key=lambda x: int(x['file_name'].split('_')[1]))

    for data in data_list:
        # 提取 prd, system_design, task 的数据
        prd = data.get('prd', {})
        system_design = data.get('system_design', {})
        if not system_design:
            system_design = data.get('design', {})
        task = data.get('task', {})

        # 添加到相应的列表中
        prd_list.append(prd)
        system_design_list.append(system_design)
        task_list.append(task)

    return prd_list, system_design_list, task_list

class FeatureNode:
    def __init__(self, id, level, key):
        self.id = id
        self.level = level
        self.key = key
        self.parent = []
        self.children = []
        self.is_cause = False

    def build_str(self):
        return f'{self.level}_{self.key}'
coding_prompt=''
def generate(method_name,generate_ids,task):
    try:
        # 进行分工流程在这里，输入了prompt为intent
        futures=[]
        regenerate = []
        if args.parallel==1:
            split_para=1
            for __ in range(split_para):
                with ProcessPoolExecutor() as executor:
                    for cnt in generate_ids:
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        new_method_name = method_name
                        idea = coding_prompt+task['prompt']
                        # print(f'idea: {idea}')
                        args_dict = vars(args)
                        future= executor.submit(startup,idea=idea,project_name=new_method_name,args=args_dict)
                    
                        futures.append(future)
                results=[]
                for cnt, future in enumerate(as_completed(futures)):
                    # print(future.result())
                    results.append(future.result())
                    new_method_name = method_name
                    file_path = '/home/zlyuaj/Causal/MetaGPT/{}/{}/{}'.format(args.workspace,new_method_name,new_method_name)
                    if not os.path.exists(file_path):
                        regenerate.append(cnt)

                return regenerate
        else:
            for cnt in generate_ids:
                # new_loop = asyncio.new_event_loop()
                # asyncio.set_event_loop(new_loop)
                new_method_name = method_name
                args_dict = vars(args)
                startup(idea=coding_prompt+task['prompt'],project_name=new_method_name,args=args_dict)
                
                # session_historys.append(session_history)
    except Exception as e:
        # raise NotImplementedError
        print(e)
        return generate_ids

def evaluate(task,new_method_name):
    file_path = '/home/zlyuaj/Causal/MetaGPT/{}/{}'.format(args.workspace,new_method_name)
    code_file_path = '/home/zlyuaj/Causal/MetaGPT/{}/{}/{}'.format(args.workspace,new_method_name,new_method_name)
    # print(os.path.exists('/home/zlyuaj/Causal/MetaGPT/workspace_gpt-4o-mini-ca_humaneval_4o-mini/humaneval_0_no_prd_Language,prd_Programming Language/humaneval_0_no_prd_Language,prd_Programming Language'))
    # print(os.path.exists('/home/zlyuaj/Causal/MetaGPT/workspace_gpt-4o-mini-ca_humaneval_4o-mini/humaneval_0_prd_Language,prd_Programming Language/humaneval_0_prd_Language,prd_Programming Language'))
    if not os.path.exists(code_file_path):
        print('no code_file_path')
        print(code_file_path)
        return False
    code = extract_code_from_repo(code_file_path)
    if not code:
        code = ''   
    code = 'from typing import List\n'+code
    import re
    def remove_input_content(code):
            # 使用正则表达式替换 input() 中的内容
        code = re.sub(r'input\([^)]*\)', 'input()', code)
        return code
    codes = []
    code = remove_input_content(code)
    codes.append(code)
    task['completions'] = codes
    # task['code'] = code

    # print('-'*100)
    if 'human' in args.dataset:
        score, passes,passAt1, passAt10= evaluate_one(task,args.num_generate)
    elif 'mbpp' in args.dataset:
        score, passes,passAt1, passAt10= evaluate_one_MBPP(task,args.num_generate)
    elif 'codecontest' in args.dataset:
        score, passes,passAt1, passAt10= evaluate_one_codecontest(task,args.num_generate)
    elif 'CoderEval' in args.dataset:
        passAt1,passes= evaluate_one_CoderEval(task,args.num_generate)

    task['pass'] = passAt1
    task['pass_num'] = passes

    with open(file_path + '/eval_result.txt','w+') as eval_f:
        eval_f.write(f'{passAt1}\n')
    return passAt1


def print_node(node):
    return
def check_equal(a,b):
    if a.id==b.id and a.level == b.level and a.key ==b.key:
        return True
    return False
def generate_feature_str(features):
    feature_str = ''
    for f in features:
        feature_str+=f.build_str()
        feature_str+=','
    feature_str=feature_str[:-1]
    return feature_str


def contain_current_cause(features,causes):
    for cause in causes:
        # 每一个cause都不被features包含,即至少有一个feature不在cause中
        res = True
        for f in cause:
            if f not in features:
                res = False
                break
        if res:
            return True
    return False





def read_feature_rank(dataset, model):
    """读取feature rank"""
    base_path = f'/home/zlyuaj/Causal/MetaGPT/output/{dataset}'
    feature_rank_file = f'{base_path}/results-{dataset}_{model}_new_modified_intervent/new_blame_change.jsonl'
    with open(feature_rank_file, 'r') as f:
        feature_rank = [json.loads(line) for line in f]
        feature_rank = feature_rank[0]
        feature_rank = list(feature_rank.keys())
        feature_rank.remove('task_File list')
    return feature_rank

def get_correct_code(task, dataset):
    """从数据集中获取正确代码"""
    if dataset == 'humaneval':
        return task.get('canonical_solution', '')
    elif dataset == 'mbpp':
        return task.get('code', '')
    elif dataset == 'CoderEval':
        # 需要从CoderEval数据集中读取
        question_id = task.get('question_id', '')
        codereval_data_path = args.codereval_data_path
        print(codereval_data_path)
        with open(codereval_data_path, 'r') as f:
            codereval_data = json.load(f)
        # print(len(codereval_data))
        # print(question_id)
        
        codereval_data = codereval_data['RECORDS']
        # print(question_id in [i['_id'] for i in codereval_data])
        for item in codereval_data:
            if item.get('_id') == question_id:
                return item.get('code', '')
        return ''
    return ''

def repair_with_llm(prompt, correct_code, field_type):
    """使用LLM修复指定字段"""
    if field_type == 'design_Implementation approach':
        system_prompt = """You are an expert software architect. Given a user's requirement and the correct implementation code, 
please generate an improved "Implementation approach" that would help guide the development towards the correct solution.

The Implementation approach should interpret the correct code step by step to provide user a clear approach without any useless or redundent information.

Output ONLY the Implementation approach text, no additional formatting or explanation."""
        
        user_prompt = f"""Requirement: {prompt}

Correct Implementation Code:
```python
{correct_code}
```

Based on the requirement and correct code above, generate an Implementation approach that would guide developers to create similar correct code. the Implementation approach should interpret the correct code step by step and attach the corresponding code snippet in the correct code. Be clear and avoid redundent information. 
"""

    else:  # prd_Original Requirements
        system_prompt = """You are an expert product manager. Given a user's original requirement and the correct implementation code,
please generate improved "Original Requirements" that would better capture what needs to be built.

The Original Requirements should:
- Be clear and specific
- Capture the essential functionality shown in the correct code
- Be concise but comprehensive

Output ONLY the Original Requirements text, no additional formatting or explanation."""
        
        user_prompt = f"""Original Requirement: {prompt}

Correct Implementation Code:
```python
{correct_code}
```

Based on the original requirement and correct code above, generate improved Original Requirements that better specify what needs to be built. Keep the signature unchanged"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini-ca",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM repair failed: {e}")
        return None

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_data_from_directory(path):
    """从目录读取数据"""
    prd_list = []
    system_design_list = []
    task_list = []

    directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    directories = sorted(directories, key=lambda x: int(x.split('_')[-2]))

    for directory in directories:
        prd_path = os.path.join(path, directory, 'docs', 'prd')
        system_design_path = os.path.join(path, directory, 'docs', 'system_design')
        task_path = os.path.join(path, directory, 'docs', 'task')

        prd_files = [f for f in os.listdir(prd_path) if f.endswith('.json')]
        if prd_files:
            prd_data = read_json_file(os.path.join(prd_path, prd_files[0]))
            prd_list.append(prd_data)
        else:
            prd_list.append({})

        system_design_files = [f for f in os.listdir(system_design_path) if f.endswith('.json')]
        if system_design_files:
            system_design_data = read_json_file(os.path.join(system_design_path, system_design_files[0]))
            system_design_list.append(system_design_data)
        else:
            system_design_list.append({})

        task_files = [f for f in os.listdir(task_path) if f.endswith('.json')]
        if task_files:
            task_data = read_json_file(os.path.join(task_path, task_files[0]))
            task_list.append(task_data)
        else:
            task_list.append({})

    return prd_list, system_design_list, task_list

def get_data_from_jsonl(path):
    """从jsonl读取数据"""
    prd_list = []
    system_design_list = []
    task_list = []

    jsonl_file_path = path + ".jsonl"
    if not os.path.exists(jsonl_file_path):
        raise FileNotFoundError(f"No JSONL file found at {jsonl_file_path}")

    data_list = []
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            data_list.append(data)

    data_list.sort(key=lambda x: int(x['file_name'].split('_')[1]))

    for data in data_list:
        prd = data.get('prd', {})
        system_design = data.get('system_design', {}) or data.get('design', {})
        task = data.get('task', {})

        prd_list.append(prd)
        system_design_list.append(system_design)
        task_list.append(task)

    return prd_list, system_design_list, task_list

class FeatureNode:
    def __init__(self, id, level, key):
        self.id = id
        self.level = level
        self.key = key

    def build_str(self):
        return f'{self.level}_{self.key}'


if __name__ == '__main__':
    # 创建输出目录
    initial_output_path = args.output_path
    if not os.path.exists(initial_output_path):
        os.mkdir(initial_output_path)
    
    args.output_path = initial_output_path + 'results-' + args.output_file_name + '/'
    x = 2
    while os.path.exists(args.output_path):
        args.output_path = initial_output_path + 'results-' + args.output_file_name + '_' + str(x) + '/'
        x += 1
    os.mkdir(args.output_path)
    
    print(f"Output path: {args.output_path}")

    # 读取feature rank
    feature_rank = read_feature_rank(args.dataset, args.model)
    print(f"Feature rank: {feature_rank}")
    # feature_rank.reverse()  # 反转以获得最重要的features

    # 获取top features
    top_features_str = feature_rank[:args.repair_feature_num]
    print(f"Top {args.repair_feature_num} features: {top_features_str}")

    # 创建feature映射
    levels = ['prd', 'design', 'task']
    level_key_maps = {
        'prd': ['Language', 'Programming Language', 'Original Requirements', 'Product Goals', 'User Stories',
                'Competitive Analysis', 'Competitive Quadrant Chart', 'Requirement Analysis', 'Requirement Pool',
                'UI Design draft', 'Anything UNCLEAR'],
        'design': ['Implementation approach', 'File list', 'Data structures and interfaces', 'Program call flow',
                   'Anything UNCLEAR'],
        'task': ['Required packages', 'Required Other language third-party packages', 'Logic Analysis', 'File list',
                 'Full API spec', 'Shared Knowledge', 'Anything UNCLEAR']
    }

    feature_int_map = {}
    cnt = 0
    feature_nodes = []
    for level in levels:
        for i in range(len(level_key_maps[level])):
            new_node = FeatureNode(cnt, level, level_key_maps[level][i])
            feature_nodes.append(new_node)
            feature_int_map[new_node.build_str()] = cnt
            cnt += 1

    # 判断是否需要修复design_Implementation approach
    repair_design = False
    design_impl_node = None
    for feature_str in top_features_str:
        if 'design_Implementation approach' in feature_str:
            repair_design = True
            feature_id = feature_int_map['design_Implementation approach']
            design_impl_node = feature_nodes[feature_id]
            break

    # 如果不修复design，则修复prd_Original Requirements
    prd_orig_req_node = None
    if not repair_design:
        feature_id = feature_int_map['prd_Original Requirements']
        prd_orig_req_node = feature_nodes[feature_id]

    print(f"Repair strategy: {'design_Implementation approach' if repair_design else 'prd_Original Requirements'}")

    # 加载数据集
    INPUTPATH = args.input_path
    loaded_dataset = []
    with open(INPUTPATH, 'r') as f:
        loaded_dataset = [json.loads(line) for line in f]

    # 加载原始结果
    original_passes = []
    with open(args.original_result_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            original_passes.append(data['pass'])

    # 加载原始特征数据
    all_original_prd = []
    all_original_design = []
    all_original_task = []
    if os.path.exists(args.original_result_workspace):
        all_original_prd, all_original_design, all_original_task = get_data_from_directory(args.original_result_workspace)
    else:
        all_original_prd, all_original_design, all_original_task = get_data_from_jsonl(args.original_result_workspace)

    args.all_original_prd = all_original_prd
    args.all_original_design = all_original_design
    args.all_original_task = all_original_task

    # 创建修改后的数据副本
    all_modified_prd = copy.deepcopy(all_original_prd)
    all_modified_design = copy.deepcopy(all_original_design)
    all_modified_task = copy.deepcopy(all_original_task)

    # 统计修复结果
    total_errors = 0
    repaired_success = 0
    repair_results = []

    # 对每个错误的样本进行修复
    for idx, task in enumerate(loaded_dataset):
        if args.begin_idx > 0 and idx < args.begin_idx:
            continue
        if args.end_idx <= 1000 and idx > args.end_idx:
            break
        
        # if idx>3:
            # break
        # 只修复原本失败的样本
        if original_passes[idx]:
            continue

        total_errors += 1
        print(f"\n{'='*50}")
        print(f"Repairing task {idx} (Error #{total_errors})")
        print(f"{'='*50}")

        if 'prompt' not in task.keys():
            if 'description' in task.keys():
                task['prompt'] = task['description']
            elif 'text' in task.keys():
                task['prompt'] = task['text']
            elif 'input' in task.keys():
                task['prompt'] = task['input']

        prompt = task['prompt']
        correct_code = get_correct_code(task, args.dataset)

        if not correct_code:
            print(f"Warning: No correct code found for task {idx}")
            continue

        # 使用LLM生成修复后的内容
        if repair_design:
            print("Repairing design_Implementation approach...")
            repaired_content = repair_with_llm(prompt, correct_code, 'design_Implementation approach')
            if repaired_content:
                all_modified_design[idx]['Implementation approach'] = repaired_content
                print(f"Repaired Implementation approach: {repaired_content[:100]}...")
        else:
            print("Repairing prd_Original Requirements...")
            repaired_content = repair_with_llm(prompt, correct_code, 'prd_Original Requirements')
            if repaired_content:
                all_modified_prd[idx]['Original Requirements'] = repaired_content
                print(f"Repaired Original Requirements: {repaired_content[:100]}...")

        if not repaired_content:
            print(f"Failed to repair task {idx}")
            continue

        # print(f'repair_design: {repair_design}')
        # print('repaired_content:',repaired_content)

        # 设置intervene参数
        feature_node = ''
        if repair_design:
            args.levels = ['design']
            args.keys = ['Implementation approach']
            feature_node = design_impl_node
            feature_str = 'design_Implementation approach'
        else:
            args.levels = ['prd']
            args.keys = ['Original Requirements']
            feature_node = prd_orig_req_node
            feature_str = 'prd_Original Requirements'

        args.cur_id = idx

        # 调用modified_changed_contents进行intervene
        modified_changed_contents(all_modified_prd, all_modified_design, all_modified_task, idx)

        if args.run_generate==1:
            levels = [feature_node.level]
            keys = [feature_node.key]
            args.levels = levels
            args.keys = keys
            generate_ids=[feature_str]
            method_name = args.dataset + '_' +  str(idx)
            new_method_name = method_name
            path = '/home/zlyuaj/Causal/MetaGPT/{}/{}/{}'.format(args.workspace,new_method_name,new_method_name)
            has_py_file = any(filename.endswith('.py') for filename in os.listdir(path)) if os.path.exists(path) else False
            max_try = 3
            while max_try>0 and not os.path.exists(path) and not has_py_file:
                max_try-=1
                generate(method_name,generate_ids,task)
                has_py_file = any(filename.endswith('.py') for filename in os.listdir(path)) if os.path.exists(path) else False
                

        result = False

        # 评估修复结果
        if args.run_evaluate == 1:
            try:
                print('evaluating ...')
                method_name = args.dataset + '_' +  str(idx)
                new_method_name = method_name
                result = evaluate(task,new_method_name)
                
                if result:
                    repaired_success += 1
                
                repair_results.append({
                    'task_idx': idx,
                    'original_pass': False,
                    'repaired_pass': result,
                    'repair_field': 'design_Implementation approach' if repair_design else 'prd_Original Requirements',
                    'repaired_content': repaired_content
                })
            except Exception as e:
                print(f"Evaluation failed for task {idx}: {e}")
                repair_results.append({
                    'task_idx': idx,
                    'original_pass': False,
                    'repaired_pass': False,
                    'repair_field': 'design_Implementation approach' if repair_design else 'prd_Original Requirements',
                    'error': str(e)
                })

    # 计算并输出修复后的pass rate
    repair_pass_rate = repaired_success / total_errors if total_errors > 0 else 0
    
    print(f"\n{'='*50}")
    print("REPAIR RESULTS")
    print(f"{'='*50}")
    print(f"Total errors: {total_errors}")
    print(f"Successfully repaired: {repaired_success}")
    print(f"Repair pass rate: {repair_pass_rate:.2%}")
    print(f"{'='*50}\n")

    # 保存结果
    results_file = os.path.join(args.output_path, 'repair_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'total_errors': total_errors,
            'repaired_success': repaired_success,
            'repair_pass_rate': repair_pass_rate,
            'repair_strategy': 'design_Implementation approach' if repair_design else 'prd_Original Requirements',
            'top_features': top_features_str,
            'detailed_results': repair_results
        }, f, indent=2)

    print(f"Results saved to {results_file}")



# def repair_with_llm(prompt, correct_code, field_type):
#     """使用LLM修复指定字段"""
#     if field_type == 'design_Implementation approach':
#         system_prompt = """You are an expert software architect. Given a user's requirement and the correct implementation code, 
# please generate an improved "Implementation approach" that would help guide the development towards the correct solution.

# The Implementation approach should:
# - Analyze the difficult points of the requirements
# - Select appropriate frameworks and methods
# - Provide clear technical guidance

# Output ONLY the Implementation approach text, no additional formatting or explanation."""
        
#         user_prompt = f"""Requirement: {prompt}

# Correct Implementation Code:
# ```python
# {correct_code}
# ```

# Based on the requirement and correct code above, generate an Implementation approach that would guide developers to create similar correct code. """

#     else:  # prd_Original Requirements
#         system_prompt = """You are an expert product manager. Given a user's original requirement and the correct implementation code,
# please generate improved "Original Requirements" that would better capture what needs to be built.

# The Original Requirements should:
# - Be clear and specific
# - Capture the essential functionality shown in the correct code
# - Be concise but comprehensive

# Output ONLY the Original Requirements text, no additional formatting or explanation."""
        
#         user_prompt = f"""Original Requirement: {prompt}

# Correct Implementation Code:
# ```python
# {correct_code}
# ```

# Based on the original requirement and correct code above, generate improved Original Requirements that better specify what needs to be built. Keep the signature unchanged"""

#     try:
#         response = client.chat.completions.create(
#             model="gpt-4o-mini-ca",
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt}
#             ],
#             temperature=0.7,
#             max_tokens=1000
#         )
#         return response.choices[0].message.content.strip()
#     except Exception as e:
#         print(f"LLM repair failed: {e}")
#         return None


#     if field_type == 'design_Implementation approach':
#         system_prompt = """You are an expert software architect. Given a user's requirement and the correct implementation code, 
# please generate an improved "Implementation approach" that would help guide the development towards the correct solution.

# The Implementation approach should interpret the correct code step by step to provide user a clear approach without any useless or redundent information.

# Output ONLY the Implementation approach text, no additional formatting or explanation."""
        
#         user_prompt = f"""Requirement: {prompt}

# Correct Implementation Code:
# ```python
# {correct_code}
# ```

# Based on the requirement and correct code above, generate an Implementation approach that would guide developers to create similar correct code. the Implementation approach should interpret the correct code step by step and be simple to avoid redundent information."""

#     else:  # prd_Original Requirements
#         system_prompt = """You are an expert product manager. Given a user's original requirement and the correct implementation code,
# please generate improved "Original Requirements" that would better capture what needs to be built.

# The Original Requirements should:
# - Be clear and specific
# - Capture the essential functionality shown in the correct code
# - Be concise but comprehensive

# Output ONLY the Original Requirements text, no additional formatting or explanation."""
        
#         user_prompt = f"""Original Requirement: {prompt}

# Correct Implementation Code:
# ```python
# {correct_code}
# ```

# Based on the original requirement and correct code above, generate improved Original Requirements that better specify what needs to be built. Keep the signature unchanged"""
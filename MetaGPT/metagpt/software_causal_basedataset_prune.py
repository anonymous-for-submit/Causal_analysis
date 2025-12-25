
# -*- coding: utf-8 -*-
import sys
sys.path.append('/home/zlyuaj/Causal/MetaGPT')
sys.path.append('/home/zlyuaj/Causal/MetaGPT/metagpt')
import asyncio
from pathlib import Path

import agentops
import typer

import torch
import torch.nn.functional as F
from const import CONFIG_ROOT
from metagpt.utils.project_repo import ProjectRepo
import os
import copy
import json
import argparse
import tqdm
import numpy as np
import time
import random
import yaml
from metagpt.logs import logger
from itertools import combinations
import itertools
# import torch
# import torch.nn.functional as F
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from datasets import load_dataset, load_from_disk
from collections import defaultdict
from evaluate_result import evaluate_one,evaluate_one_codecontest,evaluate_one_MBPP,evaluate_one_CoderEval
from concurrent.futures import as_completed, ProcessPoolExecutor
from metagpt.actions.intervent import clear_changed_contents
import multiprocessing
from _utils import prompt_split_humaneval
parser = argparse.ArgumentParser()
parser.add_argument('--output_path', type=str, default='./output/')
parser.add_argument('--input_path', type=str, default='data/HumanEval_test_case_ET.jsonl')
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
parser.add_argument('--begin_idx', type=int, default=0)
parser.add_argument('--end_idx', type=int, default=1000)
parser.add_argument('--original_result_workspace', type=str, default='')
parser.add_argument('--sim_threshold', type=float, default=0.5)
parser.add_argument('--build_feature_graph', type=int, default=1)
parser.add_argument('--max_run', type=int, default=100)
parser.add_argument('--generate_len1', type=int, default=1)
parser.add_argument('--generate_len2', type=int, default=1)
parser.add_argument('--early_stop', type=int, default=1)
parser.add_argument('--early_stop_threshold', type=int, default=5)
parser.add_argument('--max_len_cause', type=int, default=5)
parser.add_argument('--prune_feature', type=int, default=0)
parser.add_argument('--empty_intervention', type=int, default=1)
parser.add_argument('--prune_feature_num', type=int, default=5)

parser.add_argument('--do_intervent', type=int, default=0)
parser.add_argument('--run_certain_level', type=str, default='')
parser.add_argument('--original_result_path', type=str, default='')

parser.add_argument('--add_monitor', type=int, default=0)
parser.add_argument('--repair_plan', type=int, default=0)
parser.add_argument('--repair_code', type=int, default=0)
parser.add_argument('--run_multi_gen', type=int, default=0)
parser.add_argument('--repair_prompt_num', type=int, default=0)

args = parser.parse_args()
# from sentence_transformers import SentenceTransformer, util
# semantic_model = SentenceTransformer('paraphrase-MiniLM-L6-v2',device='cuda:{}'.format(0))
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
    directories = sorted(directories, key=lambda x: int(x.split('_')[1]))
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
def calc_sim(semantic_model, original_result, new_result):
    # print('in calc sim')
    # print(original_result)
    # print(new_result)
    original_result = str(original_result)
    new_result = str(new_result)
    original_result=semantic_model.encode(original_result, convert_to_tensor=True)
    new_result=semantic_model.encode(new_result, convert_to_tensor=True)
    similarity = util.cos_sim(original_result, new_result)
    similarity = float(similarity)
    # print(similarity)
    return similarity
def calc_changed_features(new_features, original_features):
    # print('-'*100)
    # print('in calc changed features')
    # print(new_features)
    # print('-'*100)
    # print(original_features)
    # print('-'*100)
    changed_features = []
    prd,system_design,task = new_features
    o_prd, o_system_design, o_task = original_features
    for key in o_prd:
        original_result = o_prd[key]
        new_result = prd[key]
        similarity = calc_sim(semantic_model, original_result, new_result)
        if similarity < args.sim_threshold:
            changed_features.append(['prd',key])
    for key in o_system_design:
        original_result = o_system_design[key]
        print()
        new_result = system_design[key]
        similarity = calc_sim(semantic_model, original_result, new_result)
        if similarity < args.sim_threshold:
            changed_features.append(['design',key])
    for key in o_task:
        original_result = o_task[key]
        new_result = task[key]
        similarity = calc_sim(semantic_model, original_result, new_result)
        if similarity < args.sim_threshold:
            changed_features.append(['task',key])
    return changed_features
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

def read_feature_rank(dataset,model):
    base_path  =f'/home/zlyuaj/Causal/MetaGPT/output/{dataset}'

    feature_rank_file = f'{base_path}/results-{dataset}_{model}_new_modified_intervent/new_blame_change.jsonl'
    with open(feature_rank_file,'r') as f:
        # 读取jsonl文件
        feature_rank = [json.loads(line) for line in f]
        feature_rank = feature_rank[0]
        feature_rank = list(feature_rank.keys())
        feature_rank.remove('task_File list')
    return feature_rank

if __name__ == '__main__':
    coding_prompt=''

    initial_output_path=args.output_path
    print(os.path.exists(initial_output_path))
    if not os.path.exists(initial_output_path):
        os.mkdir(initial_output_path)
    print(os.path.exists(initial_output_path))
    args.output_path=initial_output_path+'results-'+args.output_file_name+'/'
    x=2
    while os.path.exists(args.output_path):
        args.output_path=initial_output_path+'results-'+args.output_file_name+'_'+str(x)+'/'
        x+=1
    os.mkdir(args.output_path)
    print(args.output_path)
    print(args)
    print(type(args))




    # load dataset
    INPUTPATH=args.input_path
    original_passes = []
    loaded_dataset=[]
    with open(INPUTPATH, 'r') as f:
        # 导入输出
        loaded_dataset = [json.loads(line) for line in f]


    if args.empty_intervention==1:
        clear_changed_contents()

    # print(len(loaded_dataset))
    # passAt10s=[]


    initial_seed = loaded_dataset
    initial_seed_num=len(loaded_dataset)


    # text, code, task_id, test_list, entry_point
    levels = ['prd','design','task']
    # levels = ['prd','design']
    level_key_maps={
        'prd':['Language','Programming Language','Original Requirements','Product Goals','User Stories','Competitive Analysis','Competitive Quadrant Chart','Requirement Analysis','Requirement Pool','UI Design draft','Anything UNCLEAR'],
        'design':['Implementation approach','File list','Data structures and interfaces','Program call flow','Anything UNCLEAR'],
        'task':['Required packages','Required Other language third-party packages','Logic Analysis','File list','Full API spec','Shared Knowledge','Anything UNCLEAR']
    }
    # 11 5 7
    
    feature_int_map={}
    cnt=0
    feature_nodes = []
    for level in levels:
        for i in range(len(level_key_maps[level])):
            new_node = FeatureNode(cnt, level, level_key_maps[level][i])
            feature_nodes.append(new_node)
            feature_int_map[new_node.build_str()] = cnt
            cnt+=1
    fail_list = []
    total_fail_list = defaultdict(list)
    blame_map = defaultdict(float)
    
    feature_rank = read_feature_rank(args.dataset,args.model)
    print(feature_rank)
    feature_rank.reverse()

    pruned_features_str = feature_rank[:args.prune_feature_num]
    pruned_features = []
    for feature in pruned_features_str:
        feature_name = feature
        level = feature_name.split('_')[0]
        key = '_'.join(feature_name.split('_')[1:])
        feature_id = feature_int_map[f'{level}_{key}']
        pruned_features.append(feature_nodes[feature_id])
    print('pruned features:')
    for feature_node in pruned_features:
        print(f'{feature_node.level}_{feature_node.key}')

    for idx,task in enumerate(loaded_dataset):
        if args.begin_idx>0 and idx<args.begin_idx:
            continue
        if args.end_idx<=1000 and idx>args.end_idx:
            break

        print('-'*10+'executing task: {}'.format(idx)+'-'*10)

        
        for feature_node in feature_nodes:
            feature_node.is_cause=False
            feature_node.children=[]
            feature_node.parent=[]


        if 'prompt' not in task.keys():
            if 'description' in task.keys():
                task['prompt'] = task['description']
            elif 'text' in task.keys():
                task['prompt'] = task['text']
            elif 'input' in task.keys():
                task['prompt'] = task['input']
            else:
                raise NotImplementedError

        args.cur_id = idx
        total_runs = 0
        intent = task['prompt']
        print(intent)
        before_func=''
        method_name = args.dataset + '_' +  str(idx)
        before_func,code_in_prompt = prompt_split_humaneval(intent,method_name)

        print('-'*100)

        generate_ids = []

        # feature_node 就是这一轮要改变的feature
        if args.run_certain_level!='' and feature_node.level != args.run_certain_level:
            # feature4nextstep.append(feature_node)
            continue
        feature_str = generate_feature_str(pruned_features)
        if args.run_generate==1:
            levels = [feature_node.level for feature_node in pruned_features]
            keys = [feature_node.key for feature_node in pruned_features]
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
                

        res = False
        if args.run_evaluate==1:
            print('evaluating ...')
            method_name = args.dataset + '_' +  str(idx)
            new_method_name = method_name
            res = evaluate(task,new_method_name)
            print(res)

    passAt10s=[]
    if bool(args.run_evaluate):
        print('evluating...')
        output_path = args.output_path + '{}_prune_{}.jsonl'.format(args.dataset,args.prune_feature_num)
        with open(output_path, 'w+') as f:
            for idx,task in enumerate(loaded_dataset):
                if args.eval_start_index!=-1 and idx<args.eval_start_index:
                    continue
                codes = []
                plans=[]

                method_name = args.dataset + '_' +  str(idx)
                new_method_name = method_name
                plan_file_path = '/home/zlyuaj/Causal/MetaGPT/{}/{}'.format(args.workspace,new_method_name)
                # print(plan_file_path)
                if not os.path.exists(plan_file_path):
                    continue

                
                method_name = args.dataset + '_' +  str(idx)
                for cnt in range(args.num_generate):
                    
                    new_method_name = method_name
                    plan_file_path = '/home/zlyuaj/Causal/MetaGPT/{}/{}'.format(args.workspace,new_method_name)
                    if not os.path.exists(plan_file_path):
                        print(plan_file_path)
                        print('no plan_file_path')
                        continue
                    RA,RP,IA = extract_plan_from_repo(plan_file_path)
                    code_file_path = '/home/zlyuaj/Causal/MetaGPT/{}/{}/{}'.format(args.workspace,new_method_name,new_method_name)
                    # print(code_file_path)
                    if not os.path.exists(code_file_path):
                        print('no code_file_path')
                        continue
                    code = extract_code_from_repo(code_file_path)
                    # print(code)
                    # print(RA,RP,IA)
                    if not code or (not RA and not RP and not IA):
                        continue    
                    plan = format_plan(RA,RP,IA)
                    
                    code = 'from typing import List\n'+code
                    import re

                    def remove_input_content(code):
                        # 使用正则表达式替换 input() 中的内容
                        code = re.sub(r'input\([^)]*\)', 'input()', code)
                        return code
                    code = remove_input_content(code)
                    # if 'codecontest' in args.dataset and 'if __name__ == "__main__":' not in code:
                    #     code += '\nmain()'
                    # print('-'*100)
                    # print(code)
                    # print('-'*100)
                    # print(plan)
                    codes.append(code)
                    # plans.append(plan)
                    
                # print(RequirementAnalysis,RequirementPool,ImplementationApproach)
                # delete_repo(file_path)
                # code, session_history=future.result()
                # print('#'*30)
                # print(code)
                run_eval=True
                if not codes:
                    print(f'no answer for question {idx}')
                    codes = ['']*10
                    plans=['']*10
                    run_eval = False
                else:
                    while len(codes)<args.num_generate:
                        ran = random.randint(0,len(codes)-1)
                        codes.append(codes[ran])
                        plans.append(plans[ran])
                
                task['completions']=codes
                task['plans']=plans
                if run_eval:
                    print('evaluating ...')
                    if 'human' in args.dataset:
                        score, passes,passAt1, passAt10= evaluate_one(task,args.num_generate)
                    elif 'mbpp' in args.dataset:
                        score, passes,passAt1, passAt10= evaluate_one_MBPP(task,args.num_generate)
                    elif 'codecontest' in args.dataset:
                        score, passes,passAt1, passAt10= evaluate_one_codecontest(task,args.num_generate)
                    elif 'CoderEval' in args.dataset:
                        passAt10,passes= evaluate_one_CoderEval(task,args.num_generate)
                else:
                    score, passes,passAt1, passAt10 = False,0,False,False
                # mutated_seed.score=score
                with open(plan_file_path + '/eval_result.txt','w+') as eval_f:
                    eval_f.write(f'{passes}\n')
                
                print(passAt10)
                task['pass'] = passAt10
                task['pass_num'] = passes
                # task['round']=idx
                task2save = {}
                for key in task:
                    if key!='test' and key!='completions' and key!='plans' and key!='test_list':
                        task2save[key]=task[key]
                # entry_point = find_method_name(code)

                passAt10s.append(passAt10)
                f.write(json.dumps(task) + '\n')
                f.flush()
                if idx%10==0:
                    print('current round: '+str(idx))
                    print('current pass@10: '+ str(passAt10s.count(True)/len(passAt10s)))
    print('final pass@10: '+ str(passAt10s.count(True)/len(passAt10s))) 
    print('total pass number: '+ str(passAt10s.count(True)))   
                
            
       
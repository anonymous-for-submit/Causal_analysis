
# -*- coding: utf-8 -*-
import sys
sys.path.append('/home/zlyuaj/Causal/MetaGPT')
sys.path.append('/home/zlyuaj/Causal/MetaGPT/metagpt')
import asyncio
from pathlib import Path

import agentops
import typer

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

# import torch
# import torch.nn.functional as F
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from datasets import load_dataset, load_from_disk
from collections import defaultdict
from evaluate_result import evaluate_one,evaluate_one_codecontest,evaluate_one_MBPP,evaluate_one_CoderEval
from concurrent.futures import as_completed, ProcessPoolExecutor
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
parser.add_argument('--recover', type=int, default=0)
parser.add_argument('--begin_idx', type=int, default=0)
parser.add_argument('--end_idx', type=int, default=1000)
parser.add_argument('--do_intervent', type=int, default=0)
parser.add_argument('--levels', type=str, default='')


parser.add_argument('--add_monitor', type=int, default=0)
parser.add_argument('--repair_plan', type=int, default=0)
parser.add_argument('--repair_code', type=int, default=0)
parser.add_argument('--run_multi_gen', type=int, default=0)
parser.add_argument('--repair_prompt_num', type=int, default=0)

args = parser.parse_args()
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

# def extract_code_from_repo(file_path,prompt,method_name,code_in_prompt):
#     files=os.listdir(file_path)
#     num_py_files = len(files)
#     # print(files)
#     file_name = files[0]
#     if 'main' in file_name and num_py_files>1:
#         file_name = files[1]
#     # print(file_name)
#     sourse=''
#     code=''
#     with open(file_path+ '/'+file_name,'r') as f:
#         sourse = f.read()
#         # print(sourse)
#         # code =''
#         code  = extract_code_from_sourse(sourse,code_in_prompt)
#     return code,sourse
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

def check_all_generated(loaded_dataset):
    for idx,task in enumerate(loaded_dataset):
        method_name = args.dataset + '_' +  str(idx)
        new_method_name = method_name+"_0"
        code_file_path = '/home/zlyuaj/Causal/MetaGPT/{}/{}/{}/main.py'.format(args.workspace,new_method_name,new_method_name)
        if not os.path.exists(code_file_path):
            return False
    return True


if __name__ == '__main__':
    coding_prompt=''

    initial_output_path=args.output_path
    # print(os.path.exists(initial_output_path))
    if not os.path.exists(initial_output_path):
        os.mkdir(initial_output_path)
    # print(os.path.exists(initial_output_path))
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
    loaded_dataset=[]
    with open(INPUTPATH, 'r') as f:
        # 导入输出
        loaded_dataset = [json.loads(line) for line in f]

    # for data in loaded_dataset[:10]:
    #     print('-'*100)
    #     print([data['prompt']])
    

    # loaded_dataset = loaded_dataset[:5]

    print(len(loaded_dataset))
    passAt10s=[]

    initial_seed = loaded_dataset
    initial_seed_num=len(loaded_dataset)


    # text, code, task_id, test_list, entry_point

    fail_list = []
    
    max_try = 5

    if bool(args.run_generate):
        if check_all_generated(loaded_dataset):
            print('all code already generated, skip generating')
        else:
            while not check_all_generated(loaded_dataset) and max_try>0:
                max_try-=1
                for idx,task in enumerate(loaded_dataset):
                    if args.begin_idx>0 and idx<args.begin_idx:
                        continue
                    if args.end_idx<=1000 and idx>args.end_idx:
                        break
                    # if idx>0:
                    #     break
                    print('-'*10+'executing task: {}'.format(idx)+'-'*10)


                    if 'prompt' not in task.keys():
                        if 'description' in task.keys():
                            task['prompt'] = task['description']
                        elif 'text' in task.keys():
                            task['prompt'] = task['text']
                        elif 'input' in task.keys():
                            task['prompt'] = task['input']
                        else:
                            raise NotImplementedError
                    intent = task['prompt']
                    before_func=''
                    method_name = args.dataset + '_' +  str(idx) 
                    before_func,code_in_prompt = prompt_split_humaneval(intent,method_name)

                    
                    new_method_name = method_name+"_0"
                    code_file_path = '/home/zlyuaj/Causal/MetaGPT/{}/{}/{}/main.py'.format(args.workspace,new_method_name,new_method_name)
                    if os.path.exists(code_file_path):
                        print('already exists, skip generating')
                        continue



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
                                            new_method_name = method_name+"_"+str(cnt)
                                            # repo=startup(idea=coding_prompt+intent,project_name=method_name)
                                            # file_path = '/data/zlyuaj/muti-agent/MetaGPT/workspace/'+ new_method_name+'/'+new_method_name
                                            # code = extract_code_from_repo(file_path,intent)
                                            # file_path = '/data/zlyuaj/muti-agent/MetaGPT/workspace/'+ new_method_name
                                            # delete_repo(file_path)
                                            # future= executor.submit(session.run_session,need_second_round,finally_pass)
                                            # print(f'idea: {coding_prompt+intent}')
                                            # print('in generating repo...')
                                            idea = coding_prompt+task['prompt']
                                            # print(f'idea: {idea}')
                                            args_dict = vars(args)
                                            future= executor.submit(startup,idea=idea,project_name=new_method_name,args=args_dict)
                                        
                                            futures.append(future)
                                    results=[]
                                    for cnt, future in enumerate(as_completed(futures)):
                                        # print(future.result())
                                        results.append(future.result())
                                        new_method_name = method_name+"_"+str(cnt)
                                        file_path = '/home/zlyuaj/Causal/MetaGPT/{}/{}/{}'.format(args.workspace,new_method_name,new_method_name)
                                        if not os.path.exists(file_path):
                                            regenerate.append(cnt)

                                    return regenerate
                            else:
                                for cnt in generate_ids:
                                    new_method_name = method_name+"_"+str(cnt)
                                    args_dict = vars(args)
                                    startup(idea=coding_prompt+task['prompt'],project_name=new_method_name,args=args_dict)
                                    
                                    # session_historys.append(session_history)
                        except Exception as e:
                            # raise NotImplementedError
                            print(e)
                            return generate_ids

                    generate_ids=[i for i in range(args.num_generate)]
                    generate(method_name,generate_ids,task)


    for idx,task in enumerate(loaded_dataset):
        method_name = args.dataset + '_' +  str(idx)
        new_method_name = method_name+"_0"
        code_file_path = '/home/zlyuaj/Causal/MetaGPT/{}/{}/{}/main.py'.format(args.workspace,new_method_name,new_method_name)
        if not os.path.exists(code_file_path):
            print('no code file for question ', idx)
            raise FileNotFoundError(f'No code file found for question {idx}')


    fail_list = []
    if bool(args.run_evaluate):
        print('evluating...')
        output_path = args.output_path + '{}.jsonl'.format(args.dataset)
        with open(output_path, 'w+') as f:
            for idx,task in enumerate(loaded_dataset):
                if args.eval_start_index!=-1 and idx<args.eval_start_index:
                    continue
                codes = []
                plans=[]

                method_name = args.dataset + '_' +  str(idx)
                new_method_name = method_name+"_0"
                plan_file_path = '/home/zlyuaj/Causal/MetaGPT/{}/{}'.format(args.workspace,new_method_name)
                # print(plan_file_path)
                if not os.path.exists(plan_file_path):
                    continue

                
                method_name = args.dataset + '_' +  str(idx)
                for cnt in range(args.num_generate):
                    
                    new_method_name = method_name+"_"+str(cnt)
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

                
                print(passAt10)
                task['pass'] = passAt10
                task['pass_num'] = passes
                # task['round']=idx
                with open(plan_file_path + '/eval_result.txt','w+') as eval_f:
                    eval_f.write(f'{passes}\n')
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
    print('-'*100)
    if passAt10s:
        print(passAt10s)
        fail_list = [i for i in range(len(passAt10s)) if not passAt10s[i]]
        print('fail list:')
        print(fail_list)
        print('final_result: '+str(passAt10s.count(True)/len(passAt10s)))

    else:
        print('no evaluation result')
           




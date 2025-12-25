import json
import os
from typing import List, Dict, Tuple, Any
import random
import openai
from openai import OpenAI
from openai import AzureOpenAI

client_4o_mini = OpenAI(
        api_key = "sk-FRQxdGxCMDSPoogN0SgdGGm4IEfv3uMjUFTtgepRNC7bnxO8",
        base_url = "https://api.chatanywhere.tech/v1"
        )

client_ds_qwen= OpenAI(api_key= 'EMPTY', base_url= 'http://127.0.0.1:8000/v1/')


def call_openai(prompt,model):
    # model='gpt-4o-mini-ca'
    if  '4o' in  model:
        model = 'gpt-4o-mini-ca'
    if 'deepseek' in  model:
        model = 'deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct'
    if 'qwen' in model:
        model = 'qwen/Qwen2.5-14B-Instruct'
    message = [
            {"role": "user", "content": prompt}
        ]
    if '4o' in model:
        client = client_4o_mini
    else:
        client=client_ds_qwen
    
    response = client.chat.completions.create(
    model=model,
    messages=message,
    )

    return [choice.message.content for choice in response.choices][0]

class MASOutputModifier:
    """用于修改Multi-Agent System中间输出的工具类"""
    
    def __init__(self):
        self.modification_strategies = {
            'prd': self._get_prd_modifications,
            'design': self._get_design_modifications,
            'task': self._get_task_modifications
        }
    
    def get_data_from_jsonl(self, path: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """从JSONL文件中读取数据"""
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
            system_design = data.get('system_design', {})
            task = data.get('task', {})

            prd_list.append(prd)
            system_design_list.append(system_design)
            task_list.append(task)

        return prd_list, system_design_list, task_list
    
    def _get_prd_modifications(self) -> List[str]:
        """获取PRD可能的修改策略"""
        return [
            "Change 'Language' to a different language (e.g., 'zh_cn'), keep other field of output still in English",
            "Change 'Programming Language' to another language (e.g., 'Java'), keep the other field of code still in python",
            "Modify 'Original Requirements' to introduce logic changes or different constraints",
            "Alter 'Product Goals' to shift focus or priorities",
            "Change 'User Stories' to reflect different user personas or use cases",
            "Modify competitive analysis to include different competitors or change positioning",
            "Adjust 'Requirement Pool' priorities or add conflicting requirements",
            "Introduce ambiguity in 'Anything UNCLEAR' section"
        ]
    
    def _get_design_modifications(self) -> List[str]:
        """获取System Design可能的修改策略"""
        return [
            "Change the implementation approach to use different algorithms or data structures",
            "Modify file structure or add/remove files",
            "Alter data structures and interfaces (e.g., change function signatures, add parameters)",
            "Change the program call flow or add new interaction patterns",
            "Introduce performance constraints or optimization requirements",
            "Add or remove dependencies and third-party libraries",
            "Modify design patterns or architectural decisions"
        ]
    
    def _get_task_modifications(self) -> List[str]:
        """获取Task可能的修改策略"""
        return [
            "Change required packages to different versions or alternatives",
            "Modify required third-party packages",
            "Alter the logic analysis to introduce different implementation steps",
            "Change file name in file list or reorganize code structure",
            "Modify API specifications",
            "Introduce new shared knowledge or constraints",
            "Add complexity or simplify implementation details"
        ]
    
    def parse_llm_output_to_dict(self, llm_output: str, original_dict: Dict) -> Dict:
        """
        将LLM输出的字符串转换为字典，并验证结构与原始字典一致
        
        Args:
            llm_output: LLM返回的字符串（可能包含转义字符、markdown等）
            original_dict: 原始字典，用于验证结构
            
        Returns:
            解析后的字典
            
        Raises:
            ValueError: 如果解析失败或结构不匹配
        """
        try:
            # 步骤1: 清理输出 - 移除可能的markdown代码块标记
            cleaned_output = llm_output.strip()
            
            # 移除markdown代码块
            if cleaned_output.startswith('```'):
                lines = cleaned_output.split('\n')
                # 移除第一行（```json 或 ```）和最后一行（```）
                if lines[0].startswith('```'):
                    lines = lines[1:]
                if lines and lines[-1].strip() == '```':
                    lines = lines[:-1]
                cleaned_output = '\n'.join(lines)
            
            # 步骤2: 尝试直接解析JSON
            try:
                parsed_dict = json.loads(cleaned_output)
            except json.JSONDecodeError:
                # 如果直接解析失败，尝试处理转义字符
                # 有时LLM会输出类似 {\"key\": \"value\"} 的格式
                try:
                    # 先尝试eval（谨慎使用，但在受控环境下可以处理一些边缘情况）
                    import ast
                    parsed_dict = ast.literal_eval(cleaned_output)
                except:
                    # 最后尝试：手动处理转义字符
                    cleaned_output = cleaned_output.replace('\\n', '\n').replace('\\"', '"')
                    parsed_dict = json.loads(cleaned_output)
            
            # 步骤3: 验证结构
            validation_result = self._validate_dict_structure(parsed_dict, original_dict)
            
            if not validation_result['valid']:
                error_msg = f"Structure validation failed:\n"
                if validation_result['missing_keys']:
                    error_msg += f"Missing keys: {validation_result['missing_keys']}\n"
                if validation_result['extra_keys']:
                    error_msg += f"Extra keys: {validation_result['extra_keys']}\n"
                if validation_result['type_mismatches']:
                    error_msg += f"Type mismatches: {validation_result['type_mismatches']}\n"
                
                print(f"WARNING: {error_msg}")
                
                # 尝试修复：如果缺少key，从原始字典补充
                for key in validation_result['missing_keys']:
                    parsed_dict[key] = original_dict[key]
                    print(f"Added missing key '{key}' from original dict")
                
                # 移除多余的key
                for key in validation_result['extra_keys']:
                    del parsed_dict[key]
                    print(f"Removed extra key '{key}'")
            
            return parsed_dict
            
        except Exception as e:
            print(f"Error parsing LLM output: {e}")
            print(f"LLM output (first 500 chars): {llm_output[:500]}")
            raise ValueError(f"Failed to parse LLM output: {e}")
    
    def _validate_dict_structure(self, parsed_dict: Dict, original_dict: Dict) -> Dict[str, Any]:
        """
        验证两个字典的结构是否一致
        
        Args:
            parsed_dict: 解析后的字典
            original_dict: 原始字典
            
        Returns:
            验证结果字典，包含：
            - valid: bool, 是否完全匹配
            - missing_keys: List[str], 缺失的key
            - extra_keys: List[str], 多余的key
            - type_mismatches: List[Dict], 类型不匹配的key及其类型
        """
        missing_keys = []
        extra_keys = []
        type_mismatches = []
        
        original_keys = set(original_dict.keys())
        parsed_keys = set(parsed_dict.keys())
        
        # 检查缺失的key
        missing_keys = list(original_keys - parsed_keys)
        
        # 检查多余的key
        extra_keys = list(parsed_keys - original_keys)
        
        # 检查类型匹配
        for key in original_keys.intersection(parsed_keys):
            original_type = type(original_dict[key])
            parsed_type = type(parsed_dict[key])
            
            # 对于列表，进一步检查元素类型
            if original_type == list and parsed_type == list:
                if original_dict[key] and parsed_dict[key]:
                    original_elem_type = type(original_dict[key][0]) if original_dict[key] else None
                    parsed_elem_type = type(parsed_dict[key][0]) if parsed_dict[key] else None
                    
                    if original_elem_type != parsed_elem_type and original_elem_type is not None:
                        type_mismatches.append({
                            'key': key,
                            'expected_type': f'List[{original_elem_type.__name__}]',
                            'actual_type': f'List[{parsed_elem_type.__name__}]'
                        })
            elif original_type != parsed_type:
                type_mismatches.append({
                    'key': key,
                    'expected_type': original_type.__name__,
                    'actual_type': parsed_type.__name__
                })
        
        valid = (len(missing_keys) == 0 and 
                len(extra_keys) == 0 and 
                len(type_mismatches) == 0)
        
        return {
            'valid': valid,
            'missing_keys': missing_keys,
            'extra_keys': extra_keys,
            'type_mismatches': type_mismatches
        }
    
    def generate_modification_prompt(self, output_type: str, requirement: str, 
                                     internal_output: Dict, temperature: str = "moderate") -> str:
        """生成修改prompt"""
        
        if output_type not in self.modification_strategies:
            raise ValueError(f"Invalid output_type: {output_type}")
        
        modification_examples = self.modification_strategies[output_type]()
        
        temperature_instructions = {
            'mild': "Make subtle, realistic modifications that simulate minor misunderstandings or interpretation differences.",
            'moderate': "Make noticeable modifications that simulate common mistakes or alternative interpretations.",
            'aggressive': "Make significant modifications that simulate major misunderstandings or completely different interpretations."
        }
        
        prompt = f"""You are an expert in simulating realistic errors in multi-agent code generation systems.

**Task**: Modify the {output_type.upper()} internal output to simulate how the system might process a SIMILAR but DIFFERENT requirement, introducing realistic errors that could occur during multi-agent collaboration.

**Original Requirement**:
```
{requirement}
```

**Current {output_type.upper()} Output**:
```json
{json.dumps(internal_output, indent=2, ensure_ascii=False)}
```

**Modification Guidelines**:
{temperature_instructions.get(temperature, temperature_instructions['moderate'])}

**Modification Strategies for {output_type.upper()}**:
{chr(10).join(f"{i+1}. {strategy}" for i, strategy in enumerate(modification_examples))}

**Critical Requirements**:
1. **Modify EVERY field** in the output - no field should remain exactly the same
2. Ensure modifications are **internally consistent** with each other
3. Create realistic errors that an AI agent might make (e.g., misunderstanding requirements, wrong assumptions)
4. Maintain valid JSON structure and data types
5. The modified output should still be plausible but subtly incorrect
6. **IMPORTANT**: Keep ALL the same keys as the original output - do not add or remove any keys
7. **IMPORTANT**: Keep the same data types for all values (string stays string, list stays list, etc.)

**Output Format**:
Return ONLY the modified JSON object without any explanation, markdown formatting, or additional text.
"""
        
        return prompt
    
    def modify_internal_output(self, output_type: str, internal_output: Dict, 
                               requirement: str, model, 
                               fallback_pool: List[Dict] = None,
                               temperature: str = "moderate",
                               max_retries: int = 3) -> Dict:
        """
        修改内部输出，最多尝试3次LLM生成，失败后从fallback_pool随机选择
        
        Args:
            output_type: 'prd', 'design', 或 'task'
            internal_output: 要修改的内部输出
            requirement: 原始需求
            model: 使用的模型
            fallback_pool: 备选输出池，用于LLM失败后随机选择
            temperature: 修改强度
            max_retries: 最大重试次数
        """
        
        # 尝试使用LLM生成
        for attempt in range(max_retries):
            try:
                print(f"\n{'='*80}")
                print(f"Attempt {attempt + 1}/{max_retries} for {output_type.upper()} modification")
                print(f"{'='*80}")
                
                prompt = self.generate_modification_prompt(
                    output_type, requirement, internal_output, temperature
                )
                
                if attempt == 0:  # 只在第一次打印完整prompt
                    print(f"Generated Prompt:")
                    print(prompt)
                    print(f"{'='*80}\n")
                
                # 调用LLM API获取修改后的输出
                llm_raw_output = call_openai(prompt=prompt, model=model)

                print(f"Raw LLM Output (attempt {attempt + 1}):")
                print(llm_raw_output[:500] + "..." if len(llm_raw_output) > 500 else llm_raw_output)
                print(f"{'='*80}\n")
                
                # 将LLM输出转换为字典并验证结构
                modified_output = self.parse_llm_output_to_dict(llm_raw_output, internal_output)
                print(f"✓ Successfully parsed and validated output for {output_type} on attempt {attempt + 1}")
                return modified_output
                
            except Exception as e:
                print(f"✗ Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying...")
                continue
        
        # 所有LLM尝试都失败，使用fallback_pool
        print(f"\n{'!'*80}")
        print(f"All {max_retries} LLM attempts failed for {output_type}")
        print(f"Falling back to random selection from fallback pool")
        print(f"{'!'*80}\n")
        
        if fallback_pool and len(fallback_pool) > 0:
            # 从fallback_pool中随机选择一个不同的输出
            available_options = [item for item in fallback_pool if item != internal_output]
            if not available_options:
                available_options = fallback_pool
            
            selected_output = random.choice(available_options)
            print(f"✓ Selected random output from fallback pool (pool size: {len(fallback_pool)})")
            return selected_output
        else:
            print(f"✗ WARNING: No fallback pool provided, returning original output")
            return internal_output
    
    def _rule_based_modification(self, output_type: str, internal_output: Dict) -> Dict:
        """
        基于规则的修改（已弃用，不再使用）
        保留此函数仅为兼容性，实际不会被调用
        """
        raise NotImplementedError("Rule-based modification is deprecated and should not be used")
    
    def batch_modify_outputs(self, datasets: List[str], models: List[str], 
                            base_path: str, output_path: str, 
                            modification_type: str = 'all',
                            temperature: str = 'moderate',
                            max_retries: int = 3):
        """
        批量修改输出
        
        Args:
            datasets: 数据集列表
            models: 模型列表
            base_path: 基础路径模板
            output_path: 输出路径模板
            modification_type: 'prd', 'design', 'task', 或 'all'
            temperature: 修改强度
            max_retries: LLM最大重试次数
        """
        
        for model in models:
            for dataset in datasets:
                try:
                    original_result_path = base_path.format(model=model, dataset=dataset)
                    prd_list, design_list, task_list = self.get_data_from_jsonl(original_result_path)
                    
                    print(f"\n{'#'*80}")
                    print(f"Processing {model} - {dataset}...")
                    print(f"Found {len(prd_list)} samples")
                    print(f"{'#'*80}\n")
                    
                    modified_data = []
                    output_file = output_path.format(model=model, dataset=dataset)
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)
                    
                    with open(output_file + '.jsonl', 'w', encoding='utf-8') as f:
                        for idx, (prd, design, task) in enumerate(zip(prd_list, design_list, task_list)):
                            print(f"\n{'='*80}")
                            print(f"Processing sample {idx + 1}/{len(prd_list)}")
                            print(f"{'='*80}")
                            
                            sample = {
                                'file_name': f'sample_{idx}',
                                'original_prd': prd,
                                'original_design': design,
                                'original_task': task
                            }
                            
                            requirement = prd.get('Original Requirements', '')
                            
                            # 为每种类型准备fallback pool（排除当前样本）
                            if modification_type in ['prd', 'all']:
                                prd_fallback_pool = [p for i, p in enumerate(prd_list) if i != idx]
                                sample['prd'] = self.modify_internal_output(
                                    'prd', prd, requirement, model, 
                                    fallback_pool=prd_fallback_pool,
                                    temperature=temperature,
                                    max_retries=max_retries
                                )
                            
                            if modification_type in ['design', 'all']:
                                design_fallback_pool = [d for i, d in enumerate(design_list) if i != idx]
                                sample['design'] = self.modify_internal_output(
                                    'design', design, requirement, model,
                                    fallback_pool=design_fallback_pool,
                                    temperature=temperature,
                                    max_retries=max_retries
                                )
                            
                            if modification_type in ['task', 'all']:
                                task_fallback_pool = [t for i, t in enumerate(task_list) if i != idx]
                                sample['task'] = self.modify_internal_output(
                                    'task', task, requirement, model,
                                    fallback_pool=task_fallback_pool,
                                    temperature=temperature,
                                    max_retries=max_retries
                                )
                            
                            modified_data.append(sample)
                            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                            f.flush()
                            
                            print(f"✓ Sample {idx + 1} completed and saved")
                    
                    print(f"\n{'#'*80}")
                    print(f"✓ All samples processed and saved to {output_file}.jsonl")
                    print(f"{'#'*80}\n")
                    
                except FileNotFoundError as e:
                    print(f"✗ Error processing {model} - {dataset}: {e}")
                except Exception as e:
                    print(f"✗ Unexpected error: {e}")
                    import traceback
                    traceback.print_exc()


# 使用示例
if __name__ == "__main__":
    modifier = MASOutputModifier()
    
    # 批量处理
    datasets = ["humaneval", "mbpp", "codecontest", "CoderEval"]
    models = ["gpt-4o-mini-ca", "deepseek-coder", "qwen"]
    
    # 测试用较小的数据集
    # datasets = ["humaneval"]
    models = ["qwen"]
    datasets = ["humaneval"]
    
    modifier.batch_modify_outputs(
        datasets=datasets,
        models=models,
        base_path='/home/zlyuaj/Causal/MetaGPT/workspace_{model}_{dataset}_original_result',
        output_path='/home/zlyuaj/Causal/MetaGPT/workspace_{model}_{dataset}_modified_result',
        modification_type='all',
        temperature='moderate',
        max_retries=3  # 最多尝试3次LLM生成
    )
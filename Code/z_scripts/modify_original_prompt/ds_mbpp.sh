#!/bin/bash

# 配置参数
model=deepseek-coder
name_tag=deepseek
dataset=mbpp
num_generate=1
begin_idx=0
end_idx=427



# OpenAI API配置（用于调用gpt-4o-mini改写prompt）
export OPENAI_API_KEY="your-api-key-here"
# 如果使用自定义endpoint，取消下面这行注释
# export OPENAI_BASE_URL="https://your-custom-endpoint.com/v1"

# 运行修改脚本
python /home/zlyuaj/Causal/MetaGPT/metagpt/software_causal_basedataset_generate_modified_question.py \
    --input_path /home/zlyuaj/Causal/MetaGPT/data/mbpp_sanitized_ET.jsonl \
    --dataset ${dataset} \
    --model ${model} \
    --output_path /home/zlyuaj/Causal/MetaGPT/output/${dataset}/ \
    --output_file_name ${dataset}_${model}_original_dataset_modified_requirements \
    --workspace workspace_${model}_${dataset}_modified_orignal \
    --num_generate ${num_generate} \
    --parallel 0 \
    --begin_idx ${begin_idx} \
    --end_idx ${end_idx} \
    | tee output_${model}_${dataset}_modified_requirements_idx_${begin_idx}_${end_idx}.txt
model=qwen
name_tag=qwen
dataset=mbpp
num_generate=1
run_generate=1
run_evaluate=1
begin_idx=0
end_idx=427
repair_feature_num=5
parallel=1

# CoderEval数据路径（如果使用CoderEval数据集）
codereval_data_path=/home/zlyuaj/Causal/MetaGPT/data/CoderEval4Python.json

python /home/zlyuaj/Causal/MetaGPT/metagpt/software_causal_basedataset_repair.py  \
    --model ${model}\
    --output_path /home/zlyuaj/Causal/MetaGPT/output/repair/ \
    --input_path /home/zlyuaj/Causal/MetaGPT/data/mbpp_sanitized_ET.jsonl \
    --original_result_path /home/zlyuaj/Causal/MetaGPT/output/mbpp/results-mbpp_${model}_original_dataset/mbpp.jsonl \
    --original_result_workspace /home/zlyuaj/Causal/MetaGPT/workspace_${model}_mbpp_original_result\
    --dataset ${dataset}\
    --output_file_name ${dataset}_${model}_repair_top${repair_feature_num} \
    --workspace workspace_${model}_${dataset}_repair_top${repair_feature_num}\
    --num_generate ${num_generate}\
    --run_generate ${run_generate}\
    --run_evaluate ${run_evaluate}\
    --begin_idx ${begin_idx}\
    --end_idx ${end_idx}\
    --repair_feature_num ${repair_feature_num}\
    --codereval_data_path ${codereval_data_path}\
    --parallel ${parallel}\
    | tee output_${model}_${dataset}_repair_top${repair_feature_num}_idx_${begin_idx}_${end_idx}.txt
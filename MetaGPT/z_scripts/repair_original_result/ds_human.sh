model=deepseek-coder
name_tag=deepseek-coder
dataset=humaneval
num_generate=1
run_generate=1
run_evaluate=1
begin_idx=0
end_idx=164
repair_feature_num=5
parallel=1

# CoderEval数据路径（如果使用CoderEval数据集）
codereval_data_path=/home/zlyuaj/Causal/MetaGPT/data/CoderEval4Python.json

python /home/zlyuaj/Causal/MetaGPT/metagpt/software_causal_basedataset_repair.py  \
    --model ${model}\
    --input_path /home/zlyuaj/Causal/MetaGPT/data/HumanEval_test_case_ET.jsonl \
    --output_path /home/zlyuaj/Causal/MetaGPT/output/repair/ \
    --dataset ${dataset}\
    --output_file_name ${dataset}_${model}_repair_top${repair_feature_num} \
    --workspace workspace_${model}_${dataset}_repair_top${repair_feature_num}\
    --num_generate ${num_generate}\
    --run_generate ${run_generate}\
    --run_evaluate ${run_evaluate}\
    --begin_idx ${begin_idx}\
    --end_idx ${end_idx}\
    --repair_feature_num ${repair_feature_num}\
    --original_result_path /home/zlyuaj/Causal/MetaGPT/output/humaneval/results-humaneval_${model}_original_dataset/humaneval.jsonl \
    --original_result_workspace /home/zlyuaj/Causal/MetaGPT/workspace_${model}_humaneval_original_result\
    --codereval_data_path ${codereval_data_path}\
    --parallel ${parallel}\
    | tee output_${model}_${dataset}_repair_top${repair_feature_num}_idx_${begin_idx}_${end_idx}.txt

model=deepseek-coder
name_tag=deeepseek
dataset=humaneval
num_generate=1
run_generate=1
run_evaluate=1
begin_idx=0
end_idx=170
run_certain_level=''
prune_feature_num=20
python /home/zlyuaj/Causal/MetaGPT/metagpt/software_causal_basedataset_prune.py  \
    --model ${model}\
    --input_path /home/zlyuaj/Causal/MetaGPT/data/HumanEval_test_case_ET.jsonl \
    --output_path /home/zlyuaj/Causal/MetaGPT/output/prune_original/ \
    --dataset ${dataset}\
    --output_file_name ${dataset}_${model}_original_dataset_prune_${prune_feature_num} \
    --workspace workspace_${model}_${dataset}_original_result_prune_${prune_feature_num}\
    --num_generate ${num_generate}\
    --run_generate ${run_generate}\
    --run_evaluate ${run_evaluate}\
    --begin_idx ${begin_idx}\
    --end_idx  ${end_idx}\
    --do_intervent 1\
    --empty_intervention 1\
    --prune_feature_num ${prune_feature_num}\
    --parallel 0\
    | tee output_${model}_${dataset}_basedataset_prune_${prune_feature_num}_idx_${begin_idx}_${end_idx}.txt  
    

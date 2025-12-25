
model=deepseek-coder
name_tag=deepseek
dataset=CoderEval
num_generate=1
run_generate=1
run_evaluate=1
begin_idx=43
end_idx=100
run_certain_level=''
python /home/zlyuaj/Causal/MetaGPT/metagpt/software_causal.py  \
    --model ${model}\
    --input_path /home/zlyuaj/Causal/MetaGPT/data/CoderEval.jsonl \
    --output_path /home/zlyuaj/Causal/MetaGPT/output/${dataset}/ \
    --original_result_path /home/zlyuaj/Causal/MetaGPT/output/CoderEval/results-CoderEval_gpt-4o-mini-ca_original_dataset/CoderEval.jsonl \
    --original_result_workspace /home/zlyuaj/Causal/MetaGPT/workspace_gpt-4o-mini-ca_CoderEval_original_result \
    --dataset ${dataset}\
    --output_file_name ${dataset}_${model}_intervent_${begin_idx}_${end_idx} \
    --workspace workspace_${model}_${dataset}_len1 \
    --num_generate ${num_generate}\
    --run_generate ${run_generate}\
    --run_evaluate ${run_evaluate}\
    --generate_len1 1\
    --generate_len2 1\
    --begin_idx ${begin_idx}\
    --end_idx  ${end_idx}\
    --parallel 0\
    --early_stop_threshold 10\
    --max_len_cause 5\
    --do_intervent 1\
    --sim_threshold 0.5 \
    | tee output_${model}_${dataset}_idx_${begin_idx}_${end_idx}.txt  
    

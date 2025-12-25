
model=gpt-4o-mini-ca
name_tag=random_intervene_length_earlystop
dataset=humaneval
num_generate=1
run_generate=1
run_evaluate=1
begin_idx=101
end_idx=165
run_certain_level=''
python /home/zlyuaj/Causal/MetaGPT/metagpt/software_causal_random_length_earlystop.py  \
    --model ${model}\
    --input_path /home/zlyuaj/Causal/MetaGPT/data/HumanEval_test_case_ET.jsonl \
    --output_path /home/zlyuaj/Causal/MetaGPT/output/Abalation/ \
    --original_result_path /home/zlyuaj/Causal/MetaGPT/output/humaneval/results-humaneval_gpt-4o-mini-ca_original_dataset/humaneval.jsonl \
    --original_result_workspace /home/zlyuaj/Causal/MetaGPT/workspace_gpt-4o-mini-ca_humaneval_4o-mini_original_result\
    --dataset ${dataset}\
    --output_file_name ${dataset}_${model}_${name_tag}_${begin_idx}_${end_idx} \
    --workspace workspace_${model}_${dataset}_${name_tag}_len2 \
    --num_generate ${num_generate}\
    --run_generate ${run_generate}\
    --run_evaluate ${run_evaluate}\
    --generate_len1 1\
    --generate_len2 1\
    --begin_idx ${begin_idx}\
    --end_idx  ${end_idx}\
    --parallel 0\
    --early_stop 1\
    --early_stop_threshold 10\
    --max_len_cause 5\
    --do_intervent 1\
    --sim_threshold 0.5 \
    | tee output_${model}_${dataset}_${name_tag}_idx_${begin_idx}_${end_idx}.txt  
    

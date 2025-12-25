
model=gpt-4o-mini-ca
name_tag=4o-mini
dataset=humaneval
num_generate=1
run_generate=1
run_evaluate=1
run_certain_level=''
python /home/zlyuaj/Causal/MetaGPT/metagpt/software_causal.py  \
    --model ${model}\
    --input_path /home/zlyuaj/Causal/MetaGPT/data/HumanEval_test_case_ET.jsonl \
    --output_path /home/zlyuaj/Causal/MetaGPT/output/${dataset}/ \
    --original_result_path /home/zlyuaj/Causal/MetaGPT/output/humaneval/results-humaneval_gpt-4o-mini-ca_original_dataset/humaneval.jsonl \
    --original_result_workspace /home/zlyuaj/Causal/MetaGPT/workspace_gpt-4o-mini-ca_humaneval_4o-mini_original_result\
    --dataset ${dataset}\
    --output_file_name ${dataset}_${model}_intervent_test \
    --workspace workspace_${model}_${dataset}_len1 \
    --num_generate ${num_generate}\
    --run_generate ${run_generate}\
    --run_evaluate ${run_evaluate}\
    --parallel 0\
    --do_intervent 1\
    --sim_threshold 0.5 \
    | tee output_${model}_${dataset}_${name_tag}_${run_certain_level}.txt  
    # --majority 5 \
        # --run_certain_level ${run_certain_level} \
    

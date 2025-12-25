
model=gpt-4o-mini-ca
name_tag=4o-mini
dataset=CoderEval
num_generate=1
run_generate=1
run_evaluate=1
begin_idx=0
end_idx=170
run_certain_level=''
python /home/zlyuaj/Causal/MetaGPT/metagpt/software_causal_basedataset.py  \
    --model ${model}\
    --input_path /home/zlyuaj/Causal/MetaGPT/data/CoderEval.jsonl\
    --output_path /home/zlyuaj/Causal/MetaGPT/output/${dataset}/ \
    --dataset ${dataset}\
    --output_file_name ${dataset}_${model}_original_dataset \
    --workspace workspace_${model}_${dataset}_original_result\
    --num_generate ${num_generate}\
    --run_generate ${run_generate}\
    --run_evaluate ${run_evaluate}\
    --begin_idx ${begin_idx}\
    --end_idx  ${end_idx}\
    --parallel 0\
    | tee output_${model}_${dataset}_basedataset_idx_${begin_idx}_${end_idx}.txt  
    

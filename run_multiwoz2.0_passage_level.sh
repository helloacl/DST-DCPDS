#!/bin/bash
set -xe

data=multiwoz-update
model=model
lamb=0.5
batch=2
lr=1e-4
seed=1111
# multi-task dropout
mt_drop=0.1
output_postfix=${model}_lamb${lamb}_batch${batch}_lr${lr}_seed${seed}_mtdrop${mt_drop}_passage_level
target_slot='all'
bert_dir='/home/user/DST-DCPDS/data/pytorch_bert/.pytorch_pretrained_bert'

# train
CUDA_VISIBLE_DEVICES=2 python code/main.py --do_train --num_train_epochs 300 --data_dir data/${data} --bert_model bert-base-uncased --do_lower_case --bert_dir $bert_dir --task_name bert-gru-sumbt --model ${model} --output_dir exp/${data}/${output_postfix} --target_slot $target_slot --warmup_proportion 0.1 --learning_rate ${lr} --train_batch_size ${batch} --lamb ${lamb} --gradient_accumulation_steps 2 --eval_batch_size 16 --distance_metric euclidean --patience 15 --tf_dir tensorboard/${data}/${output_postfix} --hidden_dim 300 --max_label_length 32 --max_seq_length 64 --max_turn_length 22 --seed ${seed} --mt_drop $mt_drop --utterance_level_combine 0 --passage_level_combine 1 --mix_teaching_force 0

#CUDA_VISIBLE_DEVICES=3 python code/main.py --do_eval --num_train_epochs 300 --data_dir data/${data} --bert_model bert-base-uncased --do_lower_case --bert_dir $bert_dir --task_name bert-gru-sumbt --model ${model} --output_dir exp/${data}/${output_postfix} --target_slot $target_slot --warmup_proportion 0.1 --learning_rate ${lr} --train_batch_size ${batch} --lamb ${lamb} --gradient_accumulation_steps 1 --eval_batch_size 1 --distance_metric euclidean --patience 15 --tf_dir tensorboard/${data}/${output_postfix} --hidden_dim 300 --max_label_length 32 --max_seq_length 64 --max_turn_length 22 --utterance_level_combine 0 --passage_level_combine 1 --mix_teaching_force 0 --schedule_sampling 0

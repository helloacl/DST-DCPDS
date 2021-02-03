import pdb
import json
import os
cur_path, _ = os.path.split(os.path.abspath(__file__))

f_title = os.path.join(cur_path, 'data', 'multiwoz2.1-update', 'test_100.tsv')
f_result = os.path.join(cur_path, 'exp', 'multiwoz2.1-update', 'model_lamb0.5_batch2_lr1e-4_seed1111_mtdrop0.1_all', 'out_result.json')

with open(f_title, 'r') as f:
    for line in f:
        title = line
        break
title_list = title.strip().split('\t')[4:39]
slot_id = {}
for title_id, title_content in enumerate(title_list):
    slot_id[title_content] = title_id


with open('augment_dialog_cand.json', 'r') as f:
    f_data = json.load(f)
    dialog_list = {}
    for dialog in f_data:
        dialog_id = dialog['dialogue_id']
        if dialog_id in {'PMUL2703.json',
                         'PMUL3494.json',
                         'SNG01957.json',
                         'SNG0767.json',
                         'MUL2359.json',
                          'PMUL1920.json',
                         'PMUL2452.json',
                         'SNG1012.json',
                         'MUL2386.json',
                         'MUL1766.json',
                         'MUL1828.json'}:
            update_state = dialog['update_state']
            slot_name = update_state[0][0]
            end_turn = update_state[0][1]
            dialog_list[dialog_id] = [int(end_turn), slot_id[slot_name]]

success_num = 0
total_num = 0
available_list = []
with open(f_result, 'r') as f:
    c_result = json.load(f)
    for key_result in c_result:
        tmp0 = '{}.json'.format(key_result.split('.')[0].split('_')[0])
        tmp_end_turn, tmp_slot = dialog_list[tmp0]
        tmp2 = c_result[key_result]
        prev_turn = tmp_end_turn - 1
        prev_label = tmp2[prev_turn]['label'][tmp_slot]
        prev_pred = tmp2[prev_turn]['pred'][tmp_slot]
        end_label = tmp2[tmp_end_turn]['label'][tmp_slot]
        end_pred = tmp2[tmp_end_turn]['pred'][tmp_slot]
        if prev_label == prev_pred and end_label == end_pred:
            success_num += 1
            if tmp0 not in available_list:
                available_list.append(tmp0)
        total_num += 1
print(success_num/float(total_num))
print(available_list)
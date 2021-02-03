import pdb
import json
import os
cur_path, _ = os.path.split(os.path.abspath(__file__))

f_title = os.path.join(cur_path, 'data', 'multiwoz2.1-update', 'test.tsv')
f_result = os.path.join(cur_path, 'exp', 'multiwoz2.1-update', 'model_lamb0.5_batch2_lr1e-4_seed1111_mtdrop0.1_all', 'out_result_taxi.json')
f_target = os.path.join(cur_path, 'taxi.csv')


with open(f_title, 'r') as f:
    for line in f:
        title = line
        break
title_list = title.strip().split('\t')[4:39]
slot_id = {}
for title_id, title_content in enumerate(title_list):
    slot_id[title_content] = title_id

target_dict = {}
with open(f_target, 'r') as f:
    for line in f:
        dialog_id, dialog_turn = line.strip().split('\t')
        if dialog_id+'.json' not in target_dict:
            target_dict[dialog_id+'.json'] = []
        target_dict[dialog_id + '.json'].append(int(dialog_turn))

success_num = 0
total_num = 0
success_dialog_list = []
with open(f_result, 'r') as f:
    c_result = json.load(f)
    for key_result in c_result:
        tmp0 = '{}.json'.format(key_result.split('.')[0].split('_')[0])
        target_turn = target_dict.get(tmp0, [])
        if target_turn:
            for tmp_turn in target_turn:
                tmp2 = c_result[key_result]
                turn_info = tmp2[tmp_turn]
                pred_taxi_departure = turn_info['pred'][26]
                pred_taxi_destination = turn_info['pred'][27]
                label_taxi_departure = turn_info['label'][26]
                label_taxi_destination = turn_info['label'][27]
                if label_taxi_destination != 'none':
                    total_num += 1
                    if label_taxi_destination == pred_taxi_destination:
                        success_num += 1
                        if tmp0 not in success_dialog_list:
                            success_dialog_list.append(tmp0)
                if label_taxi_departure != 'none':
                    total_num += 1
                    if label_taxi_departure == pred_taxi_departure:
                        success_num += 1
print(success_num)
print(total_num)
print(success_num/float(total_num))



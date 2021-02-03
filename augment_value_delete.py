# coding:utf-8
import json
import os
cur_path, _ = os.path.split(os.path.abspath(__file__))
import pdb


def main():
    p_ontology = os.path.join(cur_path, 'data', 'multiwoz2.1-update', 'original')
    f_attraction = os.path.join(p_ontology, 'attraction_db.json')
    f_restaurant = os.path.join(p_ontology, 'restaurant_db.json')
    f_hotel = os.path.join(p_ontology, 'hotel_db.json')
    f_title = os.path.join(cur_path, 'data', 'multiwoz2.1-update', 'test.tsv')
    with open(f_title, 'r') as f:
        for line in f:
            title = line
            break
    with open(f_attraction, 'r') as f:
        c_attraction = json.load(f)
    with open(f_restaurant, 'r') as f:
        c_restaurant = json.load(f)
    with open(f_hotel , 'r') as f:
        c_hotel = json.load(f)

    augment_dialog = []
    with open('augment_dialog_cand.json', 'r') as f:
        f_data = json.load(f)
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
                general_tmp = []
                update_state = dialog['update_state']
                domain = update_state[0][0].split('-')[0]
                end_turn = update_state[0][1]
                if domain == 'attraction':
                    general_tmp = replace_attraction(dialog_id, dialog, c_attraction, end_turn)
                if domain == 'restaurant':
                    general_tmp = replace_restaurant(dialog_id, dialog, c_restaurant, end_turn)
                if domain == 'hotel':
                    general_tmp = replace_hotel(dialog_id, dialog, c_hotel, end_turn)
                augment_dialog.extend(general_tmp)
    augment_dialog_str = title
    for augments in augment_dialog:
        for turn in augments:
            augment_dialog_str += turn

    with open(os.path.join(cur_path, 'data', 'multiwoz2.1-update', 'update_test.tsv'), 'w') as f:
        f.write(augment_dialog_str)


def replace_hotel(dialog_id, dialog, ontology, end_turn):
    general_tmp = []
    dialog_id_list = dialog_id.split('.')
    slot_names = list(ontology[0].keys())
    slot_names = [x for x in slot_names if x not in {'location', 'signature', 'id', 'introduction',  "takesbookings", 'price'}]
    for candidate_id, candidate in enumerate(ontology):
        new_dialog_id = '{}_{}.{}'.format(dialog_id_list[0], candidate_id, dialog_id_list[1])
        dialog_history = dialog['history']
        tmp_general = []
        for origin_id, origin_content in enumerate(dialog_history):
            if origin_id > int(end_turn):
                break
            tmp = origin_content[0]
            for slot_name in slot_names:
                target_slot = '[{}-{}]'.format('hotel', slot_name)
                replace_value = candidate.get(slot_name, '?').lower()
                if replace_value == 'arbury lodge guesthouse':
                    replace_value = 'arbury lodge guest house'
                if replace_value == 'guesthouse':
                    replace_value = 'guest house'
                tmp = tmp.replace(target_slot, replace_value)
            tmp_list = tmp.split('\t')
            tmp_list[0] = new_dialog_id
            tmp = '\t'.join(tmp_list)
            tmp_general.append(tmp)
        general_tmp.append(tmp_general)
    return general_tmp


def replace_restaurant(dialog_id, dialog, ontology, end_turn):
    general_tmp = []
    dialog_id_list = dialog_id.split('.')
    slot_names = list(ontology[0].keys())
    slot_names = [x for x in slot_names if x not in {'location', 'signature', 'id', 'introduction', 'type'}]
    for candidate_id, candidate in enumerate(ontology):
        new_dialog_id = '{}_{}.{}'.format(dialog_id_list[0], candidate_id, dialog_id_list[1])
        dialog_history = dialog['history']
        tmp_general = []
        for origin_id, origin_content in enumerate(dialog_history):
            if origin_id > int(end_turn):
                break
            tmp = origin_content[0]
            for slot_name in slot_names:
                target_slot = '[{}-{}]'.format('restaurant', slot_name)
                replace_value = candidate.get(slot_name, '?').lower()
                if replace_value == 'ask restaurant':
                    replace_value = 'ask'
                if replace_value == 'meze bar':
                    replace_value = 'meze bar restaurant'
                tmp = tmp.replace(target_slot, replace_value)
            tmp_list = tmp.split('\t')
            tmp_list[0] = new_dialog_id
            tmp = '\t'.join(tmp_list)
            tmp_general.append(tmp)
        general_tmp.append(tmp_general)
    return general_tmp


def replace_attraction(dialog_id, dialog, ontology, end_turn):
    general_tmp = []
    dialog_id_list = dialog_id.split('.')
    slot_names = list(ontology[0].keys())
    slot_names = [x for x in slot_names if x not in {'location', 'openhours', 'id'}]
    for candidate_id, candidate in enumerate(ontology):
        new_dialog_id = '{}_{}.{}'.format(dialog_id_list[0], candidate_id, dialog_id_list[1])
        dialog_history = dialog['history']
        tmp_general = []
        for origin_id, origin_content in enumerate(dialog_history):
            if origin_id > int(end_turn):
                break
            tmp = origin_content[0]
            for slot_name in slot_names:
                target_slot = '[{}-{}]'.format('attraction', slot_name)
                replace_value = candidate[slot_name].lower()
                if replace_value == 'swimmingpool':
                    replace_value = 'swimming pool'
                if replace_value == 'mutliple sports':
                    replace_value = 'multiple sports'
                if replace_value == 'concerthall':
                    replace_value = 'concert hall'
                tmp = tmp.replace(target_slot, replace_value)
            tmp_list = tmp.split('\t')
            tmp_list[0] = new_dialog_id
            tmp = '\t'.join(tmp_list)
            tmp_general.append(tmp)
        general_tmp.append(tmp_general)
    return general_tmp


if __name__ == '__main__':
    main()

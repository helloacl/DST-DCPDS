# coding: utf-8
import os
import pdb
import json
cur_path, _ = os.path.split(os.path.abspath(__file__))


ontology = ['attraction-area', 'attraction-name', 'attraction-type',
            'bus-day', 'bus-departure', 'bus-destination', 'bus-leaveAt',
            'hospital-department',
            'hotel-area', 'hotel-book day', 'hotel-book people', 'hotel-book stay', 'hotel-internet', 'hotel-name', 'hotel-parking', 'hotel-price range', 'hotel-stars', 'hotel-type',
            'restaurant-area', 'restaurant-book day', 'restaurant-book people', 'restaurant-book time', 'restaurant-food', 'restaurant-name', 'restaurant-price range', 'taxi-arrive by',
            'taxi-departure', 'taxi-destination', 'taxi-leave at',
            'train-arrive by', 'train-book people', 'train-day', 'train-departure', 'train-destination', 'train-leave at']

ontology_domain = [x.split('-')[0] for x in ontology]
ontology_index = {}
for x, y in enumerate(ontology_domain):
    ontology_index[x] = y

f_title = os.path.join(cur_path, 'data', 'multiwoz2.1-update', 'test.tsv')
with open(f_title, 'r') as f:
    for line in f:
        title = line
        break


def main(f_name):
    global_dialog = {}
    tmp_dict = {
        'domain': [],
        'history': [],
    }

    with open(f_name, 'r') as f:
        for line_index, line in enumerate(f):
            if line_index == 0:
                slot_name = line.strip().split('\t')[4:]
                slot_name = [x for x in slot_name if '-transition' not in x]
                slot_num = len(slot_name)
            else:
                line_content = line.strip().split('\t')
                dialog_id = line_content[0]
                dialog_turn = line_content[1]
                slot_value = line_content[4:39]
                current_domain = [ontology_index[x] for x, y in enumerate(slot_value) if y != 'none']
                if dialog_turn == '0':
                    # update target dialog
                    if 'taxi' in tmp_dict['domain'] and len(tmp_dict['domain']) > 1:
                        global_dialog[tmp_dict['dialog_id']] = tmp_dict
                    tmp_dict = {
                        'domain': [],
                        'history': [line],
                        'dialog_id': dialog_id
                    }
                else:
                    tmp_dict['history'].append(line)
                for domain in current_domain:
                    if domain not in tmp_dict['domain']:
                        tmp_dict['domain'].append(domain)

    augment_dialog_str = title
    for augments in global_dialog:
        for turn in global_dialog[augments]['history']:
            augment_dialog_str += turn

    with open(os.path.join(cur_path, 'data', 'multiwoz2.1-update', 'select_taxi.tsv'), 'w') as f:
        f.write(augment_dialog_str)


if __name__ == '__main__':
    f_test = os.path.join(cur_path, 'data', 'multiwoz2.1-update', 'test.tsv')
    main(f_test)

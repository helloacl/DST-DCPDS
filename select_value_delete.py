# coding: utf-8
import os
import pdb
import json
cur_path, _ = os.path.split(os.path.abspath(__file__))


def main(f_name):
    slot_num = 0
    tmp_dialog = {}
    general_dialogs = []
    with open(f_name, 'r') as f:
        for line_index, line in enumerate(f):
            if line_index == 0:
                slot_name = line.strip().split('\t')[4:]
                slot_name = [x for x in slot_name if '-transition' not in x]
                slot_num = len(slot_name)
            else:
                line_content = line.strip().split('\t')
                dialog_id = line_content[0]
                dialog_id_turn = line_content[1]
                slot_value = line_content[4: 4+slot_num]
                if dialog_id_turn == '0':
                    # save record
                    if 'update_state' in tmp_dialog and tmp_dialog['update_state']:
                        general_dialogs.append(tmp_dialog)
                    tmp_dialog = {}
                    tmp_dialog['dialogue_id'] = dialog_id
                    tmp_dialog['history'] = [[line, dialog_id_turn]]
                    tmp_dialog['last_state'] = slot_value
                    tmp_dialog['update_state'] = []
                else:
                    for tuple_id, tuple_value in enumerate(zip(tmp_dialog['last_state'], slot_value)):
                        if tuple_value[0] != tuple_value[1] and tuple_value[0] != 'none' and tuple_value[1] == 'none':
                            tmp_dialog['update_state'].append([slot_name[tuple_id], dialog_id_turn, tuple_value[0], tuple_value[1]])
                    tmp_dialog['history'].append([line, dialog_id_turn])
                    tmp_dialog['last_state'] = slot_value
    output = []
    with open('selected_dialog_cand.json', 'w') as f:
        for dialogue in general_dialogs:
            dialogue.pop('last_state')
            output.append(dialogue)
        json.dump(output, f, indent=4)


if __name__ == '__main__':
    f_test = os.path.join(cur_path, 'data', 'multiwoz2.1-update', 'test.tsv')
    main(f_test)
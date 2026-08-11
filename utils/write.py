import os
import json


design_list = ['adaptec1', 'adaptec2', 'adaptec3', 'adaptec4', 'bigblue1', 'bigblue2', 'bigblue3', 'bigblue4', ]
ref_aux_dict = {'adaptec1': 'adaptec1_graphplanner.pl', 
                'adaptec2': 'adaptec2.nonlinear.pl', 
                'adaptec3': 'adaptec3.2.pl', 
                'adaptec4': 'adaptec4.pl', 
                'bigblue1': 'bigblue1.pl', 
                'bigblue2': 'bigblue2.pl', 
                'bigblue3': 'bigblue3.pl', 
                'bigblue4': 'bigblue4.pl'}


def write_bench_files(design, sample, ref_path, save_path, map_path, sign='temp'):
    with open(f'{map_path}/{design}.json', 'r') as f:
        node_name_list = json.load(f)

    ref_pl_file_path = f'{ref_path}/{design}/{ref_aux_dict[design]}'
    write_pl_file_path = f'{save_path}/{design}/{design}_{sign}.pl'
    if not os.path.exists(f'{save_path}/{design}'):
        os.makedirs(f'{save_path}/{design}')

    with open(ref_pl_file_path, 'r') as ref_pl_file, open(write_pl_file_path, 'w') as write_pl_file:
        write_pl_file.write('UCLA pl 1.0\n')
        write_pl_file.write('\n')
        while True:
            line =  ref_pl_file.readline()
            if not line: 
                break
            words = line.split()
            if len(words) < 5:
                continue
            if words[0][0] != 'o':
                continue
            if words[0] not in node_name_list:
                write_pl_file.write(line)
            elif words[-1] == '/FIXED':
                write_pl_file.write(line)
            else:
                index = node_name_list.index(words[0])
                write_pl_file.write(f'{words[0]}\t{sample[index,0].int()}\t{sample[index,1].int()}\t: N\n')

    ref_aux_file_path = f'{ref_path}/{design}/{design}.aux'
    write_aux_file_path = f'{save_path}/{design}/{design}_{sign}.aux'

    with open(ref_aux_file_path, 'r') as ref_aux_file, open(write_aux_file_path, 'w') as write_aux_file:
        while True:
            line =  ref_aux_file.readline()
            if not line: 
                break
            words = line.split()
            line = f'{words[0]} {words[1]} {words[2]} {words[3]} {words[4]} {design}_{sign}.pl {words[6]}'
            write_aux_file.write(line)

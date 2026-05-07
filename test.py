#!/usr/bin/env python3

import pathlib as pth
from pprint import pprint
from typing import Dict
data_path = pth.Path("./outputs/")
files = [file for file in data_path.rglob("*") if file.is_file()]

total_A = {}
total_B = {}


def read_maps(path):
    maps = dict()
    lens: Dict[int, int] = dict()
    shifts = dict()
    with open(path, "r") as handle:
        for line in handle:
            if ":" in line:
                continue
            entries = line.strip().split("]")
            for index, entry in enumerate(entries):
                entries[index] = entry.replace('[', '')
            source_proc = int(entries[0])
            target_proc = int(entries[1].split(',')[0])
            dict_key = (source_proc, target_proc)
            local_size = int(entries[1].split(',')[1])
            data = [int(x) for x in entries[2].strip().split()]
            if dict_key in maps:
                maps[dict_key] += data
            else:
                maps[dict_key] = data
            lens[source_proc] = local_size
    keys = list(lens.keys())
    for key in keys:
        if key == 0:
            shifts[0] = 0
        elif key == 1:
            shifts[1] = lens[0]
        else:
            shifts[key] = shifts[key-1] + lens[key-1]
    for key in maps.keys():
        new_data = [datapoint + shifts[key[0]] for datapoint in maps[key]]
        maps[key] = new_data
    global_map = dict()
    for key in maps:
        if key[1] in global_map:
            global_map[key[1]] += maps[key]
        else:
            global_map[key[1]] = maps[key]
    for key in global_map.keys():
        global_map[key].sort()
    return global_map


global_map_A = read_maps("./outputs/map_8_3.dat")
global_map_B = read_maps("./outputs/map_4_3.dat")
for key in global_map_A.keys():
    print(key, ": ", global_map_A[key] == global_map_B[key])

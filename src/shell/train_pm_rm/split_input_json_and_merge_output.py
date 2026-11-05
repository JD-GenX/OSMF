

from __future__ import absolute_import

import json
import argparse
import os
import datetime


def split(args):
    base_data_path = args.base_data_path
    patch_size = args.patch_size
    save_path = args.save_path

    with open(base_data_path, 'r') as f:
        total_samples = json.load(f)

    if len(total_samples) >= patch_size:
        split_res = []
        step = int(len(total_samples) / patch_size)
        print(step)
        start = 0
        for i in range(patch_size):
            end = start + step
            if i + 1 == patch_size:
                end = len(total_samples)
            print('part {}, [{}, {}]'.format(i + 1, start, end))
            split_res_patch = total_samples[int(start):int(end)]
            split_res.append(split_res_patch)
            start = end
        i = 0
        for split_res_patch in split_res:
            save_out_json_name = os.path.join(save_path, 'input_split_{}.json'.format(i))
            with open(save_out_json_name, 'w') as f:
                json.dump(split_res_patch, f, ensure_ascii=False, indent=4)
            i += 1
    else:
        save_out_json_name = os.path.join(save_path, 'input_split_0.json')
        with open(save_out_json_name, 'w') as f:
            json.dump(total_samples, f, ensure_ascii=False, indent=4)


def merge(args):
    base_data_path = args.base_data_path
    patch_size = args.patch_size
    save_path = args.save_path
    total_samples = []
    print('sd_prompt_split data merging...')
    for file in os.listdir(save_path):
        if file.startswith('sd_prompt_split_') and file.endswith('.json'):
            split_json_path = os.path.join(save_path, file)
            with open(split_json_path, 'r') as f:
                split_json = json.load(f)
                total_samples.extend(split_json)
    print('sd_prompt_split data merge complete')
    print('sd_prompt data length after merge:', len(total_samples))
    save_out_json_name = os.path.join(save_path, 'sd_prompt.json')
    with open(save_out_json_name, 'w') as f:
        json.dump(total_samples, f, ensure_ascii=False, indent=4)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_size", type=int,
                        default=4)
    parser.add_argument("--base_data_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--tag", type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)
    if args.tag == "split":
        split(args)
    elif args.tag == "merge":
        merge(args)



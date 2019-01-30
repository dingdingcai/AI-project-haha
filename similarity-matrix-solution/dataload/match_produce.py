#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__title__ = 'match_produce'
__author__ = 'fangwudi'
__time__ = '18-11-17 13:47'

code is far away from bugs 
     ┏┓   ┏┓
    ┏┛┻━━━┛┻━┓
    ┃        ┃
    ┃ ┳┛  ┗┳ ┃
    ┃    ┻   ┃
    ┗━┓    ┏━┛
      ┃    ┗━━━━━┓
      ┃          ┣┓
      ┃          ┏┛
      ┗┓┓┏━━┳┓┏━━┛
       ┃┫┫  ┃┫┫
       ┗┻┛  ┗┻┛
with the god animal protecting
     
"""
import os
import json
from collections import defaultdict

def main():
    target_dir = '/home/ubuntu/WorkSpace/DataSpace/M3_counting/data_source/task87_6'
    # fisrt level
    walk = os.walk(target_dir)
    dirpath, dirnames, filenames = next(walk)
    # second level
    for dirname in dirnames:
        walk = os.walk(os.path.join(dirpath, dirname))
        second_dirpath, second_dirnames, second_filenames = next(walk)
        # third level
        for second_dirname in second_dirnames:
            # init
            store_list = []
            store_list_simple_take = []
            before_num = 0
            before_data = None
            walk = os.walk(os.path.join(second_dirpath, second_dirname))
            third_dirpath, third_dirnames, third_filenames = next(walk)
            third_filenames = sorted(third_filenames)
            x_dirpath = os.path.join(target_dir, third_dirpath)
            begin_flag = False
            reverse_flag = False
            for x_filename in third_filenames:
                if x_filename.endswith('.jpg'):
                    data_num = read_json(os.path.join(x_dirpath, x_filename[:-4] + '.json'))
                    now_num = sum(data_num.values())
                    data_path = os.path.join(x_dirpath, x_filename[:-4] + '_mp.json')
                    data_mp = read_json(data_path)
                    if data_mp is None:
                        print(data_path)
                    else:
                        data_mp = dict([(k, str2float(v)) for k, v in data_mp.items() if len(v) > 0])
                        now_data = [data_path, data_mp]
                        if begin_flag:
                            if now_num >= before_num:
                                # so this one should be A image
                                a, b = now_data, before_data
                            else:
                                # so this one should be B image
                                a, b = before_data, now_data
                                # for simple_take
                                reverse_flag = True
                            # calculate labels
                            a_change_s, b_change_s, a_same_s, b_same_s, a_change_num, b_change_num = match_s(a[1], b[1])
                            match_data = {'a_path': a[0], 'a_data': a[1], 'b_path': b[0], 'b_data': b[1],
                                          'a_change': a_change_s, 'b_change': b_change_s, 'a_same': a_same_s,
                                          'b_same': b_same_s, 'a_change_num': a_change_num, 'b_change_num': b_change_num}
                            store_list.append(match_data)
                            if reverse_flag:
                                store_list_simple_take.append(match_data)
                        else:
                            begin_flag = True
                        before_data = now_data
                        before_num = now_num
            # save store_list as json
            save_path = os.path.join(target_dir, second_dirpath, second_dirname + '_match.json')
            with open(save_path, 'w') as outfile:
                json.dump(store_list, outfile)
            save_path_simple_take = os.path.join(target_dir, second_dirpath, second_dirname + '_match_simple_take.json')
            with open(save_path_simple_take, 'w') as outfile:
                json.dump(store_list_simple_take, outfile)


def str2float(x_list):
    return [[float(i) for i in j] for j in x_list]


def match_s(a, b, x_threshold=0.06, y_threshold=0.06):
    # a_change_s: the skus within a changed related to b
    # b_change_s: the skus within b changed related to a 
    # a_same_s: the rest of skus within a unchanged related to b
    # b_same_s: the rest of skus within b unchanged related to a
    a_change_s, b_change_s, a_same_s, b_same_s = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    a_change_num, b_change_num = 0, 0
    for b_sku in b:
        if b_sku in a:
            a_change, b_change, a_same, b_same = match_one(a[b_sku], b[b_sku], x_threshold=x_threshold, y_threshold=y_threshold)
            a_change_s[b_sku] = a_change
            a_change_num += len(a_change)
            b_change_s[b_sku] = b_change
            b_change_num += len(b_change)
            a_same_s[b_sku] = a_same
            b_same_s[b_sku] = b_same
        else:
            # new sku in b, as b change
            b_change_s[b_sku] = b[b_sku][:]
            b_change_num += len(b[b_sku])
    # collect a not in b, as a change
    for a_sku in a:
        if a_sku not in b:
            a_change_s[a_sku] = a[a_sku][:]
            a_change_num += len(a[a_sku])
    return a_change_s, b_change_s, a_same_s, b_same_s, a_change_num, b_change_num

def match_one(a, b, x_threshold=0.06, y_threshold=0.06):
    a_todos, a_ischanges = [1] * len(a), [0] * len(a)
    b_todos, b_ischanges = [1] * len(b), [0] * len(b)
    # do with b
    while any(b_todos):
        for i, b_todo in enumerate(b_todos):
            if b_todo:
                b_point = b[i]
                bx, by = b_point[0], b_point[1]
                # find_a_zone
                find_a_zone = []
                for j, a_todo in enumerate(a_todos): #  go through all the skus in a
                    if a_todo:
                        a_point = a[j]
                        if abs(a_point[0]-bx) <= x_threshold and abs(a_point[1]-by) <= y_threshold:
                            find_a_zone.append(j)
                if len(find_a_zone) == 0:  # the sku in b cannot be matched with any sku in a, which means it changed
                    # not found, so b_change
                    b_ischanges[i] = 1
                    b_todos[i] = 0
                else:
                    # find nearest one
                    min_a = find_a_zone[0]
                    min_r = (a[min_a][0]-bx) ** 2 + (a[min_a][1]-by) ** 2
                    if len(find_a_zone) > 1:
                        for other_a in find_a_zone[1:]:
                            r = (a[other_a][0]-bx) ** 2 + (a[other_a][1]-by) ** 2
                            if r < min_r:
                                min_r = r
                                min_a = other_a
                    # find nearest b, double check
                    min_r_back = min_r # the point of a_sku which is the closest to b_sku
                    min_b_back = None
                    for k, b_todo_back in enumerate(b_todos):
                        if b_todo_back:
                            b_point_back = b[k]
                            r_back = (b_point_back[0]-a[min_a][0]) ** 2 + (b_point_back[1]-a[min_a][1]) ** 2
                            if r_back < min_r_back:
                                min_b_back = k
                                min_r_back = r_back
                    if min_b_back is None:
                        # not found, so these two matches
                        b_ischanges[i] = 0
                        b_todos[i] = 0 #unchanged
                        a_ischanges[min_a] = 0
                        a_todos[min_a] = 0
                    else:
                        # found, so make find_a and back_b matches
                        b_ischanges[min_b_back] = 0
                        b_todos[min_b_back] = 0
                        a_ischanges[min_a] = 0
                        a_todos[min_a] = 0
    # do with a
    for j, a_todo in enumerate(a_todos):
        if a_todo:
            a_ischanges[j] = 1
            a_todos[j] = 0
    # post process
    a_change, b_change, a_same, b_same = [], [], [], []
    for i, b_ischange in enumerate(b_ischanges):
        if b_ischange:
            b_change.append(b[i])
        else:
            b_same.append(b[i])
    for j, a_ischange in enumerate(a_ischanges):
        if a_ischange:
            a_change.append(a[j])
        else:
            a_same.append(a[j])
    # a_change: the a_sku cannot be found in the same position within b;
    # b_change: the b_sku cannot be found in the same position within a;
    # 
    return a_change, b_change, a_same, b_same 


def read_json(file_name):
    f = open(file_name)
    r = f.read()
    f.close()
    if not r:
        return None
    j = json.loads(r)
    return j

if __name__ == '__main__':
    main()

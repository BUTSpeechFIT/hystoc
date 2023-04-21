# Copied out from the PERO project to avoid a heavy dependence
# https://github.com/DCGM/pero-ocr/blob/master/pero_ocr/decoding/confusion_networks.py
# https://github.com/DCGM/pero-ocr/blob/master/pero_ocr/sequence_alignment.py

import numpy as np


def levenshtein_alignment_path(source, target, sub_cost=1, ins_cost=1, del_cost=1, empty_symbol=None):
    target = np.array(target)
    backtrack = np.ones((len(source) + 1, len(target) + 1))
    backtrack[0] = -1
    dist = np.arange(len(target) + 1) * ins_cost
    for ii, s in enumerate(source):
        cost4sub = dist[:-1] + (target != s) * sub_cost
        dist += del_cost
        where_sub = cost4sub < dist[1:]
        dist[1:][where_sub] = cost4sub[where_sub]
        backtrack[ii + 1, 1:][where_sub] = 0
        for jj in range(len(dist) - 1):
            if dist[jj + 1] > dist[jj] + ins_cost:
                dist[jj + 1] = dist[jj] + ins_cost
                backtrack[ii + 1, jj + 1] = -1
    src_pos = len(source)
    tar_pos = len(target)

    align = []
    while tar_pos > 0 or src_pos > 0:
        where = backtrack[src_pos, tar_pos]
        if where >= 0:
            src_pos -= 1
        if where <= 0:
            tar_pos -= 1
        align.append(where)
    return list(reversed(align))


def get_pivot(cn):
    pivot = []
    for sausage in cn:
        pivot.append(sorted(sausage, key=lambda k: sausage[k], reverse=True)[0])

    return pivot


def add_hypothese(cn, transcript, score):
    if cn == []:
        for symbol in transcript:
            cn.append({symbol: score})

        return cn

    pivot = get_pivot(cn)
    alignment = levenshtein_alignment_path(list(transcript), pivot)
    cn_total_weight = sum(sum(position.values()) for position in cn) / len(cn)

    cn_pointer = 0
    tr_pointer = 0
    for direction in alignment:
        if direction == -1:  # move in the confusion network
            if None in cn[cn_pointer]:
                cn[cn_pointer][None] += score
            else:
                cn[cn_pointer][None] = score
            cn_pointer += 1
        elif direction == 0:  # move in both
            tr_sym = transcript[tr_pointer]
            if tr_sym in cn[cn_pointer]:
                cn[cn_pointer][tr_sym] += score
            else:
                cn[cn_pointer][tr_sym] = score
            tr_pointer += 1
            cn_pointer += 1
        elif direction == 1:  # move in the confusion network
            tr_sym = transcript[tr_pointer]
            if cn_pointer == len(cn):
                cn.append({None: cn_total_weight, tr_sym: score})
            else:
                cn = cn[:cn_pointer] + [{None: cn_total_weight, tr_sym: score}] + cn[cn_pointer:]
                cn_pointer += 1
            tr_pointer += 1
        else:
            raise RuntimeError("Got unexpected direction {}".format(direction))

    return cn


def normalize_cn(cn):
    for i in range(len(cn)):
        sausage_normalizer = sum(cn[i].values())
        for symbol in cn[i]:
            cn[i][symbol] /= sausage_normalizer

    return cn


def best_cn_path(cn):
    return [sorted(position.items(), key=lambda pair: pair[1], reverse=True)[0] for position in cn]

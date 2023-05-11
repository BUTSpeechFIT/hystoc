from typing import Dict


def write_pctm(out_f, seg_name, best_path):
    words_reprs = [f'{pos[0]} {pos[1]}' for pos in best_path]
    out_f.write(f'{seg_name} {" ".join(words_reprs)}\n')


def write_ctm(out_f, seg_name, best_path):
    word_dur = 0.15

    for i, (word, conf) in enumerate(best_path):
        out_f.write(f'{seg_name} {1} {i*word_dur:.2f} {word_dur} {word} {conf}\n')


output_formats = {
    'ctm': write_ctm,
    'pctm': write_pctm,
}


def get_hyp_score_pairs(hyps, scores):
    return [(hyps[variant], scores[variant]) for variant in scores.keys() if variant in hyps]


def split_nbest_key(key):
    fields = key.split('-')
    segment = '-'.join(fields[:-1])
    trans_id = fields[-1]

    return segment, trans_id


def load_scores_dict(scores_f):
    all_segments = {}
    curr_seg = None
    segment_utts_scores: Dict[str, float] = {}

    for line in scores_f:
        fields = line.split()
        assert len(fields) == 2

        segment, trans_id = split_nbest_key(fields[0])

        if not curr_seg:
            curr_seg = segment

        if segment != curr_seg:
            all_segments[curr_seg] = segment_utts_scores
            curr_seg = segment
            segment_utts_scores = {}

        segment_utts_scores[trans_id] = float(fields[1])

    if curr_seg is not None:
        all_segments[curr_seg] = segment_utts_scores

    return all_segments


def load_hyps_dict(scores_f):
    all_segments = {}
    curr_seg = None
    segment_utts_hyps: Dict[str, str] = {}

    for line in scores_f:
        fields = line.split()

        segment, trans_id = split_nbest_key(fields[0])

        if not curr_seg:
            curr_seg = segment

        if segment != curr_seg:
            all_segments[curr_seg] = segment_utts_hyps
            curr_seg = segment
            segment_utts_hyps = {}

        segment_utts_hyps[trans_id] = ' '.join(fields[1:])

    if curr_seg is not None:
        all_segments[curr_seg] = segment_utts_hyps

    return all_segments

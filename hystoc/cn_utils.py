import math
from typing import List, Dict
import logging

from hystoc.confusion_networks import add_hypothese, normalize_cn


def cn_from_segment(scored_hyps, temperature, only_best=False):
    cn: List[Dict[str, float]] = []

    sorted_hyps = sorted(scored_hyps, key=lambda pair: pair[1], reverse=True)
    top_score = sorted_hyps[0][1]
    logging.info(f'Highest score: {top_score}')
    sorted_hyps = [(transcript, score-top_score) for transcript, score in sorted_hyps]

    for transcript, score in sorted_hyps:
        add_hypothese(cn, transcript.split(), math.exp(score / temperature))

        if only_best:
            break  # Stopping once the first hypothesis has been added

    return normalize_cn(cn)


def filter_nones(best_path):
    return [pos for pos in best_path if pos[0] is not None]

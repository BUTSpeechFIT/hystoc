#!/usr/bin/env python3
import argparse
import logging
import itertools
import math
import numpy as np
import sys
from typing import List, Dict

from hystoc.confusion_networks import add_hypothese, normalize_cn, best_cn_path
from hystoc.io_utils import output_formats, get_hyp_score_pairs, load_scores_dict, load_hyps_dict
from hystoc.cn_utils import cn_from_segment, filter_nones


# Extracted from the PyPi package more_itertools
# on the purpose of making Hystoc rely on standard Python only.
def roundrobin(*iterables):
    """Yields an item from each iterable, alternating between them.

        >>> list(roundrobin('ABC', 'D', 'EF'))
        ['A', 'D', 'E', 'B', 'F', 'C']

    This function produces the same output as :func:`interleave_longest`, but
    may perform better for some inputs (in particular when the number of
    iterables is small).

    """
    # Recipe credited to George Sakkis
    pending = len(iterables)
    nexts = itertools.cycle(iter(it).__next__ for it in iterables)
    while pending:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            pending -= 1
            nexts = itertools.cycle(itertools.islice(nexts, pending))


def round_robin_align(scored_hyps_per_segment, temperature):
    for segment in scored_hyps_per_segment:
        segment[:] = sorted(segment, key=lambda pair: pair[1], reverse=True)

    cn: List[Dict[str, float]] = []
    for transcript, score in roundrobin(*scored_hyps_per_segment):
        add_hypothese(cn, transcript.split(), math.exp(temperature * score))

    return normalize_cn(cn)


def prenormalize_log_scores(variants):
    top_score = max(s for h, s in variants)
    variants[:] = [(h, s - top_score) for h, s in variants]


def normalize_log_scores(variants, do_sort=False):
    top_score = max(s for h, s in variants)
    pre_norm_scores = [(h, s - top_score) for h, s in variants]
    normalizer = np.logaddexp.reduce([s for _, s in pre_norm_scores])
    variants[:] = [(h, s - normalizer) for h, s in pre_norm_scores]


def fusion_direct(scored_hyps, temperature):
    # treat all directly. Unlikely to work for largely different systems
    all_variants_flat = list(itertools.chain(*scored_hyps))
    return cn_from_segment(all_variants_flat, temperature, only_best=False)


def fusion_zero_per_system(scored_hyps, temperature):
    # make the largest element have score 0.0 first
    for system in scored_hyps:
        prenormalize_log_scores(system)
    all_variants_flat = list(itertools.chain(*scored_hyps))
    return cn_from_segment(all_variants_flat, temperature, only_best=False)


def fusion_normalize_per_system(scored_hyps, temperature):
    # apply softmax on n-best scores of each model independently
    for system in scored_hyps:
        normalize_log_scores(system)
    all_variants_flat = list(itertools.chain(*scored_hyps))
    return cn_from_segment(all_variants_flat, temperature, only_best=False)


def fusion_normalized_round_robin(scored_hyps, temperature):
    # apply softmax on n-best scores of each model independently
    # the align as A1, B1, C1, A2, B2, C2, A3, ...
    for system in scored_hyps:
        normalize_log_scores(system)
    return round_robin_align(scored_hyps, temperature)


fusion_methods = {
    'direct': fusion_direct,
    'zero-per-system': fusion_zero_per_system,
    'normalize-per-system': fusion_normalize_per_system,
    'normalized-round-robin': fusion_normalized_round_robin,
}


def merge(hyps_and_scores, fusion_method, temperature):
    nb_nonmatched = 0

    all_keys = itertools.chain(*[list(s.keys()) for s, h in hyps_and_scores])
    all_segments = sorted(list(set(all_keys)))

    best_paths = {}

    for seg_name in all_segments:
        logging.info(f'Processing {seg_name}')
        relevant_systems = [(s[seg_name], h[seg_name]) for s, h in hyps_and_scores if seg_name in s]
        assert len(relevant_systems) > 0

        logging.info(f'Segment present in {len(relevant_systems)} systems')
        logging.info(f'Variant numbers: {[len(s) for s, h in relevant_systems]}')

        scored_hyps = []
        for score, hyp in relevant_systems:
            symm_diff_size = len(score.keys() - hyp.keys()) + len(hyp.keys() - score.keys())
            if symm_diff_size > 0:
                nb_nonmatched += symm_diff_size
                logging.error(f'Segment {seg_name} had {symm_diff_size} unmatched scores')

            scored_hyps.append(get_hyp_score_pairs(hyp, score))
        logging.debug(f'Lengths: {len(scored_hyps)} {[len(system) for system in scored_hyps]} {scored_hyps}')

        cn = fusion_method(scored_hyps, temperature)

        best_paths[seg_name] = filter_nones(best_cn_path(cn))

    return best_paths, nb_nonmatched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logging-level', help='Logging level as per standard logging module')
    parser.add_argument('--temperature', type=float, default=1.0, help='multiplies log-probs before exponentiation')
    parser.add_argument('--method', choices=list(fusion_methods.keys()), default='normalize-per-system', help='How to fuse the systems')
    parser.add_argument('--confidence-file', help='If none is given, standard output is used')
    parser.add_argument('--output-format', choices=list(output_formats.keys()), default='pctm', help='How to present the confidences')
    parser.add_argument('hyps_scores_files', nargs='+', help="A list of pairs, organized as: <hyps A> <scores A> <hyps B> <scores B> ...")
    args = parser.parse_args()

    if len(args.hyps_scores_files) % 2 != 0:
        raise ValueError(f'Hypotheses and scores need to come in pairs, got: {args.hyps_scores_files}')

    logging.basicConfig(format='[%(levelname)s] %(asctime)s - %(name)s - %(message)s', level=args.logging_level)

    if args.temperature < 0.0:
        raise ValueError(f'Temperatures below zero make no sense (got {args.temperature})')

    system_outputs = []
    for i in range(0, len(args.hyps_scores_files), 2):
        with open(args.hyps_scores_files[i]) as f:
            scores = load_scores_dict(f)

        with open(args.hyps_scores_files[i+1]) as f:
            hyps = load_hyps_dict(f)

        system_outputs.append((scores, hyps))

        if scores.keys() != hyps.keys():
            logging.error('Not matching segments!')
            logging.error('In files {args.hyps_scores_files[i]} and {args.hyps_scores_files[i+1]}')
            logging.error(f'Only in scores: {scores.keys() - hyps.keys()}')
            logging.error(f'Only in hypotheses: {hyps.keys() - scores.keys()}')

            exit(1)

    fusion_method = fusion_methods[args.method]
    best_paths, nb_nonmatched = merge(system_outputs, fusion_method, args.temperature)
    write_method = output_formats[args.output_format]
    if args.confidence_file:
        with open(args.confidence_file, 'w') as out_f:
            for seg_name, best_path in best_paths.items():
                write_method(out_f, seg_name, best_path)
    else:
        for seg_name, best_path in best_paths.items():
            write_method(sys.stdout, seg_name, best_path)

    if nb_nonmatched > 0:
        logging.warning(f'There was a total of {nb_nonmatched} non matched scores')


if __name__ == '__main__':
    main()

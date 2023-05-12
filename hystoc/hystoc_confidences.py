#!/usr/bin/env python3
import argparse
import logging
import sys

from hystoc.confusion_networks import best_cn_path
from hystoc.io_utils import output_formats, get_hyp_score_pairs, load_scores_dict, load_hyps_dict
from hystoc.cn_utils import cn_from_segment, filter_nones


def get_token_confidences(score, hyp, temperature, dummy=False):
    cn = cn_from_segment(get_hyp_score_pairs(hyp, score), temperature, dummy)

    return filter_nones(best_cn_path(cn))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logging-level', help='Logging level as per standard logging module')
    parser.add_argument('--temperature', type=float, default=1.0, help='multiplies log-probs before exponentiation')
    parser.add_argument('--dummy', action='store_true', help='Only produce the best hypothesis with confidences 1.0')
    parser.add_argument('--output-format', choices=list(output_formats.keys()), default='pctm', help='How to present the confidences')
    parser.add_argument('hyp_file')
    parser.add_argument('scores_file')
    parser.add_argument('confidence_file', nargs='?', help='If no output file is given, standard output is used')
    args = parser.parse_args()

    logging.basicConfig(format='[%(levelname)s] %(asctime)s -  %(message)s', level=args.logging_level)

    if args.temperature < 0.0:
        raise ValueError(f'Temperatures below zero make no sense (got {args.temperature})')

    with open(args.scores_file) as f:
        scores = load_scores_dict(f)

    with open(args.hyp_file) as f:
        hyps = load_hyps_dict(f)

    if scores.keys() != hyps.keys():
        logging.error('Not matching segments!')
        logging.error(f'Only in scores: {scores.keys() - hyps.keys()}')
        logging.error(f'Only in hypotheses: {hyps.keys() - scores.keys()}')

        exit(1)

    nb_nonmatched = 0
    outputs = {}
    for seg_name in scores.keys():
        logging.info(f'Processing {seg_name}')
        score = scores[seg_name]
        hyp = hyps[seg_name]

        symm_diff = [s for s in score if s not in hyp] + [h for h in hyp if h not in score]
        symm_diff_size = len(symm_diff)
        if symm_diff_size > 0:
            nb_nonmatched += symm_diff_size
            logging.warning(f'Segment {seg_name} had {symm_diff_size} unmatched scores ({symm_diff})')

        best_path = get_token_confidences(score, hyp, args.temperature, args.dummy)
        outputs[seg_name] = best_path

    write_method = output_formats[args.output_format]
    if args.confidence_file:
        with open(args.confidence_file, 'w') as out_f:
            for seg_name, best_path in outputs.items():
                write_method(out_f, seg_name, best_path)
    else:
        for seg_name, best_path in outputs.items():
            write_method(sys.stdout, seg_name, best_path)

    if nb_nonmatched > 0:
        logging.warning(f'There was a total of {nb_nonmatched} non matched scores')


if __name__ == '__main__':
    main()

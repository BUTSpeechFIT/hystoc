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

def write_pctm(out_f, seg_name, best_path):
    words_reprs = [f'{pos[0]} {pos[1]}' for pos in best_path]
    out_f.write(f'{seg_name} {" ".join(words_reprs)}\n')

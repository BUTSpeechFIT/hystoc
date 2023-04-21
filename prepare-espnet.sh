#!/bin/bash

ii=1

set -euo pipefail

root_input_dir=/mnt/matylda5/ibenes/projects/hystoc-sandbox/albayzin/martas-split.$ii
out_dir=/mnt/matylda5/ibenes/projects/hystoc-sandbox/out-albayzin-prep
mkdir -p $out_dir

workdir=/mnt/matylda5/ibenes/projects/hystoc-sandbox/albayzin-prep/$ii.work
mkdir -p $workdir

for fn in $(ls $root_input_dir)
do
    no=$(echo $fn | sed "s/best_recog.*//")
    sed "s/ /-$no /" < $root_input_dir/$fn/text > $workdir/$no.text
    sed "s/ /-$no /" < $root_input_dir/$fn/score > $workdir/$no.score
done

text_file=$ii.text-concat
cat $workdir/*.text | sort > $text_file

enhanced_hyp_names=$ii.full_ids
cat $text_file | cut -d" " -f1 > $enhanced_hyp_names

txt_only_output=$ii.text-only
cat $text_file | cut -d" " -f2- > $txt_only_output

paste -d" "  $enhanced_hyp_names $txt_only_output > $out_dir/$ii.nbest

score_only=$ii.score-only
cat $workdir/*.score | sort | sed "s/tensor(//" | sed "s/)//" | cut -d" " -f2 > $score_only
paste -d" " $enhanced_hyp_names $score_only > $out_dir/$ii.score

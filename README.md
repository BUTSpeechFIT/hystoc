# hystoc
Getting confidences from any end-to-end systems, developed in context of Automatic Speech Recognition.
The underlying technique was previously sucessfuly applied to [semi-supervised learning in OCR](https://arxiv.org/abs/2104.13037).
Hystoc is oblivious to the underlying task, but please note that no special care is provided for non-monotonic tasks such as Machine Translation.

When using Hystoc, please cite (currently redacted because the paper is in a double-blind review).

## Installation

Hystoc is available on PyPi, so you can directly install it:

```
pip install hystoc
```

## Usage

To obtain confidences, simply run:
```
hystoc-confidences --temperature 1.0 example/a.txt example/a.score
```

Increasing temperature (to about 3.0) leads to slightly better calibrated confidences.

Hystoc also allows to directly fuse outputs of multiple systems into a single one.

To this end a list of pairs needs to be provided like this:
```
hystoc-fusion --confidence-file fused.txt --method normalize-per-system example/a.txt example/a.score example/b.txt example/b.score
```

Please note that our experiments did not show Hystoc fusion to consistently outperform Rover.

### Usage without installion

If you prefer to just to just download rather than install, you can access the two tools as `hystoc/hystoc_confidences.py` and `hystoc/hystoc_fusion.py`.
Setting `PYTHONPATH` correctly is then you responsiblity.

## Input formats

Both text and score files follow Kaldi-inspired format.

A text file contains hypotheses with the desired level of tokenization given by whitespace:
```
uttA-1 Some example text
uttA-2 Mom example text
uttB-1 Nice bowl of rice
uttB-2 Rice bowl of nice
```

A score file contains (possibly un-normalized) posterior log-probabilities of the hypotheses.
```
uttA-1 -0.264534
uttA-2 -9.381741
uttB-1 -0.185739
uttB-2 -1.294320
```


## Output formats

Both tools accept `--output-method [pctm|ctm]` as an option.

With `ctm`, the output is a CTM file ready for rover fusion or sclite scoring, e.g.:
```
rtve2020_00000000000000000BR-C2!0008099-0008170 1 0.00 0.15 <noise> 0.9183508755328569
rtve2020_00000000000000000BR-C2!0008285-0008422 1 0.00 0.15 dijo 0.5429209752714736
rtve2020_00000000000000000BR-C2!0008285-0008422 1 0.15 0.15 irene 0.9869227855728511
rtve2020_00000000000000000BR-C2!0008450-0008736 1 0.00 0.15 creo 1.0
rtve2020_00000000000000000BR-C2!0008450-0008736 1 0.15 0.15 que 1.0
rtve2020_00000000000000000BR-C2!0008450-0008736 1 0.30 0.15 querrás 0.7093835505039835
rtve2020_00000000000000000BR-C2!0008450-0008736 1 0.45 0.15 un 1.0
rtve2020_00000000000000000BR-C2!0008450-0008736 1 0.60 0.15 poco 1.0
rtve2020_00000000000000000BR-C2!0008450-0008736 1 0.75 0.15 de 1.0
rtve2020_00000000000000000BR-C2!0008450-0008736 1 0.90 0.15 intimidad 1.0
rtve2020_00000000000000000BR-C2!0008450-0008736 1 1.05 0.15 para 1.0
rtve2020_00000000000000000BR-C2!0008450-0008736 1 1.20 0.15 este 0.9906944725165938
rtve2020_00000000000000000BR-C2!0008450-0008736 1 1.35 0.15 visionado 0.9800563178675208
```

The timing information in the CTM is made up.

With `pctm`, the output is a "pseudo-CTM", where the confidence follows after every token, e.g.:
```
rtve2020_00000000000000000BR-C2!0008099-0008170 ay 0.4045044519729132
rtve2020_00000000000000000BR-C2!0008285-0008422 me 0.7169367774080452 dejo 0.7991855335146294 irene 0.9938079240372626
rtve2020_00000000000000000BR-C2!0008450-0008736 creo 1.0 que 1.0 querrás 0.9921967974603854 un 1.0 poco 1.0 de 1.0 intimidad 1.0 para 1.0 este 1.0 visionado 0.9421039825750096
r
```

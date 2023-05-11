# hystoc
Getting confidences from any end-to-end systems, developed in context of Automatic Speech Recognition.
The underlying technique was previously sucessfuly applied to [semi-supervised learning in OCR](https://arxiv.org/abs/2104.13037).
Hystoc is oblivious to the underlying task, but please note that no special care is provided for non-monotonic tasks such as Machine Translation.

When using Hystoc, please cite (currently redacted because the paper is in a double-blind review).

## Installation

So far, installation is limited to manually downloading the package.

## Usage

To obtain confidences, Hystoc needs two inputs describing the competing hypotheses:

A text file with the desired level of tokenization given by whitespace:
```
uttA-1 Some example text
uttA-2 Mom example text
uttB-1 Nice bowl of rice
uttB-2 Rice bowl of nice
```

A score file with (possibly un-normalized) log-probabilities of the hypotheses.
```
uttA-1 -0.264534
uttA-2 -9.381741
uttB-1 -0.185739
uttB-2 -1.294320
```

Then, confidences can obtained with:
```
hystoc-confidences --temperature 1.0 hypotheses scores output
```

### Performing direct fusion with Hystoc
Hystoc also allows to directly fuse outputs of multiple systems into a single one.

To this end a list of pairs needs to be provided like this:
```
hystoc-fusion --confidence-file fused.txt --method normalize-per-system example/a.score example/a.txt example/b.score example/b.txt
```

Please note that our experiments did not show Hystoc fusion to consistently outperform Rover.

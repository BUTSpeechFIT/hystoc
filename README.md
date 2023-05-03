# hystoc
Getting confidences from any end-to-end systems, developed in context of Automatic Speech Recognition.
The underlying technique was previously sucessfuly applied to [semi-supervised learning in OCR](https://arxiv.org/abs/2104.13037).
Hystoc is oblivious to the underlying task, but please note that no special care is provided for non-monotonic tasks such as Machine Translation.

When using Hystoc, please cite us (currently readcted because the paper is in a double-blind review).

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
uttA-1 -0.264534e-1
uttA-2 -0.938174e-7
uttB-1 -0.185739e-2
uttB-2 -0.294320e-3
```

Then, confidences can obtained with:
```
nbest-to-confidence.py --temperature 1.0 hypotheses scores output
```

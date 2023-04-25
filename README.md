# Learning Sinkhorn Divergences for Supervised Change Point Detection


This repository provides the code for the implementing [Learning Sinkhorn Divergences for Change Point Detection](https://arxiv.org/pdf/2202.04000.pdf).



Available change points are used to obtain triplet pairs which are then used to learn a metric for Sinkhorn divergences.
This metric is then used by Sinkhorn divergences in two-sample tests over sliding windows for improved change point detection performance.

Requires

- Pytorch (1.8+)
- Python (3.7+)
- Matplotlib
- Scipy


Main file to run: 

main_run.py

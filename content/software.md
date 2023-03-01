---
title : "Software Projects"
hidemeta : true
disableShare: true
---

I've been programming since I was 11 years old. These days, I mostly write Python
code. In the past, I enjoyed writing C/C++/CUDA (and dabbled in JavaScript).

### Fairer Features

I think fairness in ML is of utmost importance. Fairer Features is a production modeling pipeline
that, given an image dataset, can produce a detailed PDF report conveying the demographic make-up
of the dataset (race, age, sex), as well as how each demographic group is depicted. The modeling pipeline
consists of CNNs and Large-Language Models, employing novel techniques for obtaining demographic depiction information.

Originally, I planned on launching this code into a startup--I interviewed with YCombinator for their 2023W batch. However, I think I'll end up open-sourcing it soon, in addition to releasing a paper.

### DSRPytorch

I really liked the deep symbolic regression [paper](https://arxiv.org/abs/1912.04871). The authors
only released a TensorFlow implemenation--I wanted to use this method for my own research, and I prefer
PyTorch. So, I read the paper and implemented it in PyTorch. There were some cool parts involved, such as
batching of variable-length sequences and a sequence-to-PyTorch transpiler. The code is available [here](https://github.com/dandip/DSRPytorch).


### GPU Optimization Code

I wrote a conjugate gradient solver using CUDA. I employed some fun computational tricks, which enabled my code to outperform a naive GPU implementation by an order of magnitude: (1) [Loop unrolling](https://en.wikipedia.org/wiki/Loop_unrolling); (2) Various [data optimizations](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) involving both concurrency and access patterns; (3) Floating point precision starts out very coarse and gradually becomes more precise as the algorithm iteratively approaches its tolerance

### D2 BioSoftware

Using OpenCV and Electron, I wrote an application for ingesting images of petri dishes and computing
how many bacterial colonies are present. Under the hood, I use a variety of image-processing techniques
like bilateral filtering, adaptive threshold, and various other transformations. Then, I binarize the transformed image and feed it into some blob detection code that I wrote.

This Software is currently used by the McGowan Institue of Regenerative Medicine at the University of Pittsburgh.
## Introduction

This is a repo for audio generation models. It is in some sense incomplete at the moment, though running the training script in the networks folder will create an autoencoder for audio generation. Nonetheless, there are still many to-dos and updates, as the generation quality is not ideal currently.


## Acknowledgements
The work of Phil Wang [(lucidrains)](https://github.com/lucidrains) was important both as a reference and for some functions. While this repo was mostly developed independently, a number of key steps were drawn from his work, most notably in the structure of "causal" convolution layers. Additionally, reference was made to the [encodec](https://github.com/facebookresearch/encodec) repo from Facebook Research, which were referenced for further improving the causal convolution layers.

## The Generative Model

The current state of the model is heavily based on [Soundstream](https://arxiv.org/pdf/2107.03312.pdf), a vector-quantized variational autoencoder (VQ-VAE). This uses a fully convolutional encoder, a residual vector quantization layer, and a fully (transposed) convolutional decoder. If you want to run this model, see the training.py file, which contains everything needed to train it on a dataset.

The advantage of fully convolutional encoders and decoders is that they allow arbitrary input length. Residual vector quantization involves quantizing a signal, subtracting the quantized signal from the original, and iteratively quantizing the residual from that repeating the process. It's advantageous here because it can enable the quantizer to capture a huge permutation of "symbols" without using a similarly huge codebook.

As of this writing, the main differences between this implementation and the original Soundstream paper are:

* Different parameters in general
* A new optional optimizer objective: the channel average of the layer activations for the transpose convolution steps should approximate the signal downsampled to the same size

Some training tips:

* A low learning rate seems neccesary here - the model has devolved into producing noise consistently when LRs are high
* I implemented a number of methods to learn on curriculums, though can only tentatively say they were helpful at this point. These steps include weighting different loss scales depending on the epoch (e.g. early on the model should focus on learning to match the downsampled waveform, spending less time worried about details)

Some additional thoughts:

* If we have a a good amount of metadata (e.g. genre) we could try a VQ-VAE approach where we have a quantization layer from the latent space to the metadata space. This would allow us to generate new audio conditioned on metadata. Likely would want a quantization layer for each type of metadata, with codebook size equal to the number of classes.
* Left-brain/right-brain approach : an oft-repeated simplification is that the human left hemisphere is details-oriented while the right hemisphere is "big picture" oriented. One supporting piece of evidence used for this is that damage to the left inferior frontal gyrus (Broca's area) leads to people losing the ability to produce sentences (the "details"). They can still convey feeling vocally though ("prosody") and can still vocalize emotional state through pitch. On the other hand, damage to the right inferior frontal gyrus preserves speech ability, but leads to a monotone voice and difficulty conveying emotion. I wonder if splitting the network into two components (one focused on details, the other on overall melodic content) could be advantageous.

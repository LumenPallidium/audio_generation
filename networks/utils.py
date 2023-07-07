import torch
import matplotlib.pyplot as plt
import os
import numpy as np
import torchaudio
from einops import rearrange
from warnings import warn
from sympy.ntheory import factorint
from datasets import COMMONVOICE

#TODO: add a function for weight and spectral normalization here

def approximate_square_root(x):
    factor_dict = factorint(x)
    factors = []
    for key, item in factor_dict.items():
        factors += [key] * item
    factors = sorted(factors)

    a, b = 1, 1
    for factor in factors:
        if a <= b:
            a *= factor
        else:
            b *= factor
    return a, b

def np_softmax(lis):
    """A softmax function that works on numpy arrays"""
    return np.exp(lis) / np.sum(np.exp(lis))

def add_util_norm(module, norm = "weight", **norm_kwargs):
    """Adds a norm from torch.nn.utils to a module"""
    if norm == "weight":
        norm_f = torch.nn.utils.weight_norm
    elif norm == "spectral":
        norm_f = torch.nn.utils.spectral_norm
    else:
        norm_f = lambda x: x
    return norm_f(module, **norm_kwargs)

def plot_waveform(waveform, 
                  sample_rate, 
                  save_path = None, 
                  ax = None, 
                  return_ax = False, 
                  alpha = 1,
                  color = "blue"):
    """Copied from torchaudio tutorial and extended for multiple plotting."""
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate
    if ax is None:
        figure, ax = plt.subplots(1, 1)
    else:
        figure = ax.figure

    ax.plot(time_axis, 
            waveform[0], 
            linewidth=1, 
            alpha = alpha,
            color = color)
    ax.grid(True)
    figure.suptitle("waveform")
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    if return_ax:
        return ax
    
def bitrate_calculator(stride_factor = 320, sample_rate = 24000, target_bitrate = 6000):
    """Based on the Soundstream and following papers, a quick function to print out number of
    quantizers and codebook entry count that is consistent with a given bitrate."""
    fps = sample_rate / stride_factor

    bpf = target_bitrate / fps

    example_quantizer_numbers = [i for i in range(4, 17)]
    print(f"To have a bitrate of {target_bitrate} bps, with a stride factor of {stride_factor} and a sample rate of {sample_rate}, the codebook sizes should be as follows:")
    for quantizer_number in example_quantizer_numbers:
        print(f"\tNum quantizers = {quantizer_number} -> {round(2 ** (bpf / quantizer_number))} num codebook entries")

def collator(batch, size = 72000, resampler = None):
    """A function for dealing with the irregular tensor/sequence
      sizes prevalent in audio datasets. Pads if they are too short,
      otherwise crops to the desired size"""
    new_batch = []
    for x in batch:
        # remove the label, may need to be changed for different datasets
        x = x[0]
        if resampler is not None:
            x = resampler(x)
        x_len = x.shape[-1]
        if x_len < size:
            # pad front and back with random length of zeros
            diff = size - x_len
            split = torch.randint(0, diff, (1,)).item()
            prefix_zeros = torch.zeros((x.shape[0], split))
            suffix_zeros = torch.zeros((x.shape[0], diff - split))

            new_batch.append(torch.cat([prefix_zeros, x, suffix_zeros], dim = -1))
        elif x_len > size: # believe it or not i got an error here cause the sizes were identical
            # crop to a random part of the sample
            diff = x_len - size
            start = torch.randint(0, diff, (1,)).item()
            end = start + size
            new_batch.append(x[: , start:end])
    return new_batch

def print_stale_clusters(in_clusters, out_clusters):
    """Pretty prints the number of unused codebook entries in each quantizer of RVQ,
    relies on a method in the quantizer class that returns the number of unused
    codebook entries. Could maybe be absorbed into the class."""
    for i, (in_cluster, out_cluster) in enumerate(zip(in_clusters, out_clusters)):
        print(f"\tQuantizer {i} stale cluster change : {in_cluster} -> {out_cluster}")

def dist_to_uniform(step, rate = 0.002, initial_dist = [1, 1, 1, 1]):
    """Returns a distribution that is more uniform the more steps have passed."""
    mean = sum(initial_dist) / len(initial_dist)
    dist = [i - (i - mean) * step * rate for i in initial_dist]
    return dist

def interpolate_lists(list1, list2):
    """Interpolates between two lists of the same length."""
    return lambda t : [t * i + (1 - t) * j for i, j in zip(list1, list2)]

def losses_to_running_loss(losses, alpha = 0.95):
    running_losses = []
    running_loss = losses[0]
    for loss in losses:
        running_loss = (1 - alpha) * loss + alpha * running_loss
        running_losses.append(running_loss)
    return running_losses

def get_latest_file(path, name):
    """Util to get the most recent model checkpoints easily."""
    try:
        files = [os.path.join(path, f) for f in os.listdir(path) if name in f]
        file = max(files, key = os.path.getmtime)
        # replacing backslashes with forward slashes for windows
        file = file.replace("\\","/")
    except (ValueError, FileNotFoundError):
        file = None
    return file

def tuple_checker(item, length):
    """Checks if an item is a tuple or list, if not, converts it to a list of length length.
    Also checks that an input tuple is the correct length.
    Useful for giving a function a single item when it requires a iterable."""
    if isinstance(item, (int, float, str)):
        item = [item] * length
    elif isinstance(item, (tuple, list)):
        assert len(item) == length, f"Expected tuple of length {length}, got {len(item)}"
    return item

def get_dataset(name, path):
    if name == "librispeech":
        # "C:/Projects/librispeech/"
        dataset = torchaudio.datasets.LIBRISPEECH(path, url="train-clean-100", download=True)
        sample_rate = 16000
    elif name == "commonvoice":
        # "C:/Projects/common_voice/"
        dataset = COMMONVOICE(path)

        sample_rate = 48000

    else:
        raise ValueError(f"Dataset {name} not recognised")
    
    return dataset, sample_rate

def sound_to_codebooks(sound, model):
    if not model.quantizer.use_som:
        warn("This is not a SOM model, that is okay but the codebooks have no intrinsic topology.")
        h, w = approximate_square_root(model.codebook_size[0])
    else:
        # i was too verbose with my class stuff
        h = model.quantizer.quantizers[0].som.height
        w = model.quantizer.quantizers[0].som.width
    _, _, indices = model.encode(sound)
    # indices has shape batch, length, num_quantizers
    if len(indices.shape) == 3:
        # get just the first batch
        indices = indices[0]
    indices = torch.nn.functional.one_hot(indices, num_classes = model.codebook_size[0])
    # break into hw but join codebooks on width
    indices = rearrange(indices, "l nq (h w) -> l h w nq", h = h, w = w)
    indices = indices.sum(dim = -1)
    return indices

def animate_sound(sound, model, rate = 16000, slowdown = 2):
    from matplotlib import animation
    codebooks = sound_to_codebooks(sound, model).cpu().numpy()

    time_len = sound.shape[-1] * slowdown / rate
    time_per_frame = time_len / codebooks.shape[0]

    fig, ax = plt.subplots()
    cax = ax.pcolormesh(codebooks[0], cmap="Blues")

    def animate(i):
        cax.set_array(codebooks[i])

    anim = animation.FuncAnimation(fig, animate, interval = 1000 * time_per_frame, frames = codebooks.shape[0])
    anim.save("test.mp4")

    sound_recons, _, _ = model(sound)

    torchaudio.save("test.wav", sound_recons.squeeze(0), rate // slowdown)

    # reload video and add audio to it - not sure of a better way using matplotlib
    import ffmpeg
    video = ffmpeg.input("test.mp4")
    audio = ffmpeg.input("test.wav")

    ffmpeg.output(video, audio, "output.mp4").run()

    # delete the intermediate files
    os.remove("test.mp4")
    os.remove("test.wav")




    
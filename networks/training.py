import torch
import torchaudio
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import Audio

import utils
from discriminator import discriminator_generator_loss, WaveFormDiscriminator, STFTDiscriminator
from vae import CausalVQAE



class WarmUpScheduler(object):
    """Copilot wrote this, made some small tweaks though."""
    def __init__(self, optimizer, scheduler, warmup_iter, total_iter = 300000):
        self.optimizer = optimizer
        self.scheduler = scheduler(optimizer, total_iter - warmup_iter)
        self.warmup_iter = warmup_iter
        self.iter = 0
    
    def step(self):
        if self.iter < self.warmup_iter:
            lr = self.iter / self.warmup_iter * self.scheduler.get_last_lr()[0]
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            self.scheduler.step()
        self.iter += 1

def multispectral_reconstruction_loss(original, 
                                   reconstruction,
                                   device,
                                   spectrograms,
                                   windows = [2 ** i for i in range(6, 12)],
                                   eps = 1e-8,
                                   spec_loss_weight = 1):
    """Energy based spectral loss from here:
    https://arxiv.org/pdf/2008.01160.pdf"""
    l1_f = torch.nn.functional.l1_loss
    l2_f = torch.nn.functional.mse_loss

    alphas = [np.sqrt(window / 2) for window in windows]
    spec_loss = 0
    for i, spectrogram in enumerate(spectrograms):
        original_spec = torch.nan_to_num(spectrogram(original))
        reconstruction_spec = torch.nan_to_num(spectrogram(reconstruction))
        spec_loss +=  l1_f(original_spec, reconstruction_spec)
        spec_loss += alphas[i] * l2_f((original_spec + eps).log(),
                                      (reconstruction_spec + eps).log())
    return spec_loss_weight * spec_loss

def multiscale_reconstruction_loss(original,
                                   reconstructions,
                                   device,
                                   scale_weights = [0.25, 0.5, 1, 1.25, 0.01, 2.99],
                                   sparsity_weight = 0.01):
    l2_f = torch.nn.functional.mse_loss
    downsample = torch.nn.functional.interpolate

    waveform_loss = 0

    for scale, reconstruction in zip(scale_weights, reconstructions):
        size = reconstruction.shape[-1]
        original_scaled = downsample(original, size = size)
        waveform_loss += scale * l2_f(original_scaled, reconstruction)

    return waveform_loss

def save_samples(real, fake, epoch, i, path, sample_rate = 16000):
    name = path + f"sample_{epoch}_{i}.png"

    real = real[0].detach().cpu()
    fake = fake[0].detach().cpu()

    ax = utils.plot_waveform(real, sample_rate, None, return_ax = True, alpha = 0.3)
    utils.plot_waveform(fake, sample_rate, name, ax = ax, color = "red", alpha = 0.3)

class Trainer():
    def __init__(self,
                 device,
                 save_path,
                 model,
                 dataset,
                 resampler = None,
                 config = "full",
                 model_path = None,
                 model_lr = 5e-5,
                 discriminator_lr = 2e-6,
                 scheduler = None,
                 sample_rate = 24000,
                 discriminators = None,
                 discriminator_paths = None,
                 use_one_discriminator = False,
                 codebook_update_step = 2,
                 mini_epoch_length = 100,
                 batch_size = 8,
                 spec_windows = [2 **i for i in range(6, 12)],
                 spec_bins = 64,
                 save_every = 1,
                 # these are based on experiments
                 spec_loss_weight = 0.04,
                 reconstruction_loss_weight = 10,
                 generator_loss_weight = 1,
                 loss_alpha = 0.95,
                 ):
        
        self.device = device
        self.save_path, self.image_save_path = self._init_paths(save_path)

        self.model = model.to(self.device)
        self.resampler = resampler

        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path))
            print(f"\tLoaded model from {model_path}")

        self.scheduler = scheduler
        self.dataset = dataset
        self.optimizers = self._init_optimizers(model_lr)

        self.mini_epoch_length = mini_epoch_length
        self.save_every = save_every
        self.batch_size = batch_size
        self.codebook_update_step = codebook_update_step
        self.sample_rate = sample_rate
        self.use_one_discriminator = use_one_discriminator

        self.spec_windows = spec_windows
        self.spectrograms = [torchaudio.transforms.MelSpectrogram(sample_rate = self.sample_rate, 
                                                        n_fft = max(window, 512),
                                                        win_length = window,
                                                        hop_length = window // 4,
                                                        n_mels = spec_bins,
                                                        normalized = True).to(self.device) for window in spec_windows]
        self.spec_loss_weight = spec_loss_weight
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.generator_loss_weight = generator_loss_weight

        self.loss_alpha = loss_alpha
        self.loss_breakdown = {}
        
        # load discriminators
        self.discriminators = self._init_discriminators(discriminators, discriminator_paths, discriminator_lr)

        self.epoch = 0
        self.mini_epoch_i = 0

    def _init_discriminators(self, discriminators, discriminator_paths, discriminator_lr):
        if discriminators is not None:
            for discriminator in discriminators:
                discriminator.to(self.device)
                self.optimizers.append(torch.optim.Adam(discriminator.parameters(), lr = discriminator_lr))

            if discriminator_paths is not None:
                for discriminator, path in zip(discriminators, discriminator_paths):
                    if path is not None:
                        discriminator.load_state_dict(torch.load(path))
                        print(f"\tLoaded discriminator from {path}")

        return discriminators
    

    def _init_optimizers(self, model_lr):
        # if scheduler is supplied, get its optimizer
        if scheduler is None:
            optimizers = [torch.optim.Adam(self.model.parameters(), lr = model_lr)]
        else:
            optimizers = [self.scheduler.optimizer]
        return optimizers
    
    @staticmethod
    def _init_paths(path):
        image_path = path + "waveform_plots/"
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        return path, image_path
    
    def update_loss_breakdown(self, loss, loss_name):
        if loss_name not in self.loss_breakdown:
            self.loss_breakdown[loss_name] = loss.item()
        else:
            self.loss_breakdown[loss_name] = loss.item() * (1 - self.loss_alpha) + self.loss_breakdown[loss_name] * self.loss_alpha

    def print_loss_breakdown(self):
        print("\tLoss breakdown:")
        loss_sum = sum(self.loss_breakdown.values())
        for key, value in self.loss_breakdown.items():
            print(f"\t\t{key}: {round(value, 2)} ({round(100 * value / loss_sum, 2)}%)")

    def mini_epoch(self,
                    data_loader_iter,
                    losses = None,
                    accumulation_steps = 2,
                    prioritize_early = False,
                    gan_loss = True,
                    multispectral = True,
                    multiscale = True,
                    use_reconstruction_loss = True,
                    save_plots = True,
                    sparsity_weight = 0.01,):
        """Executes a mini-epoch. Can be as part of a GAN etc."""
        optimizer = self.optimizers[0]
        if gan_loss:
            if self.use_one_discriminator:
                # only one at a time
                discriminator_number = np.random.randint(0, len(self.discriminators))
                discriminator = [self.discriminators[discriminator_number]]
                optimizer_d = [self.optimizers[discriminator_number + 1]]
            else:
                discriminator = self.discriminators
                optimizer_d = self.optimizers[1:]
        for i in range(self.mini_epoch_length // accumulation_steps):
            optimizer.zero_grad()
            
            if gan_loss:
                for optimizer_d_i in optimizer_d:
                    optimizer_d_i.zero_grad()

            for j in range(accumulation_steps):

                if (i * accumulation_steps + j) % self.codebook_update_step == 0:
                    update_codebook = True
                else:
                    update_codebook = False

                x = next(data_loader_iter)
                x = torch.vstack(x).unsqueeze(1).to(self.device)

                y, commit_loss, _, multiscales = self.model(x, multiscale = multiscale, update_codebook = update_codebook, prioritize_early = prioritize_early)

                self.update_loss_breakdown(commit_loss, "commit_loss")

                if use_reconstruction_loss:
                    if multiscale:
                        loss = multiscale_reconstruction_loss(x, multiscales, self.device)
                    else:
                        loss = torch.nn.functional.l1_loss(x, y)
                        
                    loss *= self.reconstruction_loss_weight

                    self.update_loss_breakdown(loss, "reconstruction_loss")
                else:
                    loss = 0

                # waveforms are typically somewhat sparse (silence, etc)
                if sparsity_weight > 0:
                    sparsity_loss = sparsity_weight * (y.abs()).mean()
                    self.update_loss_breakdown(sparsity_loss, "sparsity_loss")
                    loss += sparsity_loss


                loss += commit_loss
                if multispectral:
                    multispectral_loss = multispectral_reconstruction_loss(x, y, 
                                                              self.device, 
                                                              self.spectrograms, 
                                                              spec_loss_weight = self.spec_loss_weight, 
                                                              windows = self.spec_windows)
                    
                    self.update_loss_breakdown(multispectral_loss, "multispectral_loss")
                    loss += multispectral_loss

                if gan_loss:
                    discriminator_loss = 0
                    for discriminator_i in discriminator:
                        generator_loss, discriminator_loss_i = discriminator_generator_loss(x, y, discriminator_i)
                        self.update_loss_breakdown(generator_loss, "generator_loss")
                        loss += generator_loss * self.generator_loss_weight

                        discriminator_loss += discriminator_loss_i

                    discriminator_loss /= (accumulation_steps / self.generator_loss_weight)
                    discriminator_loss.backward(retain_graph = True)

                loss /= accumulation_steps
                loss.backward()


            if losses is not None:
                losses.append(loss.item())

            optimizer.step()
            if gan_loss:
                for optimizer_d_i in optimizer_d:
                    optimizer_d_i.step()
            if scheduler is not None:
                scheduler.step()
        
        if save_plots:
            save_samples(x, y, 
                         self.epoch, 
                         self.mini_epoch_i, 
                         self.image_save_path, 
                         sample_rate = self.sample_rate)

        self.mini_epoch_i += 1

        return y


    def train(self, epochs = 5, 
              losses = None, 
              gan_loss = True, 
              multispectral = True, 
              multiscale = True, 
              use_reconstruction_loss = True,
              sparsity_weight = 0.01):

        n_steps = len(self.dataset)
        
        n_mini_epochs = n_steps // (self.mini_epoch_length * self.batch_size)

        for epoch in range(epochs):
            epoch_losses = []
            epoch_start_stale_clusters = self.model.quantizer.get_stale_clusters()
            # reset the data loader each epoch
            train_loader = torch.utils.data.DataLoader(self.dataset,
                                                       batch_size=self.batch_size,
                                                       shuffle=True,
                                                       collate_fn=lambda x : utils.collator(x, resampler=self.resampler))
            
            train_loader_iter = iter(train_loader)

            for mini_epoch_i in tqdm(range(n_mini_epochs)):
                y = self.mini_epoch(train_loader_iter, 
                                    losses = epoch_losses,
                                    gan_loss = gan_loss,
                                    use_reconstruction_loss = use_reconstruction_loss,
                                    multispectral = multispectral,
                                    multiscale = multiscale,
                                    sparsity_weight = sparsity_weight)

            epoch_end_stale_clusters = model.quantizer.get_stale_clusters()

            torchaudio.save(self.save_path + f"epoch_{epoch}_sample.wav", y[0].detach().cpu(), self.sample_rate)

            print(f"Epoch {self.epoch} mean loss: ", np.mean(epoch_losses))
            self.print_loss_breakdown()
            utils.print_stale_clusters(epoch_start_stale_clusters, epoch_end_stale_clusters)

            if epoch % self.save_every == 0:
                torch.save(self.model.state_dict(), self.save_path + f"model_epoch_{self.epoch}.pt")
                if gan_loss:
                    torch.save(self.discriminators[0].state_dict(), self.save_path + f"wv_discriminator_epoch_{self.epoch}.pt")
                    torch.save(self.discriminators[1].state_dict(), self.save_path + f"stft_discriminator_epoch_{self.epoch}.pt")

            if losses is not None:
                losses = losses + epoch_losses

            self.epoch += 1

        if losses:
            plt.plot(utils.losses_to_running_loss(losses))
            plt.show()

    def om_overtrain(self, batches = 16,  n_steps = 10000):
        """Overtrains on the OM sound for good luck."""
        om = torchaudio.load(r"om.wav")[0]
        om = om.mean(dim = 0, keepdim = True).to(self.device)
        om = om.repeat(batches, 1, 1)

        om_optimizer = torch.optim.Adam(model.parameters(), lr = 3e-4)

        losses_om = []
        for step in tqdm(range(n_steps // batches)):
            y, commit_loss, index, multiscales = self.model(om, update_codebook = True, multiscale = True)

            loss = torch.mean((y - om).pow(2)) + commit_loss
            losses_om.append(loss.item())

            om_optimizer.zero_grad()
            loss.backward()
            om_optimizer.step()

            if step % 100 == 0:
                utils.plot_waveform(y[0].detach().cpu(), 16000, save_path = "C:/Projects/test_om.png")

        plt.plot(utils.losses_to_running_loss(losses_om))

    def overtrain(self, batch_size = 16, n_steps = 5000):
        """Overtrains on a single sample from the data."""
        def param_sum(model):
            return sum(p.sum() for p in model.parameters() if p.requires_grad)

        start_param_sum = param_sum(self.model)

        train_loader = torch.utils.data.DataLoader(self.dataset,
                                                   batch_size=1,
                                                   collate_fn=lambda x : utils.collator(x, resampler=self.resampler))
        optimizer = torch.optim.Adam(model.parameters(), lr = 5e-6)
        x = next(iter(train_loader))[0]

        x = x.repeat(batch_size, 1, 1).to(self.device)

        for step in tqdm(range(n_steps // batch_size)):
            
            y, commit_loss, index, multiscales = self.model(x, update_codebook = True, multiscale = True)

            loss = torch.mean((y - x).pow(2)) + commit_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                torchaudio.save("C:/Projects/test.wav", y[0].detach().cpu(), self.sample_rate)

        end_param_sum = param_sum(self.model)

        print("Param sum before overtraining: ", start_param_sum)
        print("Param sum after overtraining: ", end_param_sum)
        print("Param sum difference: ", end_param_sum - start_param_sum)

        return y[0].detach().cpu()

#TODO : look into quantizer with extensible codebook
#TODO : look into log-loss and fourier loss (L1 norm useful apparently)
#TODO : test adding regressor variables (eg speaker gender)
#TODO : try muliscale STFT discriminator like in encodec
if __name__ == "__main__":

    # update these if running on your end
    experiment_name = input("Please enter an experiment name:")
    experiment_name = "default_experiment" if experiment_name == "" else experiment_name
    save_path = "C:/Projects/singing_models/" + experiment_name + "/"
    dataset_path = "C:/Projects/librispeech/"

    use_discriminator = True
    scratch_train = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CausalVQAE(1, num_quantizers = 15, codebook_size = 256, input_format = "n c l")

    #resampler = torchaudio.transforms.Resample(16000, 24000)
    if not scratch_train:
        latest_model_path = utils.get_latest_file(save_path, "model")
    else:
        latest_model_path = None

    if use_discriminator:
        discriminators = [WaveFormDiscriminator(1), 
                        STFTDiscriminator()]

        if not scratch_train:
            latest_discriminator_path = utils.get_latest_file(save_path, "wv_discriminator")
            latest_stft_discriminator_path = utils.get_latest_file(save_path, "stft_discriminator")
        else:
            latest_discriminator_path = None
            latest_stft_discriminator_path = None
    else:
        discriminators = None
        latest_discriminator_path = None
        latest_stft_discriminator_path = None

    librispeech = torchaudio.datasets.LIBRISPEECH(dataset_path, url="train-clean-100", download=True)
    scheduler = WarmUpScheduler(torch.optim.Adam(model.parameters(), lr = 8e-5, amsgrad = True), 
                                torch.optim.lr_scheduler.CosineAnnealingLR, 
                                warmup_iter = 10)
    losses = []

    trainer = Trainer(device, 
                      save_path,
                      model, 
                      librispeech,
                      #resampler = resampler,
                      scheduler = scheduler,
                      model_path = latest_model_path,
                      discriminators = discriminators,
                      discriminator_paths = [latest_discriminator_path, latest_stft_discriminator_path],
                      sample_rate = 16000,
                      )
    
    #trainer.om_overtrain()
    trainer.train(epochs = 15, losses = losses, gan_loss = use_discriminator, use_reconstruction_loss = False, sparsity_weight = 0)
    #y = trainer.overtrain()
    #Audio(y.numpy(), rate = 16000)


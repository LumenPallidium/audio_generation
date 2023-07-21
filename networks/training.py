import torch
import torchaudio
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import Audio
import yaml
import utils
from discriminator import discriminator_generator_loss, WaveFormDiscriminator, STFTDiscriminator
from vae import CausalVQAE, ResidualQuantizer
try:
    from energy_transformer import EnergyTransformer
    ET_AVAILABLE = True
except ImportError:
    ET_AVAILABLE = False
    print("EnergyTransformer not available. See the readme if you want to install it.")

class WarmUpScheduler(object):
    """Copilot wrote this, made some small tweaks though."""
    def __init__(self, optimizer, scheduler, warmup_iter, total_iter = 300000, min_lr = None):
        if min_lr is None:
            min_lr = optimizer.param_groups[0]['lr'] / 100
        self.optimizer = optimizer
        self.scheduler = scheduler(optimizer, 
                                   total_iter - warmup_iter, 
                                   eta_min = min_lr)
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

    def save_state_dict(self):
        return {"scheduler": self.scheduler.state_dict(),
                "iter": self.iter,
                "warmup_iter": self.warmup_iter,
                }
    def load_state_dict(self, dictionary):
        self.scheduler.load_state_dict(dictionary["scheduler"])
        self.iter = dictionary["iter"]
        self.warmup_iter = dictionary["warmup_iter"]

def multispectral_reconstruction_loss(original, 
                                   reconstruction,
                                   spectrograms,
                                   windows = [2 ** i for i in range(5, 12)],
                                   eps = 1e-8,
                                   spec_loss_weight = 1,
                                   use_log_l2 = True,
                                   scale_alpha = True):
    """Energy based spectral loss from here:
    https://arxiv.org/pdf/2008.01160.pdf"""
    l1_f = torch.nn.functional.l1_loss
    l2_f = torch.nn.functional.mse_loss

    if scale_alpha:
        alphas = [np.sqrt(window / 2) for window in windows]
    else:
        alphas = [1 for window in windows]

    spec_loss = 0
    for i, spectrogram in enumerate(spectrograms):
        original_spec = torch.nan_to_num(spectrogram(original))
        reconstruction_spec = torch.nan_to_num(spectrogram(reconstruction))
        spec_loss +=  l1_f(original_spec, reconstruction_spec)
        if use_log_l2:
            spec_loss += alphas[i] * l2_f((original_spec + eps).log(), (reconstruction_spec + eps).log())
        else:
            spec_loss += alphas[i] * l2_f(original_spec, reconstruction_spec)
    return spec_loss_weight * spec_loss


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
                 model_lr = 5e-4,
                 discriminator_lr = 8e-4,
                 scheduler = None,
                 sample_rate = 24000,
                 discriminators = None,
                 discriminator_paths = None,
                 use_one_discriminator = False,
                 codebook_update_step = 4,
                 mini_epoch_length = 100,
                 steps_per_epoch = None,
                 batch_size = 8,
                 spec_windows = [2 ** i for i in range(5, 12)],
                 save_every = 5,
                 # these are based on experiments
                 spec_loss_weight = 0.01,
                 reconstruction_loss_weight = 10,
                 generator_loss_weight = 1,
                 loss_alpha = 0.95,
                 noise_aug_scale = 0.01,
                 cutoff_scale_per_epoch = 0.95,
                 accumulation_steps = 8,
                 frequency_filter = 6000, # human voice tends to max at 5k Hz
                 codebook_frequency_scale = 0.05, # force deeper entries to hear higher frequencies
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
        self.model_lr = model_lr
        self.optimizers = self._init_optimizers(model_lr)

        self.mini_epoch_length = mini_epoch_length
        self.steps_per_epoch = steps_per_epoch

        self.save_every = save_every
        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps
        self.codebook_update_step = codebook_update_step
        self.sample_rate = sample_rate
        self.use_one_discriminator = use_one_discriminator

        self.spec_windows = spec_windows
        self.spectrograms = [torchaudio.transforms.MelSpectrogram(sample_rate = self.sample_rate, 
                                                        n_fft = max(window, 512),
                                                        win_length = window,
                                                        hop_length = window // 4,
                                                        n_mels = 64,
                                                        normalized = True).to(self.device) for window in spec_windows]
        self.spec_loss_weight = spec_loss_weight
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.generator_loss_weight = generator_loss_weight

        self.loss_alpha = loss_alpha
        self.loss_breakdown = {"generator" : {},
                               "discriminator" : {}}
        
        self.noise_aug_scale = noise_aug_scale
        self.frequency_filter = frequency_filter
        self.codebook_frequency_scale = codebook_frequency_scale
        self.cutoff_scale_per_epoch = cutoff_scale_per_epoch
        
        # load discriminators
        self.discriminators, self.codebook_options = self._init_discriminators(discriminators, discriminator_paths, discriminator_lr)
        self.epoch = 0
        self.mini_epoch_i = 0

        if os.path.exists(self.save_path + "trainer_state.pkl"):
            self.load_state()

            

    def _init_discriminators(self, discriminators, discriminator_paths, discriminator_lr):
        # these help tie codebook dropout/bitrate to the discriminators
        nq = self.model.quantizer.num_quantizers
        
        if discriminators is not None:
            for discriminator in discriminators:
                discriminator.to(self.device)
                self.optimizers.append(torch.optim.Adam(discriminator.parameters(), lr = discriminator_lr))

            if discriminator_paths is not None:
                for discriminator, path in zip(discriminators, discriminator_paths):
                    if path is not None:
                        discriminator.load_state_dict(torch.load(path))
                        print(f"\tLoaded discriminator from {path}")


            nq_per_d = nq // (len(discriminators) - 1)
            # use all for waveform d, then a fraction for each spec d, all for the final spec d
            codebook_options = [nq] + [nq_per_d * (i + 1) for i in range(len(discriminators) - 2)] + [nq]

        else:
            discriminators = None
            codebook_options = [nq]

        return discriminators, codebook_options
    

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
    
    def save_state(self):
        state = {"epoch" : self.epoch,
                 "mini_epoch_i" : self.mini_epoch_i,
                 "loss_breakdown" : self.loss_breakdown,
                 "model_state_dict" : self.model.state_dict(),
                 "optimizers" : [optimizer.state_dict() for optimizer in self.optimizers],
                 "scheduler" : self.scheduler.save_state_dict() if self.scheduler is not None else None}
        torch.save(state, self.save_path + "trainer_state.pkl")
        print(f"\tSaved state to {self.save_path + 'trainer_state.pkl'}")

    def load_state(self):
        state = torch.load(self.save_path + "trainer_state.pkl")
        self.epoch = state["epoch"]
        self.mini_epoch_i = state["mini_epoch_i"]
        self.loss_breakdown = state["loss_breakdown"]
        self.model.load_state_dict(state["model_state_dict"])
        for optimizer, state_dict in zip(self.optimizers, state["optimizers"]):
            optimizer.load_state_dict(state_dict)
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        print(f"\tLoaded trainer state from {self.save_path + 'trainer_state.pkl'}")
    
    def update_loss_breakdown(self, loss, loss_name, type = "generator"):
        if loss_name not in self.loss_breakdown[type]:
            self.loss_breakdown[type][loss_name] = loss.item()
        else:
            self.loss_breakdown[type][loss_name] = loss.item() * self.loss_alpha + self.loss_breakdown[type][loss_name] * (1 - self.loss_alpha)

    def print_loss_breakdown(self):
        print("\tLoss breakdown:")
        for type in ["generator", "discriminator"]:
            print(f"\t\t{type}:")
            loss_sum = sum(self.loss_breakdown[type].values())
            for key, value in self.loss_breakdown[type].items():
                print(f"\t\t\t{key}: {round(value, 2)} ({round(100 * value / loss_sum, 2)}%)")

    def mini_epoch(self,
                    data_loader_iter,
                    losses = None,
                    prioritize_early = False,
                    gan_loss = True,
                    multispectral = True,
                    use_reconstruction_loss = True,
                    save_plots = True,
                    sparsity_weight = 0.01,
                    use_commit_loss = True,
                    discriminator_energies = None,):
        """Executes a mini-epoch. Can be as part of a GAN etc."""
        accumulation_steps = self.accumulation_steps
        optimizer = self.optimizers[0]
        if gan_loss:
            if self.use_one_discriminator:
                # weight the discriminators to favor those that are doing poorly or well (but not middle)
                if discriminator_energies is None:
                    discriminator_energies = [1] * len(self.discriminators)
                probs = utils.np_softmax(discriminator_energies)

                # only one at a time
                discriminator_number = np.random.choice(len(self.discriminators), p = probs)
                discriminator = [self.discriminators[discriminator_number]]
                optimizer_d = [self.optimizers[discriminator_number + 1]]


                # chosen discriminator determines bitrate
                codebook_n = self.codebook_options[discriminator_number]
            else:
                codebook_n = self.model.num_quantizers
                discriminator = self.discriminators
                optimizer_d = self.optimizers[1:]
        else:
            codebook_n = np.random.randint(1, self.model.num_quantizers + 1)
        
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

                if self.frequency_filter is not None:
                    # this makes so deeper codebook entries tend to hear more high freqs
                    cutoff_freq = self.frequency_filter * (1 + codebook_n * self.codebook_frequency_scale)
                    x = torchaudio.functional.lowpass_biquad(x, 
                                                             sample_rate = self.sample_rate, 
                                                             cutoff_freq = cutoff_freq)

                if self.noise_aug_scale:
                    x_ = x + torch.randn_like(x) * self.noise_aug_scale
                else:
                    x_ = x

                y, commit_loss, _ = self.model(x_, 
                                               update_codebook = update_codebook, 
                                               prioritize_early = prioritize_early,
                                               codebook_n = codebook_n)

                if use_reconstruction_loss:

                    loss = torch.nn.functional.l1_loss(x, y)
                        
                    loss *= self.reconstruction_loss_weight

                    self.update_loss_breakdown(loss, "reconstruction_loss")
                else:
                    loss = 0

                if (not self.model.use_energy_transformer) and (use_commit_loss):
                    self.update_loss_breakdown(commit_loss, "commit_loss")
                    loss += commit_loss

                # waveforms are typically somewhat sparse (silence, etc)
                if sparsity_weight > 0:
                    sparsity_loss = sparsity_weight * (y.abs()).mean()
                    self.update_loss_breakdown(sparsity_loss, "sparsity_loss")
                    loss += sparsity_loss
                
                if multispectral:
                    multispectral_loss = multispectral_reconstruction_loss(x, y,
                                                              self.spectrograms, 
                                                              spec_loss_weight = self.spec_loss_weight, 
                                                              windows = self.spec_windows)
                    
                    self.update_loss_breakdown(multispectral_loss, "multispectral_loss")
                    loss += multispectral_loss

                if gan_loss:
                    discriminator_loss = 0
                    for discriminator_i in discriminator:
                        generator_loss, discriminator_loss_i = discriminator_generator_loss(x, y, discriminator_i)
                        self.update_loss_breakdown(generator_loss, f"{discriminator_i.name}_g_loss")
                        loss += generator_loss * self.generator_loss_weight

                        discriminator_loss += discriminator_loss_i

                    discriminator_loss *=  self.generator_loss_weight
                    self.update_loss_breakdown(discriminator_loss, f"{discriminator_i.name}_loss", type = "discriminator")
                    discriminator_loss.backward(retain_graph = True)

                if torch.isnan(loss):
                    print(losses)
                    raise ValueError(f"NaN loss during iteration {i} of mini-epoch {self.mini_epoch_i}")
                else:
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
        
        if gan_loss:
            discriminator_energies = []
            mean_energy = sum(self.loss_breakdown["discriminator"].values()) / len(self.loss_breakdown["discriminator"].values())
            for discriminator_i in self.discriminators:
                try:
                    energy = self.loss_breakdown["discriminator"][f"{discriminator_i.name}_g_loss"]
                    discriminator_energies.append(energy)
                except KeyError:
                    discriminator_energies.append(mean_energy)
        else:
            discriminator_energies = None

        self.mini_epoch_i += 1

        return y, discriminator_energies


    def train(self, 
              epochs = 5, 
              losses = None, 
              gan_loss = True, 
              multispectral = True,
              use_reconstruction_loss = True,
              sparsity_weight = 0.01,
              use_commit_loss = True,
              d_energies = None):

        n_steps = len(self.dataset)
        if self.steps_per_epoch is not None:
            n_steps = min(n_steps, self.steps_per_epoch)

        n_mini_epochs = n_steps // (self.mini_epoch_length * self.batch_size)


        for epoch in range(epochs):
            epoch_losses = []
            if not self.model.use_energy_transformer:
                epoch_start_stale_clusters = self.model.quantizer.get_stale_clusters()
            # reset the data loader each epoch
            train_loader = torch.utils.data.DataLoader(self.dataset,
                                                       batch_size=self.batch_size,
                                                       shuffle=True,
                                                       collate_fn=lambda x : utils.collator(x, resampler=self.resampler))
            
            train_loader_iter = iter(train_loader)

            for mini_epoch_i in tqdm(range(n_mini_epochs)):
                y, d_energies = self.mini_epoch(train_loader_iter, 
                                    losses = epoch_losses,
                                    gan_loss = gan_loss,
                                    use_reconstruction_loss = use_reconstruction_loss,
                                    multispectral = multispectral,
                                    sparsity_weight = sparsity_weight,
                                    use_commit_loss = use_commit_loss,
                                    discriminator_energies = d_energies,)
                
            self.model.update_cutoff(ratio = self.cutoff_scale_per_epoch)

            torchaudio.save(self.save_path + f"epoch_{epoch}_sample.wav", y[0].detach().cpu(), self.sample_rate)

            print(f"Epoch {self.epoch} mean loss: ", np.mean(epoch_losses))
            self.print_loss_breakdown()
            if not self.model.use_energy_transformer:
                epoch_end_stale_clusters = model.quantizer.get_stale_clusters()
                utils.print_stale_clusters(epoch_start_stale_clusters, epoch_end_stale_clusters)

            if epoch % self.save_every == 0:
                
                torch.save(self.model.state_dict(), self.save_path + f"model_epoch_{self.epoch}.pt")
                if gan_loss:
                    for discriminator_i in self.discriminators:
                        torch.save(discriminator_i.state_dict(), self.save_path + f"{discriminator_i.name}_{self.epoch}.pt")
                        
                trainer.save_state()
            if losses is not None:
                losses = losses + epoch_losses

            self.epoch += 1

        torch.save(self.model.state_dict(), self.save_path + f"model_epoch_{self.epoch}_final.pt")
        if gan_loss:
            for discriminator_i in self.discriminators:
                torch.save(discriminator_i.state_dict(), self.save_path + f"{discriminator_i.name}_{self.epoch}_final.pt")

        if losses:
            plt.plot(utils.losses_to_running_loss(losses))
            plt.show()

        return losses
    
    def sample_data(self):
        i = np.random.randint(0, len(self.dataset))
        x = self.dataset[i]
        x = utils.collator([x], size = 72000 * 5, resampler=self.resampler)[0]
        x = x.to(self.device).unsqueeze(0)

        model.eval()
        with torch.no_grad():
          y, _, _ = model(x, codebook_n = model.quantizer.num_quantizers)

        # switch back to train mode
        model.train()
        return y[0].detach().cpu().numpy()

    def train_new_quantizer(self, 
                            new_quantizer, 
                            slow_lr = 1e-6,
                            new_experiment_path = None,
                            **train_kwargs):
        """Trains a new bottleneck with the existing model. The encoder and decoder
        learn at a slower rate (which can be 0) determined by slow_lr. Relies on trainer.train()
        so all of the same arguments can be passed in."""
        new_quantizer.to(self.device)
        self.model.replace_quantizer(new_quantizer)

        # set encoder and decoder learning rates very low
        optimizer = torch.optim.Adam([{"params": self.model.encoders.parameters(), "lr": slow_lr},
                                      {"params": self.model.decoders.parameters(), "lr": slow_lr},
                                      {"params": self.model.quantizer.parameters(), "lr": self.model_lr}])
        self.optimizers[0] = optimizer

        if new_experiment_path:
            self.save_path, self.image_save_path = self._init_paths(save_path)

        
        self.train(**train_kwargs)


#TODO : look into quantizer with extensible codebook
#TODO : test adding regressor variables (eg speaker gender)
#TODO : clean up discriminator set up - maybe make it default?
#TODO : maybe look into loss balancer like encodec uses
#TODO : make discriminator energy calcs a method or something
#TODO : convert discriminator back to complex
#TODO : look into codebook factorization
#TODO : look into adding torch.jit for speed up

if __name__ == "__main__":
    config = yaml.safe_load(open("../config/training.yml", "r"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # update these if running on your end
    if config["experiment_name"] == "default_experiment":
        experiment_name = input("Please enter an experiment name (or nothing to make it default_experiment):")
        experiment_name = "default_experiment" if experiment_name == "" else experiment_name
    else:
        experiment_name = config["experiment_name"]

    save_path = config["save_path_root"] + experiment_name + "/"
    dataset_path = config["dataset_path"]
    sample_rate = config["sample_rate"]

    dataset, data_sample_rate = utils.get_dataset(config["dataset"], 
                                                  config["dataset_path"])
    
    if data_sample_rate != sample_rate:
        resampler = torchaudio.transforms.Resample(data_sample_rate, sample_rate)
    else:
        resampler = None

    use_discriminator = config["use_discriminator"]
    scratch_train = config["scratch_train"]
    
    model = CausalVQAE(**config["vae_args"])

    if not scratch_train:
        latest_model_path = utils.get_latest_file(save_path, "model")
    else:
        latest_model_path = None

    if use_discriminator:
        # TODO: unhardcode these? otoh they match the paper
        discriminators = [WaveFormDiscriminator(1)]
        discriminators += [STFTDiscriminator(win_length = win) for win in [2048, 
                                                                           1024, 
                                                                           512,
                                                                           256,
                                                                           128
                                                                           ]]
        
        latest_d_paths = []

        if scratch_train:
            latest_d_paths = [None] * len(discriminators)
        else:
            for discriminator in discriminators:
                latest_d_paths.append(utils.get_latest_file(save_path, discriminator.name))
    else:
        discriminators = None
        latest_d_paths = None

    
    scheduler = WarmUpScheduler(torch.optim.Adam(model.parameters(), 
                                                 lr = config["lr"], 
                                                 amsgrad = True), 
                                torch.optim.lr_scheduler.CosineAnnealingLR, 
                                warmup_iter = config["scheduler_warmup"])
    losses = []

    trainer = Trainer(device, 
                      save_path,
                      model, 
                      dataset,
                      sample_rate = sample_rate,
                      resampler = resampler,
                      scheduler = scheduler,
                      model_path = latest_model_path,
                      discriminators = discriminators,
                      discriminator_paths = latest_d_paths,
                      **config["trainer_args"]
                      )
    
    losses = trainer.train(losses = losses, 
                           gan_loss = use_discriminator,
                           **config["train_run_args"])

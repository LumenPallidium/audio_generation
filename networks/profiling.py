####
# messy file i've been using to profile
####

import torch
import os
import torchaudio
from tqdm import tqdm
from vae import CausalVQAE, discriminator_generator_loss, collator, WaveFormDiscriminator, STFTDiscriminator

from torch.profiler import profile, record_function, ProfilerActivity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CausalVQAE(1, input_format = "n c l").to(device)
discriminator = WaveFormDiscriminator(1).to(device)
stft_discriminator = STFTDiscriminator().to(device)

    

# load pretrained models if they exists
if os.path.exists("C:/Projects/model_epoch_0.pt"):
    model.load_state_dict(torch.load("C:/Projects/model_epoch_0.pt"))
    print("Loaded pretrained model generator")
if os.path.exists("C:/Projects/discriminator_epoch_0.pt"):
    discriminator.load_state_dict(torch.load("C:/Projects/discriminator_epoch_0.pt"))
    print("Loaded pretrained model discriminator")
if os.path.exists("C:/Projects/stft_discriminator_epoch_0.pt"):
    stft_discriminator.load_state_dict(torch.load("C:/Projects/stft_discriminator_epoch_0.pt"))
    print("Loaded pretrained model stft discriminator")


optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
optimizer_stft_discriminator = torch.optim.Adam(stft_discriminator.parameters(), lr=1e-4)

librispeech = torchaudio.datasets.LIBRISPEECH("C:/Projects/librispeech/", url="train-clean-100", download=True)
train_loader = torch.utils.data.DataLoader(librispeech,
                                            batch_size=8,
                                            shuffle=True,
                                            collate_fn=collator,)

reconst_loss = torch.nn.MSELoss()


with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    for i, x in tqdm(enumerate(train_loader)):
        x = torch.vstack(x).unsqueeze(1).to(device)
        with record_function("autoencoder"):
            y, commit_loss, indices = model(x)
        with record_function("discriminator"):
            generator_loss, discriminator_loss = discriminator_generator_loss(x.clone(), y, discriminator)
        with record_function("stft_discriminator"):
            stft_generator_loss, stft_discriminator_loss = discriminator_generator_loss(x.clone(), y, stft_discriminator)

        with record_function("loss"):
            loss = reconst_loss(x, y) + commit_loss + generator_loss + stft_generator_loss
        with record_function("backward"):
            optimizer.zero_grad()
            optimizer_discriminator.zero_grad()
            loss.backward(retain_graph = True)
            discriminator_loss.backward(retain_graph = True)
            stft_discriminator_loss.backward()
            optimizer.step()
            optimizer_discriminator.step()
            optimizer_stft_discriminator.step()
        if i > 10:
            break
        
print(prof.key_averages(group_by_stack_n=5).table(sort_by = "cuda_time", row_limit=50))
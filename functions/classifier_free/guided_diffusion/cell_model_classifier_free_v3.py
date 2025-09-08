import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, reduce, repeat, pack, unpack
from einops.layers.torch import Rearrange
from .nn import (
    linear,
    timestep_embedding,
)

class TimeEmbedding(nn.Module):
    def __init__(self, hidden_dim):
        super(TimeEmbedding, self).__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.hidden_dim = hidden_dim

    def forward(self, t):
        return self.time_embed(timestep_embedding(t, self.hidden_dim).squeeze(1))


class LabelEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LabelEmbedding, self).__init__()
        self.label_embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, label):
        return self.label_embed(label)


class ResidualBlock_classifier_free(nn.Module): # (B, in_features) -> (B, out_features)
    def __init__(self, in_features, out_features, time_features, class_features):
        super(ResidualBlock_classifier_free, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.emb_time_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_features, out_features)
        )
        self.emb_label_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(class_features, out_features)
        )
        self.act = nn.SiLU()
        self.drop = nn.Dropout(0)

    def forward(self, x, time_emb, class_emb):
        h = self.fc(x)
        h = h * self.emb_label_layer(class_emb) + self.emb_time_layer(time_emb)
        h = self.norm(h)
        h = self.act(h)
        h = self.drop(h)
        return h


class Cell_Unet_classifier_free(nn.Module):
    def __init__(self, input_dim, class_input_dim=64, hidden_num=[512,512,256,128], dropout=0.1, num_steps=1000,
                 branch=0, cache_interval=5, non_uniform=False, mask_prob=0.2): # input_dim should be the dimension of VAE_output(128), hidden_num is defined by users
        super(Cell_Unet_classifier_free, self).__init__()
        # class_input_dim should be entered by the user, which is the embedding dimension of the perturbation(can be multiple) for each cell(can receive multiple perturbations)
        self.mask_prob = mask_prob

        self.hidden_num = hidden_num
        self.class_input_dim = class_input_dim
        self.time_embedding = TimeEmbedding(hidden_num[0])
        self.label_embedding = LabelEmbedding(class_input_dim, hidden_num[0])

        self.layers = nn.ModuleList()

        self.layers.append(ResidualBlock_classifier_free(input_dim, hidden_num[0], hidden_num[0], hidden_num[0]))

        for i in range(len(hidden_num)-1):
            self.layers.append(ResidualBlock_classifier_free(hidden_num[i], hidden_num[i+1], hidden_num[0], hidden_num[0]))

        self.reverse_layers = nn.ModuleList()
        for i in reversed(range(len(hidden_num)-1)):
            self.reverse_layers.append(ResidualBlock_classifier_free(hidden_num[i + 1], hidden_num[i], hidden_num[0], hidden_num[0]))

        self.out1 = nn.Linear(hidden_num[0], int(hidden_num[1]*2))
        self.norm_out = nn.LayerNorm(int(hidden_num[1]*2))
        self.out2 = nn.Linear(int(hidden_num[1]*2), input_dim, bias=False)

        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        if non_uniform: # Not using this line!
            self.interval_seq, _ = sample_from_quad_center(num_steps, num_steps // cache_interval, 120, 1.5)
        else:
            self.interval_seq = list(range(0, num_steps, cache_interval))
        A = [0] * num_steps
        for i in range(len(self.interval_seq)):
            A[self.interval_seq[i]] = 1
        self.interval_seq = A # self.interval_seq is like [1, 0, 0, 0, 0, 0, 1, 0, ...], skip 5 time steps, which is controlled by cache_interval(default 5)
        self.prv_f = None
        self.branch = branch # control from which downsample block to skip, default is 0
        self.context_mask = None

    def forward(self, x, t, class_emb, inference=False, all_mask=False, y=None): # all_mask is changed in diffusion settings
        # class_emb is a user-defined embedding matrix, both x and class_embedding are from DataLoader
        if inference: # Now this is the sampling procedure, we need to create \hat s_t(x, c) and \hat s_t(x, c=\emptyset) so we need two sets of label y
            # of course, when inference is set to be True, we will receive x in shape (batch_size*2, dim) and t in shape (batch_size*2, dim)
            batch_size = int(x.shape[0]//2)
            class_emb = class_emb.repeat(2, 1) # become (batch_size*2, dim)
            if self.context_mask == None:
                context_mask = torch.zeros_like(class_emb).to(t.device)
                context_mask[:batch_size] = 1
                self.context_mask = context_mask
            else:
                context_mask = self.context_mask

            if all_mask: # For sanity checking.....
                context_mask = torch.zeros_like(class_emb).to(t.device)
        else:
            num_class_emb, _ = class_emb.shape
            context_mask = torch.bernoulli(torch.full((num_class_emb, 1), 1-self.mask_prob, device=t.device)) # when self.mask_prob=0.2, around 80% will be 1(retained)
            # mask_prob will be set in model settings
            # mask should make sure in most cases, we train with many class embeddings not masked, so mask_prob should be around 0.2(80% will be retained)

        class_emb = class_emb * context_mask # The first half corresponds to \hat s_t(x, c) and the second half corresponds to \hat s_t(x, c=\emptyset)
        time_emb = self.time_embedding(t)
        class_emb = self.label_embedding(class_emb)

        x = x.float()

        if inference:
            n_layer = len(self.reverse_layers) # mid layer is counted as layers(downsampling)
            assert 0 <= self.branch < n_layer

            if self.interval_seq[t[0]] == 1:
                self.prv_f = None
            history = []

            for i, layer in enumerate(self.layers):
                x = layer(x, time_emb, class_emb)
                history.append(x)
                if i == self.branch and self.prv_f is not None:
                    break # skip the following several downsampling and upsampling
            if self.prv_f == None:
                if self.branch == n_layer - 1: # happen in the mid block
                    self.prv_f = history[-1]

                history.pop()

                for i, layer in enumerate(self.reverse_layers):
                    x = layer(x, time_emb, class_emb)
                    x = x + history.pop()
                    if self.branch == n_layer - i - 2: # happen in the upsampling
                        self.prv_f = x
                x = self.out1(x)
                x = self.norm_out(x)
                x = self.act(x)
                x = self.out2(x)

            else:
                x = self.prv_f
                for layer in self.reverse_layers[n_layer-1-self.branch:]:
                    x = layer(x, time_emb, class_emb)
                    x = x + history.pop()
                x = self.out1(x)
                x = self.norm_out(x)
                x = self.act(x)
                x = self.out2(x)


        else: # means Inference == False
            history = []
            for layer in self.layers:
                x = layer(x, time_emb, class_emb)
                history.append(x)

            history.pop()

            for layer in self.reverse_layers:
                x = layer(x, time_emb, class_emb)
                x = x + history.pop()

            x = self.out1(x)
            x = self.norm_out(x)
            x = self.act(x)
            x = self.out2(x)

        return x


def sample_from_quad_center(total_numbers, n_samples, center, pow=1.2):
    while pow > 1:
        # Generate linearly spaced values between 0 and a max value
        x_values = np.linspace((-center) ** (1 / pow), (total_numbers - center) ** (1 / pow), n_samples + 1)
        # print(x_values)
        # print([x for x in np.unique(np.int32(x_values**pow))[:-1]])
        # Raise these values to the power of 1.5 to get a non-linear distribution
        indices = [0] + [x + center for x in np.unique(np.int32(x_values ** pow))[1:-1]]
        if len(indices) == n_samples:
            break

        pow -= 0.02
    return indices, pow











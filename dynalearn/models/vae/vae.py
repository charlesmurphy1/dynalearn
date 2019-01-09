import numpy as np
import torch
import torch.nn as nn

import time


class VAE(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_embedding,
                 optimizer=None, use_cuda=False):
        super(VAE, self).__init__()
        
        # Model hyper_parameters        
        self.use_cuda = use_cuda
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_embedding = n_embedding
        self.optimizer = optimizer
        
        # Functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss(reduce="mean")
        
        # Inference network weights
        layers = []
        for i in range(1, len(self.n_hidden)):
            layers.append(nn.Linear(self.n_hidden[i - 1], self.n_hidden[i]))
            layers.append(self.relu)
        self.inference_layers = nn.Sequential(nn.Linear(self.n_inputs,
                                                        self.n_hidden[0]),
                                              self.relu, *layers)
        
        # Embedding networks weights
        self.mu_layer = nn.Linear(self.n_hidden[-1],
                                  self.n_embedding)
        self.logvar_layer = nn.Linear(self.n_hidden[-1],
                                      self.n_embedding)
        
        # Generative network weights
        layers = []
        for i in range(len(n_hidden) - 1, 0, -1):
            layers.append(nn.Linear(self.n_hidden[i],
                                    self.n_hidden[i - 1]))
            layers.append(self.relu)
        self.generative_layers = nn.Sequential(nn.Linear(self.n_embedding,
                                                         self.n_hidden[-1]),
                                               self.relu,
                                               *layers,
                                               nn.Linear(self.n_hidden[0],
                                                         self.n_inputs),
                                               self.sigmoid)
        
        if self.optimizer is None:
            self.optimizer = torch.optim.SGD(self.parameters(), lr = 1e-2)

        if use_cuda:
            self.inference_layers = self.inference_layers.cuda()
            self.mu_layer = self.mu_layer.cuda()
            self.logvar_layer = self.logvar_layer.cuda()
            self.generative_layers = self.generative_layers.cuda()
    
    def __inference_network(self, x):
        h = self.inference_layers(x)
        return self.mu_layer(h), self.logvar_layer(h)

    def __sample_embedding(self, mu, logvar):
        batch_size = mu.size(0)
        eps = torch.randn(batch_size, self.n_embedding)
        if self.use_cuda:
            eps = eps.cuda()
        return mu + torch.exp(logvar / 2) * eps

    def __generative_network(self, z):
        return self.generative_layers(z)

    def __vae_loss(self, inputs, outputs):
        recon_data = outputs[0]
        mu = outputs[1]
        logvar= outputs[2]
        
        recon_loss = self.loss(recon_data, inputs)
        
        KL_loss = torch.mean(0.5 * torch.sum(torch.exp(logvar) + \
                                            mu**2 - 1. - logvar, 1)
                           )
        return recon_loss + KL_loss

    def forward(self, x):
        z_mu, z_logvar = self.inference_network(x)
        z = self.sample_embedding(z_mu, z_logvar)
        y = self.generative_network(z)
        
        return y, z_mu, z_logvar
    
    def predict(self, batch_size=32):
        self.eval()
        z = torch.randn(batch_size, self.n_embedding)
        if self.use_cuda:
            z = z.cuda()
        sample = self.generative_network(z).detach().cpu().data.numpy()
        
        return sample, z

    def evaluate(self, dataset, batch_size=64):
        
        model.train(False)
        loss_value = []
        model.eval()

        data_loader = DataLoader(train_dataset, batch_size)

        for batch in data_loader:
        
            inputs, labels = batch
            
            if use_cuda:
                inputs = inputs.cuda()

            outputs = self.forward(inputs)
            loss = self.vae_loss(inputs, outputs)
            loss_value.append(loss.data)

        model.train(True)
        return sum(loss_value) / len(loss_value)


    def fit(self, train_dataset, val_dataset=None, epochs=10, batch_size=64,
            verbose=True, initial_epoch=0):

        train_loader = DataLoader(train_dataset, batch_size)
        val_loader = DataLoader(val_dataset, batch_size)

        self.train(True)

        start = time.time()
        for i in range(epochs):

            for batch in train_loader:
                self.optimizer.zero_grad()
                inputs, labels = batch
                
                if use_cuda:
                    inputs = inputs.cuda()

                loss = vae_loss(inputs, outputs)
                loss.backward()
                optimizer.step()

            if verbose:
                end = time.time()
                print(f"Epoch {i} - Train loss: {train_loss} -\
                        Val loss: {val_loss} -\
                        Training time: {end - start}")
                start = time.time()

        return None


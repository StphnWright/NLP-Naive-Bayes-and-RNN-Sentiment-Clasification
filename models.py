import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn

class RecurrentNetwork(nn.Module):

    def __init__(self, embeddings, num_class):
        super(RecurrentNetwork, self).__init__()

        ########## YOUR CODE HERE ##########
        # TODO: Here, create any layers and attributes your network needs.
        
        # Attributes
        self.embeddings = embeddings
        self.num_class = num_class
        
        EMBEDDING_DIM = embeddings.size(dim=1)
        
        self.layerEmbedding = nn.Embedding.from_pretrained(embeddings)
        self.layersHidden = nn.GRU(EMBEDDING_DIM, self.num_class, 2)
        self.layerDenseOut = nn.Linear(self.num_class, 4)
    
    
    # x is a PaddedSequence for an RNN
    def forward(self, x):
        ########## YOUR CODE HERE ##########
        # TODO: Fill in the forward pass of your neural network.
        # TODO: (The backward pass will be performed by PyTorch magic for you!)
        # TODO: Your architecture should...
        # TODO: 1) Put the words through an Embedding layer (which was initialized with the pretrained embeddings);
        # TODO: 2) Feed the sequence of embeddings through a 2-layer RNN; and
        # TODO: 3) Feed the last output state into a dense layer to become a 4-vector of values, one for each class
        
        # Record the length of each sentence
        lengthActual = []
        for x_item in x:
            lengthActual.append(len(torch.nonzero(x_item)))
            
        # Put the words through the embedding layer
        x = self.layerEmbedding(x).float()
        
        # Feed the sequence of embeddings through the 2-layer RNN
        x = rnn.pack_padded_sequence(x, lengthActual, batch_first = True, enforce_sorted = False)
        _, h_n = self.layersHidden(x)
        
        # Feed the last output state into the dense layer and return the 4-vector
        return self.layerDenseOut(h_n[-1])

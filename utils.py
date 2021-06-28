import torch
import torch.nn as nn
import torch.utils as utils

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer



### DATASET ###
class IMDB(utils.data.Dataset):
    """
    Create a dataset class with the necessary preprocessing steps. 
    Later this class is used in the dataloader object.
    """
    def __init__(self, path, max_seq_len):
        self.max_seq_len = max_seq_len
        
        # read in data
        df = pd.read_csv(path)
        # transform sentiment to integer
        df['sentiment'] = df.sentiment.map({'positive': 1, 'negative': 0})       

        # vectorize reviews
        vectorizer = CountVectorizer(stop_words= 'english', min_df= 0.01)
        vectorizer.fit(df.review.tolist())

        # dict(token:integer)
        self.token2idx = vectorizer.vocabulary_
        self.token2idx['<PAD>'] = max(self.token2idx.values()) +1
        # tokenizer
        tokenizer = vectorizer.build_analyzer()
        # encoder (string to integer)
        self.encode = lambda x: [self.token2idx[token] for token 
                                in tokenizer(x) if token in self.token2idx]
        # padding (in case sequence is shorter than max_seq_len)
        self.pad = lambda x: x + (max_seq_len - len(x)) * [self.token2idx['<PAD>']]
        # create sequences (take only max_seq_len)
        sequences = [self.encode(sequence)[:max_seq_len] for sequence 
                                                        in df.review.tolist()]

        sequences, self.labels = zip(*[(sequence, label) for sequence, label
                                    in zip(sequences, df.sentiment.tolist())
                                    if sequence])
        # create padded sequences
        self.sequences = [self.pad(sequence) for sequence in sequences]


    def __getitem__(self, idx):
        assert len(self.sequences[idx]) == self.max_seq_len
        return self.sequences[idx], self.labels[idx]


    def __len__(self):
        return len(self.sequences)



### RNN MODEL ### 
class RNN(nn.Module):
    """
    Create an GRU based (RNN) architecture to classify (binary) sentiment of reviews.
    """
    def __init__(
        self, 
        batch_size,
        embedded_vectors,
        hidden_size= 128, 
        n_layers= 1,
        device= 'cpu'):
        super().__init__()
        self.batch_size = batch_size
        self.embededd_vectors = embedded_vectors
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.device = device

        # architecture
        self.encoder = nn.Embedding.from_pretrained(embedded_vectors)
        self.rnn = nn.GRU(
            self.embededd_vectors.shape[1], 
            hidden_size, 
            num_layers = n_layers, 
            batch_first= True)
        self.decoder = nn.Linear(hidden_size, 1)

    # initialize first hidden state with random values
    def _init_hidden(self):
        return torch.randn(
            self.n_layers, self.batch_size, self.hidden_size).to(self.device)


    def forward(self, X):
        batch_size = X.shape[0]
        if batch_size != self.batch_size:
            self.batch_size = batch_size
        
        encoded = self.encoder(X)
        out, hidden = self.rnn(encoded, self._init_hidden())
        out = self.decoder(out[:, :, -1]).squeeze()
        return out



### REVIEWS ###
pos_review = """
Though this film deserves a very long review, as I have so much to say about 
it, what more needs to be said than has already been mentioned by the nation's 
critics, who have contributed to The Dark Knight's whopping 94% rating on 
Rotten Tomatoes? Some are saying it's better than Batman Begins, director 
Chris Nolan's first film in this new Batman series. I disagree, but only 
because I think that film is a lot better than some critics remember it. 
I do think, however, that The Dark Knight is just as masterful, and since I 
consider both to be nearly flawless enough to give them each five stars. """

neg_review = """
Where does one start? How can you mentally digest something like Birdemic? I 
am still in shock. I have seen some shitty movies in my time. But Birdemic, 
friends and neighbors, is the worst movie in the history of film-making, on 
this planet or in any other dimension for that matter. It is bad, OMG, right 
off the scale on the shitometer. The acting? Poor Alan Bagh, is he a living, 
walking wooden plank? Special effects? I swear, the birds are cardboard 
cutouts dangling from strings. For some reason, they explode when they hit 
something. Why? Why is that? Can't somebody explain, for freak's sake?
Everything stinks so very gaggingly. A rhesus monkey with a camcorder poking 
out of its arse would do better. Beware, my friends, beware of this 
abomination that is Birdemic."""
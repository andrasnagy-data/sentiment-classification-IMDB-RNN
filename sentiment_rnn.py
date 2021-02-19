import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
import torchtext as text

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import numpy as np



# configure device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



### DATASET ###
class IMDB(utils.data.Dataset):
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



### HYPER PARAMETERS ###
path = '../sentiment/data/IMDB Dataset.csv'
max_seq_len = 128
batch_size = 512
hidden_size = 128
learning_rate = 0.001
num_epochs = 30


# dataset instance
imdb = IMDB(path, max_seq_len)



### TRAIN & TEST SPLIT ###
train_size = int(0.8 * len(imdb))
test_size = len(imdb) - train_size
lengths = [train_size, test_size]
train_set, test_set = utils.data.random_split(
                imdb, lengths, generator= torch.Generator().manual_seed(42))



### DATA LOADER ###
# to tensor function
def toTensor(batch):
    inputs = torch.LongTensor([item[0] for item in batch])
    target = torch.FloatTensor([item[1] for item in batch])
    return inputs, target

# train & test data loader
train_loader = utils.data.DataLoader(train_set, batch_size= batch_size,
                                     shuffle= True, collate_fn= toTensor)
test_loader = utils.data.DataLoader(test_set, batch_size= batch_size, 
                                    shuffle= False,  collate_fn= toTensor)



### RNN MODEL ### 
# Glove embedding
glove_vectors = text.vocab.GloVe(name= '6B', dim= 50)
# list of tokens
tokens = list(imdb.token2idx.keys())
# tensor of embedded tokens
embedded_vectors = glove_vectors.get_vecs_by_tokens(tokens, 
                                                    lower_case_backup= True)


class RNN(nn.Module):
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


# model instance
model = RNN(batch_size, embedded_vectors, device= device)
model = model.to(device) # push model to GPU



### CRITERION AND OPTIMIZER ###
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr= learning_rate)



### TRAINING LOOP ###
train_loss = []
test_loss = []

for epoch in range(num_epochs):
    model.train()

    total_loss, n_samples = 0.0, 0
    for reviews, labels in train_loader:
        reviews, labels = reviews.to(device), labels.to(device)

        model.zero_grad()
        y_ = model(reviews)
    
        loss = criterion(y_, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * reviews.shape[0]
        n_samples += reviews.shape[0]

    print(
        f"Epoch {epoch+1}/{num_epochs} |"
        f"train loss: {total_loss / n_samples:9.3f} |")

    # append accuracy & loss to lists
    train_loss.append(total_loss / n_samples)

    # evaluation
    model.eval()

    total_loss, n_correct, n_samples = 0.0, 0, 0
    with torch.no_grad():
        for reviews, labels in test_loader:
            reviews = reviews.to(device)
            labels = labels.to(device)
                    
            y_ = model(reviews)
        
            loss = criterion(y_, labels)
            total_loss += loss.item() * reviews.shape[0]
            n_samples += reviews.shape[0]

    print(
        f"Epoch {epoch+1}/{num_epochs} |"
        f"valid loss: {total_loss / n_samples:9.3f} |")

    # append accuracy & loss to lists
    test_loss.append(total_loss / n_samples)



### VISUAL EVALUATION ###
# visualize training and testing statistics
X_values = np.arange(num_epochs)

fig = plt.figure()
ax = plt.subplot(1, 1, 1)
ax.plot(X_values, train_loss, label= 'training')
ax.plot(X_values, test_loss, label= 'testing')
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
plt.title('Training & Testing loss')
ax.legend()

plt.show()
# save figure
name = 'training_evaluation.png'
fig.savefig(name)



### PREDICTING REVIEW SENTIMENT ###
def predict_sentiment(review):
    model.eval()
    with torch.no_grad():
        review_vector = torch.LongTensor(
                                [imdb.pad(imdb.encode(review))]).to(device)
        
        y_ = model(review_vector)
        prediction = torch.sigmoid(y_).item()

        if prediction > 0.5:
            print(f'{prediction:0.4}: Positive sentiment')
        else:
            print(f'{prediction:0.4}: Negative sentiment')


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


# check sentiment of reviews
print(f"{pos_review} \n")
print(predict_sentiment(pos_review))
print(f"{neg_review} \n")
print(predict_sentiment(neg_review))
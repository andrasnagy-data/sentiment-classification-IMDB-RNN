import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
import torchtext as text

import matplotlib.pyplot as plt
import numpy as np

from utils import IMDB, RNN, pos_review, neg_review



# configure device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



### HYPER PARAMETERS ###
path = 'data/IMDB Dataset.csv'
max_seq_len = 128
batch_size = 512
hidden_size = 128
learning_rate = 0.001
num_epochs = 25
model_ckp = 'best_model.pth'
name = 'training_evaluation.png'



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



# Glove embedding
glove_vectors = text.vocab.GloVe(name= '6B', dim= 50)
# list of tokens
tokens = list(imdb.token2idx.keys())
# tensor of embedded tokens
embedded_vectors = glove_vectors.get_vecs_by_tokens(tokens, 
                                                    lower_case_backup= True)



# model instance
model = RNN(batch_size, embedded_vectors, device= device)
model = model.to(device) # push model to GPU



### TRAINING LOOP ###
"""
GRU architecture with Adam optimizer and BinaryCrossEntoryLogitsLoss 
objective function. "Early stoppage" is implemented to adress over-fitting.
"""
# criterion and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr= learning_rate)

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

    # append train loss to lists
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

    # append test loss to lists
    test_loss.append(total_loss / n_samples)

    # early stopping
    if test_loss[epoch] == min(test_loss):
        torch.save(model.state_dict(), model_ckp)
        print(f"The best parameter setting is from epoch {epoch+1}.")



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
fig.savefig(name)


# Training done
print('Training done!')

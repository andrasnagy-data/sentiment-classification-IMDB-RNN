import torch
import torchtext as text

from utils import IMDB, RNN, pos_review, neg_review



# configure device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



### HYPER PARAMETERS ###
path = 'data/IMDB Dataset.csv'
max_seq_len = 128
batch_size = 512
hidden_size = 128
checkpoint = 'best_model.pth'



# dataset instance
imdb = IMDB(path, max_seq_len)



# Glove embedding
glove_vectors = text.vocab.GloVe(name= '6B', dim= 50)
# list of tokens
tokens = list(imdb.token2idx.keys())
# tensor of embedded tokens
embedded_vectors = glove_vectors.get_vecs_by_tokens(tokens, 
                                                    lower_case_backup= True)



# model instance
model = RNN(batch_size, embedded_vectors, device= device)
model.load_state_dict(torch.load(checkpoint))
model = model.to(device) # push model to GPU



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

# check sentiment of reviews
print(f"{pos_review} \n")
print(predict_sentiment(pos_review))
print(f"{neg_review} \n")
print(predict_sentiment(neg_review))
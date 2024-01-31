import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

class BiLSTM(nn.Module):

    def __init__(self, num_layers, in_dims, hidden_dims, out_dims):
        super().__init__()

        self.lstm = nn.LSTM(in_dims, hidden_dims, num_layers, bidirectional=True)
        self.proj = nn.Linear(hidden_dims * 2, out_dims)

    def forward(self, feat):
        hidden, _ = self.lstm(feat)
        output = self.proj(hidden)
        return output


class Wav2Vec2CTC(nn.Module):

    def __init__(self, out_dims, num_freeze_layers):
        super().__init__()
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.model.freeze_feature_encoder()
        self.freeze_layers(num_freeze_layers)
        self.proj = nn.Linear(768, out_dims)

    def forward(self, feat):
        hidden = self.model(feat)
        output = self.proj(hidden.last_hidden_state)
        return output

    def freeze_layers(self, num_freeze_layers):
        counter = 0
        if num_freeze_layers != 0:
            for param in self.model.parameters():
                if counter == num_freeze_layers:
                    break 
                param.requires_grad = False
                counter += 1 

    def unfreeze_layers(self):
        for param in self.model.parameters():
            param.requires_grad = True
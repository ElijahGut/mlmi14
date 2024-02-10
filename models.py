import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

NUM_WAV2VEC_LAYERS = 13

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
    def __init__(self, out_dims, freeze_up_to_lyr, inter_rep, combine_reps):
        super().__init__()
        self.combine_reps = combine_reps
        if self.combine_reps:
            self.layer_weights = nn.Parameter(torch.ones(NUM_WAV2VEC_LAYERS) / NUM_WAV2VEC_LAYERS)
        else:
            self.layer_weights = None
        
        self.inter_rep = inter_rep
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.model.freeze_feature_encoder()
        self.freeze_layers(freeze_up_to_lyr)
        self.proj = nn.Linear(768, out_dims)

    def forward(self, feat):
        hidden = self.model(feat, output_hidden_states=True)
        representation_to_send = hidden.last_hidden_state
        if not self.combine_reps:
            # probe intermediate representation
            if self.inter_rep != 0:
                representation_to_send = hidden.hidden_states[self.inter_rep-1]
        else:
            hidden_states = hidden.hidden_states
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            representation_to_send = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        output = self.proj(representation_to_send)
        return output

    def freeze_layers(self, freeze_up_to_lyr):
        if freeze_up_to_lyr != 0:
            for n,p in self.model.named_parameters():
                if freeze_up_to_lyr == -1:
                    p.requires_grad = False
                elif freeze_up_to_lyr > 0:
                    if f'encoder.layers' in n:
                        layer_no = int(n.split('.')[2])
                        if layer_no < freeze_up_to_lyr:
                            p.requires_grad = False 
        print('FROZEN LAYERS: ')
        for n,p in self.model.named_parameters():
            print(n, p.requires_grad)
        return

    def unfreeze_layers(self):
        for param in self.model.encoder.parameters():
            param.requires_grad = True
        for param in self.model.feature_extractor.parameters():
            param.requires_grad = True

        print('UNFROZEN LAYERS: ')
        for n,p in self.model.named_parameters():
            print(n, p.requires_grad)

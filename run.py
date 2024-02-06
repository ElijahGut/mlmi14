from dataloader import get_dataloader
import torch
from collections import Counter
from datetime import datetime
from trainer import train
import models
from decoder import decode
import numpy as np
import argparse

parser = argparse.ArgumentParser(description = 'Running MLMI2 experiments')

# set arguments for training and decoding. 
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--train_json', type=str, default="data/train.json")
parser.add_argument('--val_json', type=str, default="data/dev.json")
parser.add_argument('--test_json', type=str, default="data/test.json")
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--num_layers', type=int, default=1, help="number of rnn layers")
parser.add_argument('--fbank_dims', type=int, default=23, help="filterbank dimension")
parser.add_argument('--model_dims', type=int, default=128, help="model size for rnn layers")
parser.add_argument('--concat', type=int, default=3, help="concatenating frames")
parser.add_argument('--lr', type=float, default=0.0001, help="learning rate")
parser.add_argument('--vocab', type=str, default="data/vocab.txt", help="vocabulary file path")
parser.add_argument('--use_fbank', action="store_true")
parser.add_argument('--model', type=str, default="wav2vec2")
parser.add_argument('--report_interval', type=int, default=50, help="report interval during training")
parser.add_argument('--num_epochs', type=int, default=10)

# Inserted arguments for ease of experiments
parser.add_argument('--optimiser', type=str, default="adam")
parser.add_argument('--schedule-lr', action="store_true")
parser.add_argument('--freeze-layers', type=int, default=0, help="The number of first N transformer layers to freeze. -1 is for all.")
parser.add_argument('--inter-rep', type=int, default=0, help="The transformer layer whose representation we want to directly send to the output layer")
parser.add_argument('--combine-reps', action="store_true", help="Combine representations?")
parser.add_argument('--warmup', action="store_true", help="Use warmup?")

args = parser.parse_args()


vocab = {}
with open(args.vocab) as f:
    for id, text in enumerate(f):
        vocab[text.strip()] = id

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

print(args)
args.device = device
args.vocab = vocab

if args.model == "wav2vec2":
    model = models.Wav2Vec2CTC(len(args.vocab), args.freeze_layers, args.inter_rep, args.combine_reps)
else:
    model = models.BiLSTM(args.num_layers, args.fbank_dims * args.concat, args.model_dims, len(args.vocab))

num_params = sum(p.numel() for p in model.parameters())
print('Total number of model parameters is {}'.format(num_params))


start = datetime.now()
model.to(args.device)
model_path = train(model, args)
end = datetime.now()
duration = (end - start).total_seconds()
print('Training finished in {} minutes.'.format(divmod(duration, 60)[0]))
print('Model saved to {}'.format(model_path))

print('Loading model from {}'.format(model_path))
model.load_state_dict(torch.load(model_path))
model.eval()
model.to(device)
results_clean = decode(model, args, args.test_json)
# results_other = decode(model, args, 'data/test_other.json')
print("CLEAN\n SUB: {:.2f}%, DEL: {:.2f}%, INS: {:.2f}%, COR: {:.2f}%, PER: {:.2f}%\n".format(*results_clean))
# print("OTHER\n SUB: {:.2f}%, DEL: {:.2f}%, INS: {:.2f}%, COR: {:.2f}%, PER: {:.2f}%\n".format(*results_other))

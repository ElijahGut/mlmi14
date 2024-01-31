from jiwer import compute_measures, cer
import torch
import models

from dataloader import get_dataloader, get_dataloader_wav
from utils import concat_inputs
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
parser.add_argument('--optimiser', type=str, default="sgd")
parser.add_argument('--schedule-lr', action="store_true")
parser.add_argument('--freeze-layers', type=int, default=0, help="The number of first N transformer layers to freeze. -1 is for all.")

args = parser.parse_args()

# context code

vocab = {}
with open('data/vocab.txt') as f:
    for id, text in enumerate(f):
        vocab[text.strip()] = id

model = models.Wav2Vec2CTC(len(vocab), 0)
model_path = 'checkpoints/20240127_220426/model_20'

device = 'cpu'

print('Loading model from {}'.format(model_path))
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()
model.to(device)

def decode(model, args, json_file, char=True):
    idx2grapheme = {y: x for x, y in vocab.items()}
    test_loader = get_dataloader_wav(json_file, 1, False)
    stats = [0., 0., 0., 0.]
    for data in test_loader:
        inputs, in_lens, trans, _ = data
        inputs = inputs.to(device)
        in_lens = in_lens.to(device)
        if args.use_fbank:
            inputs, in_lens = concat_inputs(inputs, in_lens, factor=args.concat)
        else:
            inputs = inputs.transpose(0, 1)
            in_lens = None
        with torch.no_grad():
            outputs = torch.nn.functional.softmax(model(inputs), dim=-1)
            outputs = torch.argmax(outputs, dim=-1)
            if in_lens is not None:
                outputs = outputs.transpose(0, 1)
        outputs = [[idx2grapheme[i] for i in j] for j in outputs.tolist()]
        outputs = [[v for i, v in enumerate(j) if i == 0 or v != j[i - 1]] for j in outputs]
        outputs = [list(filter(lambda elem: elem != "_", i)) for i in outputs]
        outputs = [" ".join(i) for i in outputs]
        if char:
            cur_stats = cer(trans, outputs, return_dict=True)
        break

decode(model, args, 'data/test.json')
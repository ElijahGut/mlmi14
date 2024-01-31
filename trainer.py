from datetime import datetime
from pathlib import Path

import torch
from torch.nn import CTCLoss
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import log_softmax
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ChainedScheduler, LinearLR, ConstantLR
from decoder import decode
from utils import concat_inputs

from dataloader import get_dataloader, get_dataloader_wav

UNFREEZE_EPOCH = 3

def train(model, args):
    torch.manual_seed(args.seed)
    if args.use_fbank:
        train_loader = get_dataloader(args.train_json, args.batch_size, True)
        val_loader = get_dataloader(args.val_json, args.batch_size, False)
    else:
        train_loader = get_dataloader_wav(args.train_json, args.batch_size, True)
        val_loader = get_dataloader_wav(args.val_json, args.batch_size, False)

    criterion = CTCLoss(zero_infinity=True)

    if args.optimiser == "sgd":
        optimiser = SGD(model.parameters(), lr=args.lr)
    else:
        optimiser = Adam(model.parameters(), lr=args.lr)

    # Set the scheduler
    if args.schedule_lr:
        first_milestone = round(0.5*args.num_epochs)
        second_milestone = args.num_epochs - first_milestone
        scheduler_1 = ConstantLR(optimiser, factor=1, total_iters=first_milestone, verbose=True)
        scheduler_2 = LinearLR(optimiser, start_factor=1, end_factor=0, total_iters=second_milestone, verbose=True)

    def train_one_epoch(epoch):
        running_loss = 0.0
        last_loss = 0.0

        for idx, data in enumerate(train_loader):
            inputs, in_lens, trans, durations = data
            inputs = inputs.to(args.device)
            in_lens = in_lens.to(args.device)
            if args.use_fbank:
                inputs, in_lens = concat_inputs(inputs, in_lens, factor=args.concat)
            else:
                inputs = inputs.transpose(0, 1)
                in_lens = None

            targets = [
                torch.tensor(
                    list(map(lambda x: args.vocab[x], target.split())), dtype=torch.long
                )
                for target in trans
            ]
            out_lens = torch.tensor(
                [len(target) for target in targets], dtype=torch.long
            )
            targets = pad_sequence(targets, batch_first=True)
            targets = targets.to(args.device)

            optimiser.zero_grad()
            outputs = log_softmax(model(inputs), dim=-1)
            if in_lens is None:
                in_lens = torch.tensor(
                    [
                        outputs.shape[1] * duration / max(durations)
                        for duration in durations
                    ],
                    dtype=torch.long,
                )
                outputs = outputs.transpose(0, 1)
            loss = criterion(outputs, targets, in_lens, out_lens)
            loss.backward()
            optimiser.step()

            running_loss += loss.item()
            if idx % args.report_interval + 1 == args.report_interval:
                last_loss = running_loss / args.report_interval
                print("  batch {} loss: {}".format(idx + 1, last_loss))
                tb_x = epoch * len(train_loader) + idx + 1
                running_loss = 0.0
        return last_loss

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Path("checkpoints/{}".format(timestamp)).mkdir(parents=True, exist_ok=True)
    best_val_loss = 1e6

    for epoch in range(args.num_epochs):
        model.train(True)

        print(f"EPOCH {epoch+1}:")

        # ADD THE FREEZING HERE. ASSUME MODEL IS FROZEN BEFORE.
        if args.schedule_lr:
            if epoch == UNFREEZE_EPOCH:
                model.unfreeze_layers()
            if epoch < first_milestone:
                scheduler = scheduler_1
            else:
                scheduler = scheduler_2
            print(f"ETA {scheduler.get_last_lr()}\n")

        for param in model.parameters():
            print('isNotFrozen ', param.requires_grad)
            break
        avg_train_loss = train_one_epoch(epoch)

        model.train(False)
        running_val_loss = 0.0
        for idx, data in enumerate(val_loader):
            inputs, in_lens, trans, durations = data
            inputs = inputs.to(args.device)
            in_lens = in_lens.to(args.device)
            if args.use_fbank:
                inputs, in_lens = concat_inputs(inputs, in_lens, factor=args.concat)
            else:
                inputs = inputs.transpose(0, 1)
                in_lens = None
            targets = [
                torch.tensor(
                    list(map(lambda x: args.vocab[x], target.split())), dtype=torch.long
                )
                for target in trans
            ]
            out_lens = torch.tensor(
                [len(target) for target in targets], dtype=torch.long
            )
            targets = pad_sequence(targets, batch_first=True)
            targets = targets.to(args.device)
            with torch.no_grad():
                outputs = log_softmax(model(inputs), dim=-1)
            if in_lens is None:
                in_lens = torch.tensor(
                    [
                        outputs.shape[1] * duration / max(durations)
                        for duration in durations
                    ],
                    dtype=torch.long,
                )
                outputs = outputs.transpose(0, 1)
            val_loss = criterion(outputs, targets, in_lens, out_lens)
            running_val_loss += val_loss
        avg_val_loss = running_val_loss / len(val_loader)
        if args.schedule_lr:
            scheduler.step()
            
        val_decode = decode(model, args, args.val_json)
        print(
            "LOSS train {:.5f} valid {:.5f}, valid PER {:.2f}%".format(
                avg_train_loss, avg_val_loss, val_decode[4]
            )
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = "checkpoints/{}/model_{}".format(timestamp, epoch + 1)
            torch.save(model.state_dict(), model_path)
    return model_path

import os

os.environ["LOG_LEVEL"] = "DEBUG"

import torch
from torch import nn
from tqdm import tqdm

import eegain
from eegain.data import EEGDataloader
from eegain.data.datasets import DEAP, MAHNOB, SeedIV
from eegain.logger import EmotionLogger
from eegain.models import DeepConvNet, EEGNet, ShallowConvNet, TSception, AdaptiveTSception
from helper import count_parameters


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_one_epoch(model, loader, loss_fn):
    model.eval()
    all_preds, all_actuals = [], []
    dataset_size, running_loss, epoch_loss = 0, 0, None

    pbar = tqdm(enumerate(loader), total=len(loader), desc="Test ")
    with torch.no_grad():
        for i, (x_batch, y_batch) in pbar:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            out = model(x_batch)
            loss = loss_fn(out, y_batch)
            _, pred = torch.max(out, 1)

            batch_size = x_batch.size(0)
            dataset_size += batch_size
            all_preds.extend(pred.data.tolist())
            all_actuals.extend(y_batch.data.tolist())
            running_loss += loss.item() * batch_size
            epoch_loss = running_loss / dataset_size
            pbar.set_postfix(loss=f"{epoch_loss:0.4f}")

    return all_preds, all_actuals, epoch_loss


def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    all_preds, all_actuals = [], []
    dataset_size, running_loss, epoch_loss = 0, 0, None

    pbar = tqdm(enumerate(loader), total=len(loader), desc="Train ")
    for i, (x_batch, y_batch) in pbar:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        out = model(x_batch)
        loss = loss_fn(out, y_batch)
        _, pred = torch.max(out, 1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = x_batch.size(0)
        dataset_size += batch_size
        all_preds.extend(pred.data.tolist())
        all_actuals.extend(y_batch.data.tolist())
        running_loss += loss.item() * batch_size
        epoch_loss = running_loss / dataset_size
        pbar.set_postfix(loss=f"{epoch_loss:0.4f}")

    return all_preds, all_actuals, epoch_loss


def run(
    model,
    train_dataloader,
    test_dataloader,
    test_subject_ids,
    optimizer,
    loss_fn,
    epoch,
):
    for i in range(epoch):
        print(f"\nEpoch {i}/{epoch}")

        train_pred, train_actual, train_loss = train_one_epoch(
            model, train_dataloader, optimizer, loss_fn
        )
        test_pred, test_actual, test_loss = test_one_epoch(
            model, test_dataloader, loss_fn
        )

        logger.log(test_subject_ids[0], train_pred, train_actual, i, "train", train_loss)
        logger.log(test_subject_ids[0], test_pred, test_actual, i, "val", test_loss)

    logger.log_summary()


# -------------- Preprocessing --------------
transform = eegain.transforms.Construct(
    [
        eegain.transforms.Crop(t_min=30, t_max=-30),
        eegain.transforms.DropChannels(
            [
                "EXG1",
                "EXG2",
                "EXG3",
                "EXG4",
                "EXG5",
                "EXG6",
                "EXG7",
                "EXG8",
                "GSR1",
                "GSR2",
                "Erg1",
                "Erg2",
                "Resp",
                "Temp",
                "Status",
            ]
        ),
        eegain.transforms.Filter(l_freq=0.3, h_freq=45),
        eegain.transforms.NotchFilter(freq=50),
        eegain.transforms.Resample(s_rate=128),
        eegain.transforms.Segment(duration=4, overlap=0),
    ]
)


# -------------- Dataset --------------
mahnob_dataset = MAHNOB(
    "../Sessions/",
    label_type="V",
    transform=transform,
)

# -------------- Dataloader --------------
eegloader = EEGDataloader(mahnob_dataset, batch_size=32).get_subject(1, 4)

# -------------- Training --------------
logger = EmotionLogger(log_dir="logs/full_fine_tuning/", class_names=["low", "high"])

# # -------------- Model --------------
sd = torch.load("../tsceptionMahnob.pth")

adapter_model = AdaptiveTSception(
    num_classes=2,
    input_size=(1, 32, 512),
    sampling_r=128,
    num_t=15,
    num_s=15,
    hidden=32,
    dropout_rate=0.3,
    add_bitfit=False,
    add_rebuffi_before=False,
    add_rebuffi_after=False,
    add_chapter=False,
    adapter_compression_rate=15)

adapter_model.load_state_dict(sd, strict=False)
adapter_model.load_missing_parametes(sd)
# adapter_model.freeze()

adapter_count = count_parameters(adapter_model)
print(adapter_count)

batch_size = 1
num_eeg_channels = 32  # Number of EEG channels
num_data_points = 512  # Number of data points per channel

adapter_model = adapter_model.to(device)
optimizer = torch.optim.Adam(adapter_model.parameters(), lr=1e-4, weight_decay=0)
# optimizer = torch.optim.SGD(adapter_model.parameters(), lr=1e-5)
loss_fn = nn.CrossEntropyLoss()
run(
    model =adapter_model,
    train_dataloader = eegloader["train"],
    test_dataloader  = eegloader["test"],
    test_subject_ids = eegloader["test_subject_indexes"],
    optimizer = optimizer,
    loss_fn   = loss_fn,
    epoch = 100,
)
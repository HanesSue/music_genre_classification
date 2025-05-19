import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import torchaudio.transforms as T
from torchsummary import summary
from tqdm import tqdm
from datetime import datetime
from utils.FMADataset import FMADataset
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
)
import torch.optim as optim
from functools import partial
from transformers import get_cosine_schedule_with_warmup

from model import *
from utils import FMADataset, collate_fn

def train(
    model,
    train_loader,
    test_loader,
    device,
    optimizer,
    criterion,
    scheduler,
    num_epochs,
    early_stop,
    patience=30,
):
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    log = []
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        # 使用 tqdm 包装 train_loader
        train_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{num_epochs}",
        )

        for i, (feat, _, genre_tops) in train_bar:
            mel, _, _ = feat
            mel = mel.to(device)
            genre_tops = genre_tops.to(device)
            outputs = model(mel)
            loss = criterion(outputs, genre_tops)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == genre_tops).sum().item()
            total += genre_tops.size(0)

            # 实时更新 tqdm 描述
            train_bar.set_postfix(loss=loss.item())
            # scheduler.step()

        accuracy = 100 * correct / total
        train_losses.append(total_loss / len(train_loader))
        train_accuracies.append(accuracy)
        print(
            f"[Train] Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}, Accuracy: {accuracy:.2f}%"
        )

        # 验证阶段
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            test_bar = tqdm(test_loader, desc="Validation", total=len(test_loader))
            for feat, _, genre_tops in test_bar:
                mel, _, _ = feat
                mel = mel.to(device)
                genre_tops = genre_tops.to(device)
                outputs = model(mel)
                loss = criterion(outputs, genre_tops)
                test_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == genre_tops).sum().item()
                total += genre_tops.size(0)

                # 实时更新 tqdm 描述
                test_bar.set_postfix(loss=loss.item())
        scheduler.step()

        accuracy = 100 * correct / total
        val_losses.append(test_loss / len(test_loader))
        val_accuracies.append(accuracy)

        print(
            f"[Eval] Test Loss: {test_loss / len(test_loader):.4f}, Accuracy: {accuracy:.2f}%"
        )
        log.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_losses[-1],
                "train_accuracy": train_accuracies[-1],
                "val_loss": val_losses[-1],
                "val_accuracy": val_accuracies[-1],
            }
        )

        model_name = model.__class__.__name__

        with open(f"./checkpoint/{model_name}_log.json", "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2)

        if early_stop:
            if accuracy > best_val_acc:
                best_val_acc = accuracy
                patience_counter = 0
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                torch.save(
                    model.state_dict(),
                    f"./checkpoint/{model_name}_epoch_{epoch+1}_{timestamp}.pth",
                )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping")
                    break
        else:
            if (epoch + 1) // 5 == 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                torch.save(
                    model.state_dict(),
                    f"./checkpoint/{model_name}_epoch_{epoch+1}_{timestamp}.pth",
                )


sample_rate = 12000
n_fft = 512
hop_length = 256
n_mels = 96
batch_size = 16


def get_cb_loss_weights(class_counts, beta=0.9999):
    effective_num = 1.0 - torch.pow(beta, class_counts)
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum() * len(weights)
    return weights


if __name__ == "__main__":
    fma_train = FMADataset(
        metadata_dir="./albums/train/fma_metadata",
        audio_dir="./albums/train/fma_medium",
        duration=30,
        train=True,
        validation=False,
        sample_rate=sample_rate,
    )
    fma_test = FMADataset(
        metadata_dir="./albums/train/fma_metadata",
        audio_dir="./albums/train/fma_medium",
        duration=30,
        train=False,
        validation=True,
        sample_rate=sample_rate,
    )
    fma_small_id_mappings = {
        2: 0,
        10: 1,
        12: 2,
        15: 3,
        17: 4,
        21: 5,
        38: 6,
        1235: 7,
    }
    fma_medium_id_mappings = {
        12: 0,
        15: 1,
        38: 2,
        21: 3,
        5: 4,
        17: 5,
        8: 6,
        4: 7,
        1235: 8,
        10: 9,
        9: 10,
        20: 11,
        14: 12,
        3: 13,
        2: 14,
        13: 15,
    }
    class_counts = torch.tensor(
        [
            4881,  # Rock
            4250,  # Electronic
            1001,  # Experimental
            961,  # Hip-Hop
            495,  # Classical
            415,  # Folk
            408,  # Old-Time / Historic
            306,  # Jazz
            245,  # Instrumental
            145,  # Pop
            142,  # Country
            94,  # Spoken
            94,  # Soul-RnB
            58,  # Blues
            14,  # International
            13,  # Easy Listening
        ],
        dtype=torch.float,
    )
    fn = partial(
        collate_fn,
        id_mappings=fma_medium_id_mappings,
        chunk=False,
        one_hot=False,
        resize=False,
    )
    train_loader = DataLoader(
        fma_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=fn,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        fma_test,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=fn,
        num_workers=4,
        pin_memory=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNNwithResNet(num_classes=16, resnet="resnet34", full_train=True).to(
        device
    )

    # 计算 class_weights，使得频次小的类别权重大
    cb_weights = get_cb_loss_weights(class_counts).to(device)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    criterion = nn.CrossEntropyLoss(weight=cb_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    # optimizer = torch.optim.AdamW(
    #     model.parameters(),
    #     lr=0.0001,
    # )
    # 余弦退火
        # num_training_step = len(train_loader) * num_epochs
    # num_warmup_step = int(0.1 * num_training_step)
    # scheduler = get_cosine_schedule_with_warmup(
    #     optimizer,
    #     num_training_steps=num_training_step,
    #     num_warmup_steps=num_warmup_step,
    #     num_cycles=1.0,
    # )
    # scheduler = OneCycleLR(optimizer, max_lr=0.005, steps_per_epoch=len(train_loader), epochs=num_epochs, pct_start=0.1, cycle_momentum=False, anneal_strategy="cos", last_epoch=-1)

    # warmup = LinearLR(optimizer, start_factor=0.1, total_iters=5)
    # cosine = CosineAnnealingLR(optimizer, T_max=num_epochs - 5)
    # scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])
    num_epochs = 75
    train(
        model,
        train_loader,
        test_loader,
        device,
        optimizer,
        criterion,
        scheduler,
        num_epochs,
        early_stop=True,
        patience=5,
    )

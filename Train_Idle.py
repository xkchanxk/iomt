import os
import csv
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class IdleDetectorTransformer(nn.Module):
    def __init__(self, input_dim=11, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.cls_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        cls_token = x[:, 0]
        out = self.cls_head(cls_token)
        return out.squeeze(-1)

class IMUSegmentDataset(Dataset):
    def __init__(self, idle_dir, active_dir, seq_len=100):
        self.samples = []
        self.labels = []
        self.seq_len = seq_len

        base_dir = os.path.dirname(os.path.abspath(__file__))
        idle_dir = os.path.join(base_dir, idle_dir)
        active_dir = os.path.join(base_dir, active_dir)

        # 加载闲暇段落
        for fname in sorted(os.listdir(idle_dir)):
            if fname.endswith(".csv"):
                path = os.path.join(idle_dir, fname)
                data = self.load_csv(path)
                if len(data) >= seq_len:
                    self.samples.append(torch.tensor(data[:seq_len], dtype=torch.float32))
                    self.labels.append(1.0)

        # 加载活动段落
        for fname in sorted(os.listdir(active_dir)):
            if fname.endswith(".csv"):
                path = os.path.join(active_dir, fname)
                data = self.load_csv(path)
                if len(data) >= seq_len:
                    self.samples.append(torch.tensor(data[:seq_len], dtype=torch.float32))
                    self.labels.append(0.0)

    def load_csv(self, path):
        with open(path, newline='') as f:
            reader = csv.reader(f)
            next(reader)  # 跳过表头
            rows = [
                [
                    float(row[1]), float(row[2]), float(row[3]),
                    float(row[4]), float(row[5]), float(row[6]),
                    float(row[7]), float(row[8]), float(row[9]),
                    int(row[10].lower() == 'true'), int(row[11].lower() == 'true')
                ] for row in reader
            ]
            return rows

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], torch.tensor(self.labels[idx], dtype=torch.float32)

def train_transformer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = IMUSegmentDataset("idle", "active", seq_len=20)
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=16)

    model = IdleDetectorTransformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    writer = SummaryWriter("runs/idle_transformer")

    for epoch in range(20):
        model.train()
        total_loss, total_correct, total_samples = 0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = (out > 0.5).float()
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)

        train_acc = total_correct / total_samples
        writer.add_scalar("Loss/train", total_loss / total_samples, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                preds = (out > 0.5).float()
                val_correct += (preds == y).sum().item()
                val_total += x.size(0)

        val_acc = val_correct / val_total
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        print(f"Epoch {epoch+1:02d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    torch.save(model.state_dict(), "idle_detector_transformer.pt")
    print("模型已保存为 idle_detector_transformer.pt")

if __name__ == "__main__":
    train_transformer()
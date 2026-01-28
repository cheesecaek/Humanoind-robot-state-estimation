import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import glob
import pandas as pd
import numpy as np


class LSTM_MLP(nn.Module):
    """
    2层LSTM + 3层MLP
    """
    
    def __init__(
        self,
        input_features: int = 47,
        output_size: int = 3,
        lstm_hidden_size: int = 128,
        mlp_hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1
    ):
        super(LSTM_MLP, self).__init__()
        
        self.lstm_hidden_size = lstm_hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0
        )

        self.mlp = nn.Sequential(
            nn.Linear(lstm_hidden_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_size, output_size)
        )
    
    def forward(self, x, hidden=None):
        lstm_out, hidden = self.lstm(x, hidden)
        print(f"LSTM输出形状: {lstm_out.shape}")

        output = self.mlp(lstm_out)
        
        return output, hidden
    
    def init_hidden(self, batch_size, device=None):
        if device is None:
            device = next(self.parameters()).device
        h0 = torch.zeros(self.num_layers, batch_size, self.lstm_hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.lstm_hidden_size, device=device)
        return (h0, c0)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader = None,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    device: str = None,
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # train
        model.train()
        total_train_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(batch_x) 
            
            loss = sum(1/3*criterion(outputs[:,:,i], batch_y[:,:,i]) for i in range(3))
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = total_train_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # validation
        avg_val_loss = None
        if val_loader is not None:
            model.eval()
            total_val_loss = 0.0
            total_val_mape_per_dim = torch.zeros(3).to(device)
            num_val_batches = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    
                    outputs, _ = model(batch_x)
                    loss = sum(1/3*criterion(outputs[:,:,i], batch_y[:,:,i]) for i in range(3))
                    mape_per_dim = (torch.abs(outputs - batch_y) / (torch.abs(batch_y) + 1e-8)).mean(dim=(0, 1))
                    
                    total_val_loss += loss.item()
                    total_val_mape_per_dim += mape_per_dim
                    num_val_batches += 1
            
            avg_val_loss = total_val_loss / num_val_batches
            avg_val_mape_per_dim = total_val_mape_per_dim / num_val_batches
            val_losses.append(avg_val_loss)
        
        if (epoch + 1) % 10 == 0:
            msg = f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}"
            if avg_val_loss is not None:
                msg += f", Val Loss: {avg_val_loss:.6f}"
                msg += f"\n    MAPE per dim: [{avg_val_mape_per_dim[0]:.4f}, {avg_val_mape_per_dim[1]:.4f}, {avg_val_mape_per_dim[2]:.4f}]"
            print(msg)
    
    return train_losses, val_losses


def load_csv_data(
    seq_len: int = 30,
    train_ratio: float = 0.8,
    stride: int = 30
):
    """
    Args:
        seq_len: 序列长度
        train_ratio: 训练集比例
        stride: 滑动窗口步长
    """
    X_sequences = []
    y_sequences = []

    for filepath in glob.glob('lin_vel_est_data/vel_est_train_*.csv'):
        df = pd.read_csv(filepath, header=None)
        print(f"CSV数据形状: {df.shape}")

        X_raw = pd.concat([df.iloc[:, 0:36], df.iloc[:, 37:48]], axis = 1).astype(np.float32)
        y_raw = df.iloc[:, 48:51].values.astype(np.float32)

        for i in range(0, len(X_raw) - seq_len + 1, stride):
            X_sequences.append(X_raw[i:i + seq_len])
            y_sequences.append(y_raw[i:i + seq_len])
    
    X_sequences = np.array(X_sequences)*100
    # x_min = X_sequences.min(axis=(0, 1))  # 每个特征的最小值
    # x_max = X_sequences.max(axis=(0, 1))  # 每个特征的最大值
    # X_sequences = 2 * (X_sequences - x_min) / (x_max - x_min + 1e-8) - 1
    y_sequences = np.array(y_sequences)*100
    # y_min = y_sequences.min(axis=(0, 1))  # 每个特征的最小值
    # y_max = y_sequences.max(axis=(0, 1))  # 每个特征的最大值
    # y_sequences = 2 * (y_sequences - y_min) / (y_max - y_min + 1e-8) - 1
    
    print(f"序列样本数: {len(X_sequences)}")
    print(f"X序列形状: {X_sequences.shape}")
    print(f"y序列形状: {y_sequences.shape}")
    

    num_samples = len(X_sequences)
    train_size = int(num_samples * train_ratio)
    
    X_train = torch.tensor(X_sequences[:train_size], dtype=torch.float32)
    y_train = torch.tensor(y_sequences[:train_size], dtype=torch.float32)
    X_val = torch.tensor(X_sequences[train_size:], dtype=torch.float32)
    y_val = torch.tensor(y_sequences[train_size:], dtype=torch.float32)
    
    print(f"\n训练集: {X_train.shape[0]} 个样本")
    print(f"验证集: {X_val.shape[0]} 个样本")
    
    return X_train, y_train, X_val, y_val


def create_dataloaders(X_train, y_train, X_val, y_val, batch_size=32):
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


if __name__ == "__main__":
    torch.manual_seed(42)

    print("=" * 50)
    print("加载数据...")
    print("=" * 50)

    X_train, y_train, X_val, y_val = load_csv_data(
        train_ratio=0.8,
    )

    train_loader, val_loader = create_dataloaders(
        X_train, y_train, X_val, y_val
    )
    
    print("\n" + "=" * 50)
    print("创建模型...")
    print("=" * 50)
    model = LSTM_MLP(
        input_features=47,
        dropout=0.1
    )
    
    print(f"模型结构:\n{model}")
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n" + "=" * 50)
    print("开始训练...")
    print("=" * 50)
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader
    )
    
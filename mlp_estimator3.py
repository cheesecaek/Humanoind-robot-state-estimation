import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

scaler_X = StandardScaler()
scaler_y = StandardScaler()

class MAPELoss(nn.Module):
    """Mape 计算"""
    def __init__(self):
        super(MAPELoss, self).__init__()
    
    def forward(self, y_pred, y_true):
        epsilon = 1e-8 
        abs_percentage_errors = torch.abs((y_true - y_pred) / (torch.abs(y_true) + epsilon))
        return abs_percentage_errors.mean()

class MLP(nn.Module):
    """
    输入: (batch_size, 102) - 当前关节46维 + 上一时刻关节46维 + joints2joints 10维
    输出: (batch_size, 3)
    """
    
    def __init__(
        self,
        input_features: int = 102,
        output_size: int = 3,
        hidden_sizes: list = None,
        dropout: float = 0.1
    ):
        super(MLP, self).__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [128, 64, 64]

        layers = []
        in_dim = input_features
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(in_dim, hidden_size),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_size

        layers.append(nn.Linear(in_dim, output_size))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader = None,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    device: str = None,
):
    global scaler_y, scaler_X

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = model.to(device)

    criterion = nn.MSELoss()

    mape_eval = MAPELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)

            loss = sum(criterion(outputs[:, i:i+1], batch_y[:, i:i+1]) for i in range(3))

            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = total_train_loss / num_batches
        train_losses.append(avg_train_loss)

        avg_val_loss = None
        avg_val_mape = None
        avg_val_mape_per_dim = None
        
        if val_loader is not None:
            model.eval()
            total_val_loss = 0.0
            total_val_mape = 0.0
            total_val_mape_per_dim = torch.zeros(3).to(device)
            num_val_batches = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    
                    outputs = model(batch_x)

                    loss = sum(criterion(outputs[:, i:i+1], batch_y[:, i:i+1]) for i in range(3))

                    # 反变换后计算MAPE

                    # outputs = outputs.cpu().numpy()
                    # outputs = scaler_y.inverse_transform(outputs)
                    # outputs = torch.from_numpy(outputs).to(device)
                    # batch_y = batch_y.cpu().numpy()
                    # batch_y = scaler_y.inverse_transform(batch_y)
                    # batch_y = torch.from_numpy(batch_y).to(device)


                    mape = mape_eval(outputs, batch_y).item()

                    mape_per_dim = (torch.abs(outputs - batch_y) / (torch.abs(batch_y) + 1e-8)).mean(dim=0)
                    
                    total_val_loss += loss.item()
                    total_val_mape += mape
                    total_val_mape_per_dim += mape_per_dim
                    num_val_batches += 1
            
            avg_val_loss = total_val_loss / num_val_batches
            avg_val_mape = total_val_mape / num_val_batches
            avg_val_mape_per_dim = total_val_mape_per_dim / num_val_batches
            val_losses.append(avg_val_loss)

        if(epoch + 1) % 10 == 0:
            msg = f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}"
            if avg_val_loss is not None:
                msg += f", Val Loss: {avg_val_loss:.6f}, Val MAPE: {avg_val_mape:.4f}"
                msg += f"\n    MAPE per dim: [{avg_val_mape_per_dim[0]:.4f}, {avg_val_mape_per_dim[1]:.4f}, {avg_val_mape_per_dim[2]:.4f}]"
            print(msg)
    
    return train_losses, val_losses

def load_csv_data(train_ratio: float = 0.8, normalize: bool = True):
    global scaler_X, scaler_y
    joints2joints = np.array([0.0614, 0.0438, 0.1041, 0.0915, 0.2, 0, 0.2751, 0.0438, 0.1041, 0.0413]) #base2hip_p, hpi_p2hip_r, hip_r2hip_y, hip_y2knee, knee2ankle_p, ankle_p2ankle_r, base2shoulder_p, shoulder_p2shoulder_r, shoulder_r2shoulder_y, shoulder_y2elbow
    
    all_files = sorted(glob.glob('lin_vel_est_data/vel_est_train_run_*.csv'))
    np.random.shuffle(all_files)
    
    num_files = len(all_files)
    train_file_count = int(num_files * train_ratio)
    train_files = all_files[:train_file_count]
    val_files = all_files[train_file_count:]
    
    def process_files(file_list):
        X_list = []
        y_list = []
        for filepath in file_list:
            df = pd.read_csv(filepath, header=None)
            print(f"加载 {filepath}, 形状: {df.shape}")

            # 从第1行开始取数据，这样可以直接使用第0行作为上一时刻的关节信息
            # 当前时刻的关节信息 (从第1行开始)
            joints_current = pd.concat([df.iloc[1:, 1:36], df.iloc[1:, 37:48]], axis=1).astype(np.float32)
            # 上一时刻的关节信息 (从第0行开始，到倒数第2行)
            joints_prev = pd.concat([df.iloc[:-1, 1:36], df.iloc[:-1, 37:48]], axis=1).astype(np.float32)

            joints_current = joints_current.reset_index(drop=True)
            joints_prev = joints_prev.reset_index(drop=True)
            X_raw = pd.concat([joints_current, joints_prev], axis=1).astype(np.float32)
            
            vel_raw = df.iloc[1:, 48:51].values.astype(np.float32)
            pos_raw = df.iloc[:, 51:54]
            pos_diff = pos_raw.diff(axis=0).iloc[1:].values.astype(np.float32)#第t行-第t-1行 = t时刻位移
            y_raw = np.concatenate([vel_raw, pos_diff], axis=1).astype(np.float32)
            
            X_list.append(X_raw.values)
            y_list.append(y_raw[:, 3:6])  # 0-3速度，3-6位移
        
        X = np.concatenate(X_list, axis=0)
        X_joints = np.tile(joints2joints, (X.shape[0], 1))
        X = np.concatenate([X, X_joints], axis=1)
        y = np.concatenate(y_list, axis=0)
        return X, y

    X_train, y_train = process_files(train_files)
    X_val, y_val = process_files(val_files)
    
    print(f"\n训练集原始统计:")
    print(f"  X: min={X_train.min():.4f}, max={X_train.max():.4f}, mean={X_train.mean():.4f}")
    print(f"  y: min={y_train.min():.4f}, max={y_train.max():.4f}, mean={y_train.mean():.4f}")

    if normalize:
        X_train = scaler_X.fit_transform(X_train)
        X_val = scaler_X.transform(X_val)
        y_train = scaler_y.fit_transform(y_train)
        y_val = scaler_y.transform(y_val)
        print(f"\n标准化后:")
        print(f"  X_train: mean={X_train.mean():.4f}, std={X_train.std():.4f}")
        print(f"  y_train: mean={y_train.mean():.4f}, std={y_train.std():.4f}")
        print(f"  y_scaler mean: {scaler_y.mean_}, scale: {scaler_y.scale_}")

    print(f"\n训练集样本数: {len(X_train)}")
    print(f"验证集样本数: {len(X_val)}")
    print(f"X形状: {X_train.shape}")
    print(f"y形状: {y_train.shape}")
    
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    
    print(f"训练集: {X_train.shape[0]} 个样本")
    print(f"验证集: {X_val.shape[0]} 个样本")
    
    return X_train, y_train, X_val, y_val


def create_dataloaders(X_train, y_train, X_val, y_val, batch_size=128):
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 50)
    print("加载数据...")
    print("=" * 50)

    X_train, y_train, X_val, y_val = load_csv_data(train_ratio=0.8, normalize=True)

    train_loader, val_loader = create_dataloaders(
        X_train, y_train, X_val, y_val
    )

    print("\n" + "=" * 50)
    print("创建模型...")
    print("=" * 50)
    
    model = MLP()
    # Baseline计算
    y_mean = y_train.mean(dim=0)
    baseline_pred = y_mean.unsqueeze(0).expand_as(y_val)
    mape_list = []
    for i in range(len(y_val)):
        mape = torch.abs(y_mean - y_val[i]) / (torch.abs(y_val[i]) + 1e-8)
        mape_list.append(mape)

    mape_all = torch.cat([mape_list[i].unsqueeze(0) for i in range(len(mape_list))], dim=0)
    print(mape_all.shape)
    print(f"Baseline Validation MAPE: {mape_all.mean(dim=0)}")

    print("\n" + "=" * 50)
    print("开始训练...")
    print("=" * 50)
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
    )
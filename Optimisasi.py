# optimize.py
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import os
import numpy as np
import time
import warnings
import joblib
import matplotlib.pyplot as plt

from model import ModifiedIntegratedModel  

warnings.filterwarnings('ignore')

# ==================== DEVICE SETUP ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
else:
    torch.set_num_threads(8)
    print("⚠️ Using CPU")

# ==================== PROGRESS TRACKER ====================
class ProgressTracker:
    def __init__(self, total_trials):
        self.total = total_trials
        self.current = 0
        self.start = time.time()
        self.best_val = 0
        self.best_params = None
        self.history = []
        self.times = []

    def __call__(self, study, trial):
        self.current += 1
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None:
            elapsed = time.time() - self.start
            avg_time = elapsed / self.current
            remaining = avg_time * (self.total - self.current)

            if trial.value > self.best_val:
                self.best_val = trial.value
                self.best_params = trial.params

            self.history.append(trial.value)
            self.times.append(elapsed / 60)

            print(f"Trial {self.current:2d}/{self.total} | "
                  f"Best: {self.best_val:.4f} | Current: {trial.value:.4f} | "
                  f"Elapsed: {elapsed/60:5.1f}m | Remaining: {remaining/60:5.1f}m")

# ==================== OBJECTIVE FUNCTION ====================
def objective(trial):
    # Suggest hyperparameters
    hidden_size = trial.suggest_categorical('hidden_size', [512, 768, 1024, 1280, 1536, 1792])
    num_heads = trial.suggest_categorical('num_heads', [4, 8, 12, 16])
    
    if hidden_size % num_heads != 0:
        raise optuna.TrialPruned()

    config = {
        'hidden_size': hidden_size,
        'num_of_attention_heads': num_heads,
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.1),
        'learning_rate': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64])
    }

    # Data loading
    data_dir = '/home/nathasyasiregar/opsi_alzheimer/env_alzheimer/10_per_files'
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_set = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform)
    val_set = datasets.ImageFolder(os.path.join(data_dir, 'valid'), transform)
    num_classes = len(train_set.classes)

    num_workers = 4 if device.type == 'cuda' else 2
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True,
                              num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False,
                            num_workers=num_workers, pin_memory=True, persistent_workers=True)

    # Model & training setup
    model = ModifiedIntegratedModel(config, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'],
                            weight_decay=config['weight_decay'], amsgrad=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_acc = 0
    patience_counter = 0
    max_epochs = 15
    patience = 6

    for epoch in range(max_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        scheduler.step(val_acc)
        trial.report(val_acc, epoch)

        if trial.should_prune() or (epoch >= 3 and val_acc < 0.4):
            raise optuna.TrialPruned()

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return best_acc

# ==================== VISUALIZATION ====================
def plot_results(study, tracker):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # 1. Trial history
    if tracker.history:
        axes[0].plot(tracker.history, 'b-o')
        axes[0].set_title('Validation Accuracy per Trial')
        axes[0].set_xlabel('Trial')
        axes[0].set_ylabel('Accuracy')

    # 2. Parameter importance
    try:
        imp = optuna.importance.get_param_importances(study)
        axes[1].bar(imp.keys(), imp.values(), color='skyblue')
        axes[1].set_title('Parameter Importance')
        axes[1].tick_params(axis='x', rotation=45)
    except:
        axes[1].text(0.5, 0.5, 'Importance\nN/A', ha='center', va='center')

    # 3. Histogram
    if tracker.history:
        axes[2].hist(tracker.history, bins=15, color='lightgreen', edgecolor='black')
        axes[2].set_title('Accuracy Distribution')

    # 4. Time vs accuracy
    if len(tracker.history) == len(tracker.times):
        axes[3].scatter(tracker.times, tracker.history, c=range(len(tracker.history)), cmap='viridis')
        axes[3].set_xlabel('Time (min)')
        axes[3].set_ylabel('Accuracy')
        axes[3].set_title('Performance vs Time')

    # 5. Convergence
    if tracker.history:
        cum_best = np.maximum.accumulate(tracker.history)
        axes[4].plot(cum_best, 'g-o')
        axes[4].set_title('Best Accuracy So Far')

    # 6. Summary
    axes[5].axis('off')
    if tracker.history:
        axes[5].text(0.1, 0.9, f"Best: {tracker.best_val:.4f}", fontsize=12)
        axes[5].text(0.1, 0.7, f"Trials: {len(tracker.history)}", fontsize=12)
        axes[5].text(0.1, 0.5, f"Mean: {np.mean(tracker.history):.4f}", fontsize=12)
        axes[5].text(0.1, 0.3, f"Std: {np.std(tracker.history):.4f}", fontsize=12)

    plt.tight_layout()
    plt.savefig('optimization_results.png', dpi=300, bbox_inches='tight')
    plt.show()

# ==================== MAIN ====================
if __name__ == '__main__':
    print("🔍 Starting hyperparameter optimization...")
    
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(n_startup_trials=10, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    )

    n_trials = 100  # ✅ Ubah di sini sesuai kebutuhan
    tracker = ProgressTracker(n_trials)

    try:
        study.optimize(objective, n_trials=n_trials, timeout=6*3600, callbacks=[tracker])
    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")

    # Results
    if study.best_trial:
        print(f"\n🏆 Best accuracy: {study.best_trial.value:.4f}")
        print("Best params:", study.best_trial.params)

        # Save
        joblib.dump(study, 'study.pkl')
        with open('best_params.txt', 'w') as f:
            f.write(f"Best accuracy: {study.best_trial.value:.4f}\n")
            for k, v in study.best_trial.params.items():
                f.write(f"{k}: {v}\n")

        plot_results(study, tracker)
        print("✅ Optimization complete. Results saved.")
    else:
        print("❌ No successful trials.")

import torch
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
from kornia.augmentation import RandomGaussianNoise

from reading_weights.data import build_image_dataloaders
from reading_weights.model import build_image_classifier
from reading_weights.decomposition import decompose_bilinear_model
from reading_weights.utils import resolve_device, set_seed


DIGIT_NAMES = {
    'mnist': [str(i) for i in range(10)],
    'fashion_mnist': ['T-shirt','Trouser','Pullover','Dress','Coat',
                      'Sandal','Shirt','Sneaker','Bag','Boot'],
}


class ProjectNoise:
    def __init__(self, norm):
        self.norm = norm

    def __call__(self, x):
        if self.norm == 0:
            return x
        noise = torch.randn_like(x)
        flat = noise.flatten(1)
        scale = self.norm / flat.norm(dim=1, keepdim=True).clamp(min=1e-8)
        noise = (flat * scale).view_as(x)
        return (x + noise).clamp(0.0, 1.0)


def get_noise(mode, value, device):
    if mode == 'norm':
        return ProjectNoise(value)
    else:  # std
        return RandomGaussianNoise(std=value).to(device)


def train_model(config, noise_fn, device, epochs):
    set_seed(config['seed'])
    bundle = build_image_dataloaders(config['dataset'], config['train'])
    model = build_image_classifier(config['model'], seed=config['seed']).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['train']['lr'],
        weight_decay=config['train']['wd'],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    model.train()
    for epoch in range(epochs):
        for x, y in bundle.train_loader:
            x, y = x.to(device), y.to(device)
            x = noise_fn(x)
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
        scheduler.step()
    return model, bundle.test_loader


def eval_accuracy(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(dim=-1) == y).sum().item()
            total += y.size(0)
    return correct / total


def run_replication(
    config_path='configs/mnist_baseline.yaml',
    noise_mode='norm',       
    noise_values=None,
    epochs=100,
):
    
    if noise_values is None:
        noise_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    if epochs is None:
        epochs = int(config['train']['epochs'])

    dataset_name = config['dataset']['name']
    class_names = DIGIT_NAMES[dataset_name]
    device = resolve_device('auto')

    out_dir = Path('results/analysis/noise_sweep') / dataset_name / noise_mode
    out_dir.mkdir(parents=True, exist_ok=True)

    # train one model per noise value, collect artifacts + accuracies
    all_artifacts = []
    all_accs = []
    for value in noise_values:
        print(f'training  mode={noise_mode}  value={value}')
        noise_fn = get_noise(noise_mode, value, device)
        model, test_loader = train_model(config, noise_fn, device, epochs=epochs)
        acc = eval_accuracy(model, test_loader, device)
        artifacts = decompose_bilinear_model(model)
        all_artifacts.append(artifacts)
        all_accs.append(acc)
        print(f'  val_acc={acc:.4f}')

    label = 'norm' if noise_mode == 'norm' else 'std'

    # one figure per digit
    for class_idx, class_name in enumerate(class_names):
        fig, axes = plt.subplots(1, len(noise_values), figsize=(3 * len(noise_values), 3))
        if len(noise_values) == 1:
            axes = [axes]

        for col, (value, artifacts, acc) in enumerate(
            zip(noise_values, all_artifacts, all_accs)
        ):
            top_v = (
                artifacts.eigenvectors_input[class_idx, -1, :]
                .view(28, 28).cpu().numpy()
            )
            vmax = abs(top_v).max()
            axes[col].imshow(top_v, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            axes[col].set_title(f'{label}={value}\n{acc*100:.1f}%', fontsize=9)
            axes[col].axis('off')

        fig.suptitle(
            f'Fig 4 — {class_name} | dataset: {dataset_name} | noise: {noise_mode}',
            fontsize=11,
        )
        fig.tight_layout()

        fname = out_dir / f'class_{class_idx:02d}_{class_name.lower().replace("/","_")}.png'
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'saved → {fname}')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Train for noise sweep analysis')
    p.add_argument('--config',  default='configs/mnist_baseline.yaml')
    p.add_argument('--mode',    default='norm', choices=['norm', 'std'])
    p.add_argument('--values',  nargs='+', type=float,
                   default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument('--epochs',  type=int, default=20)
    args = p.parse_args()

    run_replication(args.config, args.mode, args.values, args.epochs)
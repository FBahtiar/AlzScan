import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from scipy import linalg
from torchvision.models import inception_v3
import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
nz = 100
ngf = 64
ndf = 32
num_epochs = 500
lr_g = 0.0002
lr_d = 0.00005
beta1, beta2 = 0.5, 0.999
batch_size = 16
image_size = 224
num_classes = 3
real_label_smooth = 0.9
fake_label_smooth = 0.1

def weights_init(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, nz, ngf, nc=1, num_classes=3):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, nz)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf // 2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_emb = self.label_emb(labels).unsqueeze(-1).unsqueeze(-1)
        x = noise + label_emb
        x = self.main(x)
        if x.size(-1) != 224:
            x = torch.nn.functional.interpolate(x, size=224, mode='bilinear', align_corners=False)
        return x

class Discriminator(nn.Module):
    def __init__(self, nc=1, ndf=32, num_classes=3):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, 50)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Linear(ndf * 16 * 7 * 7 + 50, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        x = self.conv_layers(x).view(x.size(0), -1)
        label_emb = self.label_emb(labels)
        x = torch.cat([x, label_emb], dim=1)
        return self.classifier(x).squeeze()

class FIDCalculator:
    def __init__(self, device):
        self.device = device
        self.inception = inception_v3(pretrained=True, transform_input=False).to(device)
        self.inception.eval()
        self.inception.fc = nn.Identity()

    def get_activations(self, images):
        with torch.no_grad():
            if images.size(1) == 1:
                images = images.repeat(1, 3, 1, 1)
            images = torch.nn.functional.interpolate(images, size=299, mode='bilinear')
            images = (images - 0.5) / 0.5
            return self.inception(images).cpu().numpy()

    def calculate_fid(self, real, fake):
        act_real = self.get_activations(real)
        act_fake = self.get_activations(fake)

        mu_r, sigma_r = np.mean(act_real, axis=0), np.cov(act_real, rowvar=False)
        mu_f, sigma_f = np.mean(act_fake, axis=0), np.cov(act_fake, rowvar=False)

        diff = mu_r - mu_f
        covmean, _ = linalg.sqrtm(sigma_r @ sigma_f, disp=False)

        if not np.isfinite(covmean).all():
            covmean = linalg.sqrtm((sigma_r + np.eye(sigma_r.shape[0]) * 1e-6) @ 
                                   (sigma_f + np.eye(sigma_f.shape[0]) * 1e-6))
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        return diff.dot(diff) + np.trace(sigma_r + sigma_f - 2 * covmean)

def create_dataloaders(data_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(0.3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train = ImageFolder(os.path.join(data_path, 'train'), transform)
    valid = ImageFolder(os.path.join(data_path, 'valid'), transform)
    test = ImageFolder(os.path.join(data_path, 'test'), transform)

    train_loader = DataLoader(train, batch_size, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid, batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test, batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, valid_loader, test_loader, train.classes

def save_generated_images(generator, epoch, device, num_samples=64):
    generator.eval()
    os.makedirs(f'generated_samples/epoch_{epoch}', exist_ok=True)
    with torch.no_grad():
        for class_id in range(num_classes):
            noise = torch.randn(num_samples, nz, 1, 1, device=device)
            labels = torch.full((num_samples,), class_id, device=device, dtype=torch.long)
            fake = generator(noise, labels)
            vutils.save_image(fake, f'generated_samples/epoch_{epoch}/class_{class_id}_samples.png',
                              normalize=True, nrow=8)
    generator.train()

def train_gan(data_path):
    train_loader, valid_loader, test_loader, class_names = create_dataloaders(data_path)
    print(f"Classes: {class_names}, Batches: {len(train_loader)}")

    netG = Generator(nz, ngf, nc=1, num_classes=num_classes).to(device)
    netD = Discriminator(nc=1, ndf=ndf, num_classes=num_classes).to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr_d, betas=(beta1, beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=lr_g, betas=(beta1, beta2))
    schedulerD = optim.lr_scheduler.ExponentialLR(optimizerD, gamma=0.995)
    schedulerG = optim.lr_scheduler.ExponentialLR(optimizerG, gamma=0.995)

    fid_calculator = FIDCalculator(device)
    G_losses, D_losses, fid_scores = [], [], []
    best_fid = float('inf')

    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    fixed_labels = torch.randint(0, num_classes, (64,), device=device, dtype=torch.long)

    print("Starting training...")

    for epoch in range(num_epochs):
        epoch_d_loss = epoch_g_loss = 0.0

        g_train_freq = 1 if epoch < 50 else 2
        d_train_freq = 1

        for i, (real_images, real_labels) in enumerate(train_loader):
            real_images = real_images.to(device)
            real_labels = real_labels.to(device)
            bsz = real_images.size(0)

            real_label = torch.full((bsz,), real_label_smooth, device=device)
            fake_label = torch.full((bsz,), fake_label_smooth, device=device)

            noise_factor = max(0.05 - epoch * 0.0001, 0.01)
            real_noisy = torch.clamp(real_images + torch.randn_like(real_images) * noise_factor, -1, 1)

            # Train Discriminator
            if i % d_train_freq == 0:
                netD.zero_grad()
                out_real = netD(real_noisy, real_labels)
                errD_real = criterion(out_real, real_label)
                errD_real.backward()

                noise = torch.randn(bsz, nz, 1, 1, device=device)
                fake_labels = torch.randint(0, num_classes, (bsz,), device=device)
                with torch.no_grad():
                    fake_images = netG(noise, fake_labels)
                fake_noisy = torch.clamp(fake_images + torch.randn_like(fake_images) * noise_factor, -1, 1)

                out_fake = netD(fake_noisy, fake_labels)
                errD_fake = criterion(out_fake, fake_label)
                errD_fake.backward()

                errD = errD_real + errD_fake
                torch.nn.utils.clip_grad_norm_(netD.parameters(), 1.0)
                optimizerD.step()
            else:
                errD = torch.tensor(0.0)

            # Train Generator
            for _ in range(g_train_freq):
                netG.zero_grad()
                noise = torch.randn(bsz, nz, 1, 1, device=device)
                fake_labels = torch.randint(0, num_classes, (bsz,), device=device)
                fake_images = netG(noise, fake_labels)
                out = netD(fake_images, fake_labels)
                errG = criterion(out, real_label)
                errG.backward()
                torch.nn.utils.clip_grad_norm_(netG.parameters(), 1.0)
                optimizerG.step()

            epoch_d_loss += errD.item()
            epoch_g_loss += errG.item()

            if i % 50 == 0:
                print(f"[{epoch}/{num_epochs}][{i}/{len(train_loader)}] "
                      f"D: {errD.item():.4f} G: {errG.item():.4f}")

        if epoch > 50:
            schedulerD.step()
            schedulerG.step()

        avg_d = epoch_d_loss / len(train_loader)
        avg_g = epoch_g_loss / len(train_loader)
        G_losses.append(avg_g)
        D_losses.append(avg_d)

        # Mode collapse safeguard
        if avg_g < 0.001 and avg_d > 10.0 and epoch > 20:
            print("Mode collapse detected! Adjusting learning rates.")
            for pg in optimizerD.param_groups: pg['lr'] *= 0.5
            for pg in optimizerG.param_groups: pg['lr'] *= 2.0

        # FID every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Calculating FID at epoch {epoch + 1}...")

            real_imgs = []
            for imgs, _ in valid_loader:
                real_imgs.append(imgs)
                if len(real_imgs) * batch_size >= 500:
                    break
            real_imgs = torch.cat(real_imgs)[:500].to(device)

            netG.eval()
            fake_imgs = []
            with torch.no_grad():
                for start in range(0, 500, batch_size):
                    n = min(batch_size, 500 - start)
                    noise = torch.randn(n, nz, 1, 1, device=device)
                    labels = torch.randint(0, num_classes, (n,), device=device)
                    fake_imgs.append(netG(noise, labels))
            fake_imgs = torch.cat(fake_imgs)[:500]
            netG.train()

            fid = fid_calculator.calculate_fid(real_imgs, fake_imgs)
            fid_scores.append(fid)
            print(f"FID: {fid:.2f}")

            if fid < best_fid:
                best_fid = fid
                torch.save(netG.state_dict(), 'best_generator.pth')
                torch.save(netD.state_dict(), 'best_discriminator.pth')
                print("New best model saved!")

            save_generated_images(netG, epoch + 1, device)

        print(f"Epoch {epoch+1} – D: {avg_d:.4f}, G: {avg_g:.4f}")

    print(f"Training done. Best FID: {best_fid:.2f}")

    # Plot results
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(G_losses, label='Generator', alpha=0.7)
    plt.plot(D_losses, label='Discriminator', alpha=0.7)
    plt.legend(); plt.title('Losses')

    plt.subplot(1, 3, 2)
    if fid_scores:
        epochs = list(range(10, len(fid_scores) * 10 + 1, 10))
        plt.plot(epochs, fid_scores, 'ro-')
        plt.axhline(50, color='r', linestyle='--')
        plt.title('FID Score')

    plt.subplot(1, 3, 3)
    with torch.no_grad():
        netG.eval()
        samples = netG(fixed_noise[:16], fixed_labels[:16])
        grid = vutils.make_grid(samples, nrow=4, normalize=True)
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy()[:, :, 0], cmap='gray')
        plt.axis('off'); plt.title('Generated Samples')
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300)
    plt.show()

    return netG, netD, fid_scores

def generate_class_samples(generator_path, num_samples_per_class=200, save_path='final_generated_samples'):
    netG = Generator(nz, ngf, nc=1, num_classes=num_classes).to(device)
    netG.load_state_dict(torch.load(generator_path, map_location=device))
    netG.eval()

    class_names = ['AD', 'MCI', 'NOR']
    os.makedirs(save_path, exist_ok=True)

    with torch.no_grad():
        for class_id, name in enumerate(class_names):
            print(f"Generating {num_samples_per_class} samples for {name}...")
            folder = os.path.join(save_path, name)
            os.makedirs(folder, exist_ok=True)

            for i in range(0, num_samples_per_class, batch_size):
                n = min(batch_size, num_samples_per_class - i)
                noise = torch.randn(n, nz, 1, 1, device=device)
                labels = torch.full((n,), class_id, device=device, dtype=torch.long)
                fake = netG(noise, labels)

                for j in range(n):
                    img = (fake[j] + 1) / 2
                    pil_img = transforms.ToPILImage()(img.cpu())
                    pil_img.save(os.path.join(folder, f'{name}_{i+j+1:04d}.png'))

if __name__ == "__main__":
    DATA_PATH = "/home/nathasyasiregar/IIII/dataset/10_per_files"

    if not os.path.exists(DATA_PATH):
        print("Please set DATA_PATH to your dataset root.")
        print("Expected structure: train/{ad,mci,nor}, valid/..., test/...")
    else:
        generator, discriminator, fid_history = train_gan(DATA_PATH)
        generate_class_samples('best_generator.pth', num_samples_per_class=200)
        print("Done! Best FID:", min(fid_history) if fid_history else "N/A")

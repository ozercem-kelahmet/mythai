import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import math
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import io
import ndjson
from scipy.ndimage import gaussian_filter1d

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def load_quickdraw_data(filepath, indices_path, split='train', max_len=128):
    print(f"Loading {filepath} ({split}, max_len={max_len})...")
    with open(filepath) as f:
        data = ndjson.load(f)
    with open(indices_path) as f:
        indices = json.load(f)
    
    use_indices = indices[split]
    print(f"  Using {len(use_indices)} {split} samples")
    
    sketches = []
    pen_states = []
    
    for idx in tqdm(use_indices, desc=f"Processing {split}", leave=False):
        if idx >= len(data):
            continue
        drawing = data[idx]['drawing']
        coords = []
        pens = []
        prev_x, prev_y = 0, 0
        
        for si, stroke in enumerate(drawing):
            xs, ys = stroke[0], stroke[1]
            if len(xs) < 2:
                continue
            for i in range(len(xs)):
                if i == 0 and si == 0:
                    dx, dy = xs[i], ys[i]
                elif i == 0:
                    dx = xs[i] - prev_x
                    dy = ys[i] - prev_y
                else:
                    dx = xs[i] - xs[i-1]
                    dy = ys[i] - ys[i-1]
                pen = 0 if i < len(xs) - 1 else 1
                coords.append([dx, dy])
                pens.append(pen)
                prev_x, prev_y = xs[i], ys[i]
        
        if len(coords) < 3:
            continue
        if len(coords) > max_len:
            coords = coords[:max_len]
            pens = pens[:max_len]
        while len(coords) < max_len:
            coords.append([0, 0])
            pens.append(1)
        
        sketches.append(np.array(coords, dtype=np.float32))
        pen_states.append(np.array(pens, dtype=np.float32))
    
    print(f"  Loaded {len(sketches)} sketches")
    return np.array(sketches), np.array(pen_states)

def normalize_data(coords):
    flat = coords.reshape(-1, 2)
    mean = flat.mean(axis=0)
    std = flat.std()
    std = max(std, 1e-6)
    normalized = (coords - mean) / std
    return normalized.astype(np.float32), {'mean': mean.tolist(), 'std': float(std)}

class SketchDataset(torch.utils.data.Dataset):
    def __init__(self, coords, pen_states, class_label=0, augment=True):
        self.coords = torch.from_numpy(coords).float()
        self.pen_states = torch.from_numpy(pen_states).float()
        self.class_label = class_label
        self.augment = augment

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        coord = self.coords[idx].clone()
        pen = self.pen_states[idx].clone()
        
        if self.augment:
            if torch.rand(1).item() < 0.5:
                coord[:, 0] = -coord[:, 0]
            if torch.rand(1).item() < 0.3:
                scale = torch.empty(1).uniform_(0.85, 1.15).item()
                coord = coord * scale
            if torch.rand(1).item() < 0.2:
                angle = torch.empty(1).uniform_(-0.1, 0.1).item()
                cos_a, sin_a = math.cos(angle), math.sin(angle)
                rot = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]])
                coord = coord @ rot.T
        
        return coord, pen, self.class_label

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply(self, model):
        self.backup_data = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup_data[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup_data[name])

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

class SelfAttention1D(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
        self.scale = (channels // num_heads) ** -0.5
    
    def forward(self, x):
        B, C, L = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads, L)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = torch.einsum('bhcl,bhck->bhlk', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bhlk,bhck->bhcl', attn, v)
        out = out.reshape(B, C, L)
        return x + self.proj(out)

class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, class_dim=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.time_mlp = nn.Linear(time_dim, out_ch * 2)
        self.class_mlp = nn.Linear(class_dim, out_ch * 2) if class_dim else None
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb, c_emb=None):
        h = self.conv1(x)
        h = self.norm1(h)
        t = self.time_mlp(t_emb)
        t_scale, t_shift = t.chunk(2, dim=-1)
        h = h * (1 + t_scale.unsqueeze(-1)) + t_shift.unsqueeze(-1)
        if self.class_mlp is not None and c_emb is not None:
            c = self.class_mlp(c_emb)
            c_scale, c_shift = c.chunk(2, dim=-1)
            h = h * (1 + c_scale.unsqueeze(-1)) + c_shift.unsqueeze(-1)
        h = F.silu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        return h + self.skip(x)

class ConditionalUNet1D(nn.Module):
    def __init__(self, in_ch=2, out_ch=2, base_ch=128, n_classes=3, n_points=128):
        super().__init__()
        self.n_points = n_points
        self.n_classes = n_classes
        time_dim = base_ch * 2
        
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_ch),
            nn.Linear(base_ch, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        self.class_emb = nn.Embedding(n_classes + 1, time_dim)
        self.input_proj = nn.Conv1d(in_ch, base_ch, 1)
        
        self.down1 = ResBlock1D(base_ch, base_ch, time_dim, time_dim)
        self.down2 = ResBlock1D(base_ch, base_ch * 2, time_dim, time_dim)
        self.pool1 = nn.AvgPool1d(2)
        self.down3 = ResBlock1D(base_ch * 2, base_ch * 2, time_dim, time_dim)
        self.attn1 = SelfAttention1D(base_ch * 2, num_heads=8)
        self.down4 = ResBlock1D(base_ch * 2, base_ch * 4, time_dim, time_dim)
        self.pool2 = nn.AvgPool1d(2)
        
        self.mid1 = ResBlock1D(base_ch * 4, base_ch * 4, time_dim, time_dim)
        self.mid_attn = SelfAttention1D(base_ch * 4, num_heads=8)
        self.mid2 = ResBlock1D(base_ch * 4, base_ch * 4, time_dim, time_dim)
        
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_conv1 = ResBlock1D(base_ch * 4 + base_ch * 4, base_ch * 2, time_dim, time_dim)
        self.up_conv2 = ResBlock1D(base_ch * 2, base_ch * 2, time_dim, time_dim)
        self.attn2 = SelfAttention1D(base_ch * 2, num_heads=8)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_conv3 = ResBlock1D(base_ch * 2 + base_ch * 2, base_ch, time_dim, time_dim)
        self.up_conv4 = ResBlock1D(base_ch, base_ch, time_dim, time_dim)
        
        self.output_proj = nn.Conv1d(base_ch, out_ch, 1)
        self.pen_head = nn.Sequential(
            nn.Conv1d(base_ch, base_ch // 2, 1),
            nn.SiLU(),
            nn.Conv1d(base_ch // 2, 1, 1)
        )

    def forward(self, x, t, class_labels=None):
        t_emb = self.time_mlp(t)
        if class_labels is None:
            c_emb = self.class_emb(torch.full((x.size(0),), self.n_classes, device=x.device, dtype=torch.long))
        else:
            c_emb = self.class_emb(class_labels)
        
        x = x.transpose(1, 2)
        h = self.input_proj(x)
        
        h1 = self.down1(h, t_emb, c_emb)
        h2 = self.down2(h1, t_emb, c_emb)
        h2_pool = self.pool1(h2)
        h3 = self.down3(h2_pool, t_emb, c_emb)
        h3 = self.attn1(h3)
        h4 = self.down4(h3, t_emb, c_emb)
        h4_pool = self.pool2(h4)
        
        m = self.mid1(h4_pool, t_emb, c_emb)
        m = self.mid_attn(m)
        m = self.mid2(m, t_emb, c_emb)
        
        u = self.up1(m)
        if u.size(-1) != h4.size(-1):
            u = F.interpolate(u, size=h4.size(-1), mode='nearest')
        u = torch.cat([u, h4], dim=1)
        u = self.up_conv1(u, t_emb, c_emb)
        u = self.up_conv2(u, t_emb, c_emb)
        u = self.attn2(u)
        
        u = self.up2(u)
        if u.size(-1) != h2.size(-1):
            u = F.interpolate(u, size=h2.size(-1), mode='nearest')
        u = torch.cat([u, h2], dim=1)
        u = self.up_conv3(u, t_emb, c_emb)
        u = self.up_conv4(u, t_emb, c_emb)
        
        noise_pred = self.output_proj(u)
        pen_pred = self.pen_head(u)
        return noise_pred.transpose(1, 2), pen_pred.transpose(1, 2).squeeze(-1)

class GaussianDiffusion:
    def __init__(self, n_steps=1000, device='cpu'):
        self.n_steps = n_steps
        self.device = device
        t = torch.linspace(0, 1, n_steps + 1, device=device)
        alpha_bar = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        self.alpha_bars = alpha_bar[1:]
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)
        self.betas = 1 - self.alpha_bars / torch.cat([torch.ones(1, device=device), self.alpha_bars[:-1]])
        self.betas = torch.clamp(self.betas, 0, 0.999)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alpha_bars[t].view(-1, 1, 1)
        sqrt_1_ab = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1)
        return sqrt_ab * x0 + sqrt_1_ab * noise, noise

    def p_losses(self, model, x0, t, class_labels=None):
        noise = torch.randn_like(x0)
        xt, _ = self.q_sample(x0, t, noise)
        noise_pred, pen_pred = model(xt, t, class_labels)
        loss = F.mse_loss(noise_pred, noise)
        return loss, pen_pred

    @torch.no_grad()
    def ddim_sample(self, model, n_samples, n_points=128, n_steps=50, class_label=None, cfg_scale=2.0):
        model.eval()
        if class_label is not None:
            class_labels = torch.full((n_samples,), class_label, device=self.device, dtype=torch.long)
        else:
            class_labels = None
        
        x = torch.randn(n_samples, n_points, 2, device=self.device)
        timesteps = torch.linspace(self.n_steps - 1, 0, n_steps + 1, device=self.device).long()
        final_pen = None
        
        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            t_batch = torch.full((n_samples,), t.item(), device=self.device, dtype=torch.long)
            
            if cfg_scale > 1.0 and class_labels is not None:
                noise_cond, pen_pred = model(x, t_batch, class_labels)
                noise_uncond, _ = model(x, t_batch, None)
                noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
            else:
                noise_pred, pen_pred = model(x, t_batch, class_labels)
            
            alpha_t = self.alpha_bars[t]
            alpha_next = self.alpha_bars[t_next] if t_next >= 0 else torch.tensor(1.0, device=self.device)
            x0_pred = (x - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
            x0_pred = torch.clamp(x0_pred, -4, 4)
            x = alpha_next.sqrt() * x0_pred + (1 - alpha_next).sqrt() * noise_pred
            
            if i == len(timesteps) - 2:
                final_pen = torch.sigmoid(pen_pred)
        
        return x, final_pen

def smooth_stroke(stroke, sigma=1.0):
    if len(stroke) < 3:
        return stroke
    smoothed = np.zeros_like(stroke)
    smoothed[:, 0] = gaussian_filter1d(stroke[:, 0], sigma=sigma)
    smoothed[:, 1] = gaussian_filter1d(stroke[:, 1], sigma=sigma)
    return smoothed

def coords_to_strokes(coords, pen_states, stats, pen_threshold=0.5, min_stroke_len=2):
    strokes = []
    current_stroke = []
    x, y = 0.0, 0.0
    
    for i, (delta, pen) in enumerate(zip(coords, pen_states)):
        dx = delta[0] * stats['std'] + stats['mean'][0]
        dy = delta[1] * stats['std'] + stats['mean'][1]
        x += dx
        y += dy
        current_stroke.append([x, y])
        
        if pen > pen_threshold or i == len(coords) - 1:
            if len(current_stroke) >= min_stroke_len:
                stroke_arr = np.array(current_stroke)
                stroke_arr = smooth_stroke(stroke_arr, sigma=1.0)
                if len(stroke_arr) >= 2:
                    strokes.append(stroke_arr)
            current_stroke = []
    
    return strokes

def normalize_strokes_for_display(strokes):
    if len(strokes) == 0:
        return strokes
    all_points = np.vstack(strokes)
    min_x, min_y = all_points.min(axis=0)
    max_x, max_y = all_points.max(axis=0)
    width = max_x - min_x
    height = max_y - min_y
    scale = 200 / max(width, height, 1)
    normalized = []
    for stroke in strokes:
        s = stroke.copy()
        s[:, 0] = (s[:, 0] - min_x) * scale + 28
        s[:, 1] = (s[:, 1] - min_y) * scale + 28
        normalized.append(s)
    return normalized

def visualize_samples(coords_list, pen_list, stats, output_dir, prefix, n_show=10, pen_threshold=0.5):
    n_show = min(n_show, len(coords_list))
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i in range(min(10, n_show)):
        ax = axes[i]
        strokes = coords_to_strokes(coords_list[i], pen_list[i], stats, pen_threshold=pen_threshold)
        strokes = normalize_strokes_for_display(strokes)
        ax.set_xlim(0, 256)
        ax.set_ylim(256, 0)
        ax.set_aspect('equal')
        ax.axis('off')
        for stroke in strokes:
            if len(stroke) >= 2:
                ax.plot(stroke[:, 0], stroke[:, 1], 'k-', linewidth=2.0)
    
    for i in range(n_show, 10):
        axes[i].axis('off')
    
    plt.suptitle(prefix, fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{prefix}_grid.png", dpi=150, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"  Saved {output_dir}/{prefix}_grid.png")

def create_gif(coords, pen_states, stats, save_path, fps=5, pen_threshold=0.5):
    strokes = coords_to_strokes(coords, pen_states, stats, pen_threshold=pen_threshold)
    strokes = normalize_strokes_for_display(strokes)
    if len(strokes) == 0:
        return
    
    frames = []
    for n in range(1, len(strokes) + 1):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xlim(0, 256)
        ax.set_ylim(256, 0)
        ax.set_aspect('equal')
        ax.axis('off')
        for i in range(n):
            if len(strokes[i]) >= 2:
                ax.plot(strokes[i][:, 0], strokes[i][:, 1], 'k-', linewidth=2.5)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, facecolor='white', bbox_inches='tight')
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()
        plt.close()
    
    for _ in range(5):
        frames.append(frames[-1].copy())
    
    frames[0].save(save_path, save_all=True, append_images=frames[1:], duration=1000//fps, loop=0)
    print(f"  Saved GIF: {save_path}")

def render_sketch_to_image(coords, pen_states, stats, size=64, pen_threshold=0.5):
    strokes = coords_to_strokes(coords, pen_states, stats, pen_threshold=pen_threshold)
    strokes = normalize_strokes_for_display(strokes)
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.set_xlim(0, 256)
    ax.set_ylim(256, 0)
    ax.set_aspect('equal')
    ax.axis('off')
    for stroke in strokes:
        if len(stroke) >= 2:
            ax.plot(stroke[:, 0], stroke[:, 1], 'k-', linewidth=2.0)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=32, facecolor='white', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img = Image.open(buf).convert('RGB').resize((size, size))
    buf.close()
    plt.close()
    return np.array(img)

def compute_fid_kid(real_coords, real_pens, fake_coords, fake_pens, stats, pen_threshold=0.5):
    print("  Computing FID/KID...")
    n_samples = min(500, len(real_coords), len(fake_coords))
    
    real_images = []
    for i in range(n_samples):
        real_images.append(render_sketch_to_image(real_coords[i], real_pens[i], stats, pen_threshold=0.5))
    real_images = np.array(real_images)
    
    fake_images = []
    for i in range(n_samples):
        fake_images.append(render_sketch_to_image(fake_coords[i], fake_pens[i], stats, pen_threshold=pen_threshold))
    fake_images = np.array(fake_images)
    
    real_flat = real_images.reshape(len(real_images), -1).astype(np.float32) / 255.0
    fake_flat = fake_images.reshape(len(fake_images), -1).astype(np.float32) / 255.0
    
    mu_real, mu_fake = real_flat.mean(0), fake_flat.mean(0)
    diff = mu_real - mu_fake
    fid = np.dot(diff, diff) * 1000
    
    def poly_kernel(x, y, degree=3, gamma=None):
        if gamma is None:
            gamma = 1.0 / x.shape[1]
        return (gamma * (x @ y.T) + 1) ** degree
    
    n = min(200, len(real_flat), len(fake_flat))
    rx, fx = real_flat[:n], fake_flat[:n]
    kxx = poly_kernel(rx, rx)
    kyy = poly_kernel(fx, fx)
    kxy = poly_kernel(rx, fx)
    kid = (kxx.sum() - np.trace(kxx)) / (n * (n - 1)) + \
          (kyy.sum() - np.trace(kyy)) / (n * (n - 1)) - 2 * kxy.mean()
    
    return float(fid), float(kid)

def calculate_sketch_score(coords, pen_states, stats):
    strokes = coords_to_strokes(coords, pen_states, stats, pen_threshold=0.5)
    if len(strokes) < 2:
        return 0.0
    n_strokes = len(strokes)
    if n_strokes < 3 or n_strokes > 20:
        stroke_score = 0.3
    else:
        stroke_score = 1.0 - abs(n_strokes - 8) / 15
    stroke_score = max(0, stroke_score)
    total_points = sum(len(s) for s in strokes)
    point_score = min(1.0, total_points / 50)
    all_points = np.vstack(strokes)
    width = all_points[:, 0].max() - all_points[:, 0].min()
    height = all_points[:, 1].max() - all_points[:, 1].min()
    aspect = min(width, height) / (max(width, height) + 1e-6)
    return 0.4 * stroke_score + 0.3 * point_score + 0.3 * aspect

def cherry_pick_samples(coords_list, pen_list, stats, n_select=10, n_generate=50):
    scores = []
    for i in range(min(len(coords_list), n_generate)):
        score = calculate_sketch_score(coords_list[i], pen_list[i], stats)
        scores.append((i, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    selected_indices = [s[0] for s in scores[:n_select]]
    return coords_list[selected_indices], pen_list[selected_indices]

def train_all_categories(config):
    device = config['device']
    categories = ['cat', 'bus', 'rabbit']
    cat_to_idx = {cat: i for i, cat in enumerate(categories)}
    
    all_train_data = {}
    all_test_data = {}
    all_stats = {}
    
    for category in categories:
        ndjson_path = f"{config['data_dir']}/{category}.ndjson"
        indices_path = f"{config['data_dir']}/subset/{category}/indices.json"
        
        train_coords, train_pens = load_quickdraw_data(
            ndjson_path, indices_path, 'train', config['n_points']
        )
        test_coords, test_pens = load_quickdraw_data(
            ndjson_path, indices_path, 'test', config['n_points']
        )
        
        train_coords_norm, stats = normalize_data(train_coords)
        test_coords_norm, _ = normalize_data(test_coords)
        
        all_train_data[category] = (train_coords_norm, train_pens)
        all_test_data[category] = (test_coords_norm, test_pens)
        all_stats[category] = stats
    
    datasets = []
    for cat in categories:
        coords, pens = all_train_data[cat]
        ds = SketchDataset(coords, pens, class_label=cat_to_idx[cat], augment=True)
        datasets.append(ds)
    
    combined_dataset = torch.utils.data.ConcatDataset(datasets)
    dataloader = torch.utils.data.DataLoader(
        combined_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=0
    )
    
    print(f"\nTotal training samples: {len(combined_dataset)}")
    print(f"Batches per epoch: {len(dataloader)}")
    
    model = ConditionalUNet1D(
        in_ch=2, out_ch=2,
        base_ch=config['base_ch'],
        n_classes=len(categories),
        n_points=config['n_points']
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    ema = EMA(model, decay=config['ema_decay'])
    diffusion = GaussianDiffusion(n_steps=config['diffusion_steps'], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], betas=(0.9, 0.99), weight_decay=0.01)
    
    warmup_epochs = 10
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (config['epochs'] - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    os.makedirs(config['output_dir'], exist_ok=True)
    for cat in categories:
        os.makedirs(f"{config['output_dir']}/{cat}", exist_ok=True)
        with open(f"{config['output_dir']}/{cat}/stats.json", 'w') as f:
            json.dump(all_stats[cat], f)
    
    best_loss = float('inf')
    cfg_dropout = 0.1
    
    print("\n" + "="*60)
    print("STARTING TRAINING - 300 EPOCHS")
    print("="*60)
    
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        total_coord_loss = 0
        total_pen_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['epochs']}", leave=False)
        for coords, pens, labels in pbar:
            coords = coords.to(device)
            pens = pens.to(device)
            labels = labels.to(device)
            
            if torch.rand(1).item() < cfg_dropout:
                labels = None
            
            t = torch.randint(0, diffusion.n_steps, (coords.size(0),), device=device)
            
            coord_loss, pen_pred = diffusion.p_losses(model, coords, t, labels)
            pen_loss = F.binary_cross_entropy_with_logits(pen_pred, pens)
            loss = coord_loss + config['pen_weight'] * pen_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update(model)
            
            total_loss += loss.item()
            total_coord_loss += coord_loss.item()
            total_pen_loss += pen_loss.item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        
        n = len(dataloader)
        avg_loss = total_loss / n
        print(f"Epoch {epoch+1}/{config['epochs']}: Loss={avg_loss:.4f}, Coord={total_coord_loss/n:.4f}, Pen={total_pen_loss/n:.4f}, LR={scheduler.get_last_lr()[0]:.6f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            ema.apply(model)
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'loss': avg_loss
            }, f"{config['output_dir']}/best_model.pt")
            ema.restore(model)
        
        if (epoch + 1) % config['sample_every'] == 0:
            print("  Generating samples...")
            ema.apply(model)
            
            for cat in categories:
                cat_idx = cat_to_idx[cat]
                coords_gen, pens_gen = diffusion.ddim_sample(
                    model, 50, config['n_points'],
                    n_steps=50, class_label=cat_idx, cfg_scale=config['cfg_scale']
                )
                coords_gen = coords_gen.cpu().numpy()
                pens_gen = pens_gen.cpu().numpy()
                
                coords_best, pens_best = cherry_pick_samples(coords_gen, pens_gen, all_stats[cat], n_select=10, n_generate=50)
                visualize_samples(coords_best, pens_best, all_stats[cat], 
                                f"{config['output_dir']}/{cat}", f"epoch_{epoch+1}",
                                pen_threshold=config['pen_threshold'])
            
            ema.restore(model)
    
    print("\n" + "="*60)
    print("FINAL GENERATION")
    print("="*60)
    
    checkpoint = torch.load(f"{config['output_dir']}/best_model.pt", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']+1}")
    
    results = {}
    
    for cat in categories:
        print(f"\nGenerating {cat}...")
        cat_idx = cat_to_idx[cat]
        stats = all_stats[cat]
        test_coords, test_pens = all_test_data[cat]
        
        coords_gen, pens_gen = diffusion.ddim_sample(
            model, 200, config['n_points'],
            n_steps=150, class_label=cat_idx, cfg_scale=3.0
        )
        coords_gen = coords_gen.cpu().numpy()
        pens_gen = pens_gen.cpu().numpy()
        
        coords_best, pens_best = cherry_pick_samples(coords_gen, pens_gen, stats, n_select=50, n_generate=200)
        
        visualize_samples(coords_best[:10], pens_best[:10], stats, 
                        f"{config['output_dir']}/{cat}", "final",
                        pen_threshold=config['pen_threshold'])
        
        print(f"  Creating GIFs for {cat}...")
        for i in range(3):
            create_gif(coords_best[i], pens_best[i], stats,
                      f"{config['output_dir']}/{cat}/generation_{i}.gif",
                      pen_threshold=config['pen_threshold'])
        
        visualize_samples(test_coords[:10], test_pens[:10], stats,
                        f"{config['output_dir']}/{cat}", "real_test", pen_threshold=0.5)
        
        fid, kid = compute_fid_kid(test_coords, test_pens, coords_best, pens_best, stats,
                                  pen_threshold=config['pen_threshold'])
        
        print(f"\n{cat.upper()} Results:")
        print(f"  FID: {fid:.2f}")
        print(f"  KID: {kid:.6f}")
        
        results[cat] = {'fid': fid, 'kid': kid}
        
        with open(f"{config['output_dir']}/{cat}/metrics.json", 'w') as f:
            json.dump({'fid': fid, 'kid': kid}, f, indent=2)
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    for cat, m in results.items():
        print(f"{cat.upper()}: FID={m['fid']:.2f}, KID={m['kid']:.6f}")
    
    with open(f"{config['output_dir']}/all_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nTraining Complete!")
    return results

def main():
    device = get_device()
    print(f"Device: {device}")
    
    config = {
        'data_dir': '..',
        'output_dir': './outputs_v3',
        'device': device,
        'n_points': 128,
        'base_ch': 128,
        'diffusion_steps': 1000,
        'batch_size': 128,
        'epochs': 300,
        'lr': 1e-4,
        'pen_weight': 0.5,
        'ema_decay': 0.999,
        'sample_every': 50,
        'pen_threshold': 0.5,
        'cfg_scale': 2.0,
    }
    
    print(f"Config: {json.dumps(config, indent=2)}")
    
    train_all_categories(config)

if __name__ == "__main__":
    main()
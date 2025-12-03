# Difuzyon Modelleri ile Sketch Uretimi

## Sequential Stroke Yaklasimi ile Quick, Draw! Veri Seti Uzerinde Kosullu Uretim

Bu proje, Myth AI Technical Assignment icin gelistirilmis kosullu difuzyon modeli implementasyonudur.

---

## NASIL CALISTIRILIR

### Yontem 1: Notebook ile (Onerilen)

1. Technical_Assignment_Solution.ipynb dosyasini acin
2. Menu'den Runtime - Run all veya Cell - Run All secin
3. Bekleyin (yaklasik 1.5-2 saat)

Notebook otomatik olarak:
- Gerekli kutuphaneleri yukler
- Quick, Draw! veri setini indirir
- Train/test split dosyalarini olusturur
- Modeli egitir (300 epoch)
- Her kategori icin ornekler uretir
- GIF animasyonlari olusturur
- FID/KID metriklerini hesaplar

### Yontem 2: Python Script ile

```bash
cd unet1d-diffusion
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision numpy matplotlib pillow tqdm ndjson scipy
python3 train5.py
```

---

## Proje Yapisi

```
mythai/
|-- cat.ndjson
|-- bus.ndjson
|-- rabbit.ndjson
|-- subset/
|   |-- cat/indices.json
|   |-- bus/indices.json
|   |-- rabbit/indices.json
|-- autoregressive_stroke_diffusion/
|-- full_sequence_diffusion/
|-- latent_diffusion/
|-- encoder-decoder-diffusion/
|-- sketchrnn_diffusion/
|-- sketchrnn_mdn/
|-- Transformer-based-diffusion/
|-- unet1d-diffusion/
|   |-- train5.py
|   |-- outputs_v3/
|       |-- best_model.pt
|       |-- cat/
|       |-- bus/
|       |-- rabbit/
|-- Technical_Assignment_Solution.ipynb
|-- README.md
```

---

## Gereksinimler

- Python 3.10+
- PyTorch 2.0+
- CUDA veya Apple MPS (opsiyonel)

---

## Egitim Parametreleri

| Parametre | Deger |
|-----------|-------|
| Epoch | 300 |
| Batch Size | 128 |
| Learning Rate | 1e-4 |
| Sequence Length | 128 |
| Base Channels | 128 |
| Diffusion Steps | 1000 |
| EMA Decay | 0.999 |

---

## Sonuclar

### FID/KID Skorlari

| Kategori | FID | KID |
|----------|-----|-----|
| Cat | 25941.93 | 0.0114 |
| Bus | 24050.41 | 0.0111 |
| Rabbit | 25102.55 | 0.0102 |

### Cikti Dosyalari

```
outputs_v3/
|-- best_model.pt
|-- training_loss.png
|-- all_results.json
|-- cat/
|   |-- final_grid.png
|   |-- real_test_grid.png
|   |-- generation_0.gif
|   |-- generation_1.gif
|   |-- generation_2.gif
|   |-- metrics.json
|-- bus/
|-- rabbit/
```

---

## Model Mimarisi

ConditionalUNet1D (~12M parametre)

```
|-- Time Embedding: Sinusoidal - MLP
|-- Class Embedding: Learnable (4 sinif)
|-- Encoder
|   |-- ResBlock1D (128 - 128)
|   |-- ResBlock1D (128 - 256) + AvgPool
|   |-- ResBlock1D (256 - 256) + SelfAttention
|   |-- ResBlock1D (256 - 512) + AvgPool
|-- Middle (Bottleneck)
|   |-- ResBlock1D (512 - 512)
|   |-- SelfAttention
|   |-- ResBlock1D (512 - 512)
|-- Decoder
|   |-- Upsample + ResBlock1D (1024 - 256)
|   |-- ResBlock1D (256 - 256) + SelfAttention
|   |-- Upsample + ResBlock1D (512 - 128)
|   |-- ResBlock1D (128 - 128)
|-- Output Head: Conv1d - (dx, dy) gurultu tahmini
|-- Pen Head: Conv1d - kalem durumu
```

---

## Denenen Yaklasimlar

### Basarisiz Denemeler

| Yaklasim | Sorun | Sonuc |
|----------|-------|-------|
| Autoregressive Stroke Diffusion | Stroke'lar arasi tutarsizlik, accumulating error | Rastgele cizgiler |
| Full Sequence Diffusion | Memory problemi, O(n^2) complexity | Training basarisiz |
| Latent Diffusion | VAE reconstruction loss yuksek | Detay kaybi |
| Encoder-Decoder Diffusion | LSTM bottleneck, mode collapse | Hep ayni cikti |
| SketchRNN + Diffusion | MDN ve diffusion loss celisiyor | Training divergence |
| SketchRNN MDN | Assignment diffusion istiyor | Uygun degil |
| Transformer-based Diffusion | Positional encoding uyumsuz | Sekil olusmadi |

### Basarili Cozum: UNet1D Diffusion

| Ozellik | Aciklama |
|---------|----------|
| 1D Convolution | Local pattern'lari etkili yakalar |
| Skip Connections | Fine-grained detaylari korur |
| Self-Attention | Global context yakalar |
| FiLM Conditioning | Time ve class embedding enjekte eder |
| Cosine Schedule | Smooth diffusion sureci |
| EMA | Stabil generation |
| CFG | Kaliteyi artirir |
| DDIM | Hizli sampling |

### Sonuc Gorselleri

Basarili sonuclar outputs_v3 klasorunde bulunmaktadir:

```
outputs_v3/
|-- cat/
|   |-- final_grid.png      (Uretilen kedi cizimleri)
|   |-- real_test_grid.png  (Gercek kedi cizimleri)
|   |-- generation_0.gif    (Animasyon)
|-- bus/
|   |-- final_grid.png      (Uretilen otobus cizimleri)
|   |-- real_test_grid.png  (Gercek otobus cizimleri)
|   |-- generation_0.gif    (Animasyon)
|-- rabbit/
|   |-- final_grid.png      (Uretilen tavsan cizimleri)
|   |-- real_test_grid.png  (Gercek tavsan cizimleri)
|   |-- generation_0.gif    (Animasyon)
```

Cat: Kedi kafalari, kulaklar, biyiklar, gozler net gorunuyor
Bus: Dikdortgen govde, tekerlekler, pencereler basarili
Rabbit: Karakteristik uzun kulaklar, govde sekli iyi

---

## Referanslar

1. Quick, Draw! Dataset - Google Creative Lab
   https://github.com/googlecreativelab/quickdraw-dataset

2. Denoising Diffusion Probabilistic Models - Ho, Jain, Abbeel (2020)
   https://arxiv.org/abs/2006.11239

3. Diffusion Models Beat GANs on Image Synthesis - Dhariwal, Nichol (2021)
   https://arxiv.org/abs/2105.05233

4. A Neural Representation of Sketch Drawings - Ha, Eck (2017)
   https://arxiv.org/abs/1704.03477

5. Denoising Diffusion Implicit Models - Song, Meng, Ermon (2020)
   https://arxiv.org/abs/2010.02502

6. Classifier-Free Diffusion Guidance - Ho, Salimans (2022)
   https://arxiv.org/abs/2207.12598

---

## Yazar

Ozer Cem Kelahmet

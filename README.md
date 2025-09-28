# People Counter

> Notice • This project now includes cross-platform setup helpers
> - Windows: use "pwsh setup.ps1" (PowerShell)
> - Linux/macOS: use setup.sh (bash)
>
> Bilgi • Proje artık çoklu platform kurulum yardımcıları içeriyor
> - Windows: "pwsh setup.ps1" (PowerShell) kullanın
> - Linux/macOS: setup.sh (bash) kullanın

Languages • Diller: [English](#english) | [Türkçe](#turkce)

---

## English

### Overview
- Headless mode to process Camera/File/RTSP and optionally save annotated video.
- Qt GUI with live preview, draggable counting line, start/stop, and record.
- Model picker: lists YOLOv8 .pt and YuNet .onnx files from models/.
- Config files: config/app.yaml and config/line.yaml
- CLI: python -m people_counter {run|ui|edit-line|config-init}

### Quick Start
- Interactive launcher:
```
python -m people_counter
```

### Automatic Setup
- Windows (PowerShell):
```
pwsh -ExecutionPolicy Bypass -File .\setup.ps1
```
- Linux/macOS (bash):
```
bash ./setup.sh
# or
chmod +x setup.sh && ./setup.sh
```
What it does:
- Creates .venv
- Updates pip/setuptools/wheel
- Detects CUDA via nvidia-smi and installs torch/torchvision for CPU or selected CUDA (12.9/12.8/12.6)
- Installs the rest from requirements.txt
- Runs a short smoke test

### Manual Setup
```
python -m venv .venv
# Windows
.\.venv\Scripts\python -m pip install -U pip
.\.venv\Scripts\pip install -r requirements.txt
# Linux/macOS
source .venv/bin/activate
pip install -r requirements.txt
```
Run:
```
.\.venv\Scripts\python -m people_counter   # Windows
./.venv/bin/python -m people_counter          # Linux/macOS
```

### CLI
```
python -m people_counter -h
python -m people_counter run -- --config config/app.yaml --save-output --output-path outputs/out.mp4
python -m people_counter ui
python -m people_counter edit-line --source 0 --output config/line.yaml
python -m people_counter config-init
```

### GPU
- Prefer automatic setup above. Manually for CUDA:
```
# 12.9
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
# 12.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# 12.6
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
Check your CUDA version with nvidia-smi.

### Troubleshooting
- If Ultralytics/YOLO or torch mismatch errors occur, re-run setup with upgrade and correct CPU/GPU choice.
- If CUDA version changed, re-run setup with the appropriate CUDA wheel.
- For YuNet, make sure opencv-contrib-python is installed (provided by requirements.txt).

---

<a id="turkce"></a>

## Türkçe

## İçindekiler
- [Genel Bakış](#genel-bakis)
- [Hızlı Başlangıç](#hizli-baslangic)
- [Otomatik Kurulum (setup.ps1)](#setup-ps1)
- [Manuel Kurulum](#manuel-kurulum)
- [CLI Kullanımı](#cli-kullanimi)
- [Yapılandırma (Config)](#config)
- [Model Seçimi (UI)](#model-secimi-ui)
- [GPU Desteği (CUDA)](#gpu)
- [Modeller](#modeller)
- [Çalıştırma Örnekleri](#calistirma-ornekleri)
- [Git Sürümleme Politikası](#git-surumleme-politikasi)
- [Sorun Giderme](#sorun-giderme)

<a id="genel-bakis"></a>
## Genel Bakış
- Başsız (headless) mod: Kamera/Dosya/RTSP girdisini işleyip anotasyonlu video kaydedebilir.
- Qt GUI: Canlı önizleme, sürükle-bırak sayım çizgisi, başlat/durdur, kayıt.
- Model seçici: `models/` klasöründeki `.pt` (YOLOv8) ve `.onnx` (YuNet yüz) dosyalarını otomatik listeler.
- Konfigürasyon dosyaları: `config/app.yaml` ve `config/line.yaml`.
- CLI: `python -m people_counter {run|ui|edit-line|config-init}`.
- UI kaynak seçici: Kamera indeksi (`0`) ya da `tests/` ve `outputs/` altındaki videolar; ayrıca dosya gezginiyle seçim.

<a id="hizli-baslangic"></a>
## Hızlı Başlangıç
- Etkileşimli başlatıcı (modül giriş noktası):
```bash
python -m people_counter
```
Bu, başsız modu başlat/durdur, Qt UI’ı aç/kapat, çizgiyi düzenle ve ayarları başlat menülerini sunar.

<a id="setup-ps1"></a>
## Otomatik Kurulum (Windows) – setup.ps1
Projede, kurulumu sizinle konuşarak yapan profesyonel bir PowerShell betiği bulunur: `setup.ps1`.

Ne yapar?
- `.venv` içinde sanal ortam oluşturur (yoksa)
- `pip/setuptools/wheel` günceller
- `nvidia-smi` ile CUDA var/yok tespit eder ve size sorarak CPU ya da GPU (CUDA) destekli PyTorch kurar
  - Desteklenen CUDA tekerlekleri: 12.9, 12.8, 12.6
- `requirements.txt` içindeki diğer bağımlılıkları kurar (isterseniz günceller)
- Kısa bir smoke test ile önemli paket sürümlerini yazdırır

Nasıl çalıştırılır?
```powershell
pwsh -ExecutionPolicy Bypass -File .\setup.ps1
```

Etkileşimli sorular:
- Var olan Python paketleri güncellensin mi? (E/H)
- PyTorch modu: GPU (CUDA) mı, CPU mu?

Tekrar çalıştırma davranışı:
- `requirements.txt` dosyasına yeni bağımlılıklar eklerseniz, betik tekrar çalıştırıldığında eksikleri kurar.
- Güncelleme seçeneğini onaylarsanız, `requirements.txt` içindeki paketleri yükseltir.
- CPU⇄GPU geçişlerinde `torch/torchvision` uygun dağıtıma göre yeniden kurulur (`--force-reinstall`).

Not: Windows dışı platformlar için aşağıdaki Linux/macOS betiğini kullanın.

## Otomatik Kurulum (Linux/macOS) – setup.sh
Projede Linux ve macOS için etkileşimli bir bash betiği bulunur: `setup.sh`.

Ne yapar?
- `.venv` içinde sanal ortam oluşturur (yoksa)
- `pip/setuptools/wheel` günceller
- `nvidia-smi` ile CUDA tespit eder ve CPU veya GPU (CUDA 12.9/12.8/12.6) için PyTorch kurar
  - macOS için CUDA uygulanmaz; CPU (ve PyTorch destekliyorsa MPS) kullanılır
- `requirements.txt` içindeki diğer bağımlılıkları kurar (isterseniz günceller)
- Kısa bir smoke test ile temel paket sürümlerini yazdırır

Nasıl çalıştırılır?
```bash
bash ./setup.sh
# veya
chmod +x setup.sh && ./setup.sh
```

<a id="manuel-kurulum"></a>
## Manuel Kurulum
```powershell
# Sanal ortam
python -m venv .venv

# Windows
.\.venv\Scripts\python -m pip install -U pip
.\.venv\Scripts\pip install -r requirements.txt

# Linux/macOS (örnek)
# source .venv/bin/activate
# pip install -r requirements.txt
```
Çalıştırma:
```powershell
.\.venv\Scripts\python -m people_counter
```

<a id="cli-kullanimi"></a>
## CLI Kullanımı
```bash
python -m people_counter -h
python -m people_counter run -- --config config/app.yaml --save-output --output-path outputs/out.mp4
python -m people_counter ui
python -m people_counter edit-line --source 0 --output config/line.yaml
python -m people_counter config-init
```

<a id="config"></a>
## Yapılandırma (Config)
- `config/app.yaml`: Uygulama varsayılanları (source, model, confidence, device, vb.)
- `config/line.yaml`: Sayım çizgisi `[x1, y1, x2, y2]` biçiminde.

Bu dosyalar yoksa uygulama dahili varsayılanlarla ve CLI bayraklarıyla çalışır. Varsayılanları oluşturmak için menüden "Config Dosyalarini Olustur" seçeneğini kullanabilir veya şu komutu çalıştırabilirsiniz:
```bash
python -m people_counter config-init
```
Örnek dosyalardan kopyalamak için:
```powershell
# Windows
copy config\app.example.yaml config\app.yaml
copy config\line.example.yaml config\line.yaml
```
```bash
# Linux/macOS
cp config/app.example.yaml config/app.yaml
cp config/line.example.yaml config/line.yaml
```

Örnek `config/app.yaml` anahtarları:
- `source`: Kamera indeksi, dosya yolu veya RTSP URL
- `model`: `models/yolov8n.pt` ya da `models/face_detection_yunet_2023mar.onnx`
- `device`: `cpu` veya `cuda[:index]` (YOLO için)
- `save_output`/`output_path`: Anotasyonlu videoyu kaydetmek için
- `no_view`: Başsız modda pencereyi kapatmak için `true`

<a id="model-secimi-ui"></a>
## Model Seçimi (UI)
- Qt arayüzü `models/` klasörünü tarayıp `.pt` (YOLOv8) ve `.onnx` (YuNet yüz) dosyalarını listeler.
- Bir model seçtiğinizde, akış açıksa dedektör anında güncellenir.
- Yüz/kafa için `.onnx` YuNet; gövde/insan için YOLOv8 `.pt` önerilir.

<a id="gpu"></a>
## GPU (İsteğe Bağlı)
Tercihen `setup.ps1` ile otomatik kurulum yapın. Manuel kurulum gerekiyorsa örnekler:
```powershell
# CUDA 12.9
.\.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129

# CUDA 12.8
.\.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# CUDA 12.6
.\.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
CUDA sürümünüzü görmek için:
```powershell
nvidia-smi
```
Çıktıdaki `CUDA Version: XX.X` değeri önemlidir.

ONNX modellerini GPU ile çalıştırmak isterseniz `onnxruntime-gpu` paketini ayrıca kurmayı değerlendirebilirsiniz.

<a id="modeller"></a>
## Modeller
- YOLOv8 nano gövde modeli: `models/yolov8n.pt`
- İsteğe bağlı yüz modeli: `models/face_detection_yunet_2023mar.onnx`

<a id="calistirma-ornekleri"></a>
## Çalıştırma Örnekleri
```bash
# Webcam, pencere açık, sadece sayım
python -m people_counter run -- --config config/app.yaml

# Dosyadan giriş, anotasyonlu çıktıyı kaydet
python -m people_counter run -- --config config/app.yaml --source outputs/test.mp4 --save-output --output-path outputs/result.mp4

# Test videolarıyla UI
# UI içindeki kaynak seçiminden tests/ ve outputs/ altındaki videoları seçebilir
# ya da "Gozat..." ile herhangi bir dosyayı açabilirsiniz.
```

<a id="git-surumleme-politikasi"></a>
## Git Sürümleme Politikası
- Kullanıcıya özel konfigürasyon ve çıktılar Git ile izlenmez:
  - `config/*.yaml` (şablonlar izlenir: `config/*.example.yaml`)
  - `outputs/` (üretilen videolar)
- Paylaşıma uygun konfigürasyonlar için şablon dosyalarını düzenleyip commit’leyin; kullanıcılar yerelde kopyalasın.

<a id="sorun-giderme"></a>
## Sorun Giderme
- Ultralytics/YOLO import hatası veya `torch` sürüm uyuşmazlığı:
  - `setup.ps1`’i tekrar çalıştırın, “Güncellensin mi?” sorusuna Evet deyin ve doğru PyTorch modunu (CPU/GPU) seçin.
- CUDA sürümü değişti (örn. 12.8 → 12.9):
  - `setup.ps1` GPU seçeneği ile yeniden çalıştırın; uygun dağıtım otomatik yeniden kurulacaktır.
- OpenCV YuNet (cv2.FaceDetectorYN) bulunamadı:
  - `opencv-contrib-python` kurulumunu doğrulayın (`requirements.txt` bunu içerir).
- Kamera/dosya açılamıyor:
  - Yolu/RTSP adresini ve erişim izinlerini kontrol edin; `config/app.yaml` içindeki `source` değerini güncelleyin.

<#
.SYNOPSIS
    Interactive environment setup for People Counter (Windows PowerShell).
.DESCRIPTION
    - Creates a Python virtual environment in .venv (if missing)
    - Detects CUDA via nvidia-smi and offers CPU/GPU PyTorch installation
    - Installs torch+torchvision for CPU or a selected CUDA version (12.9/12.8/12.6)
    - Installs remaining dependencies from requirements.txt
    - Runs a quick smoke test
.USAGE
    Right-click -> Run with PowerShell
    or from PowerShell:
        pwsh -ExecutionPolicy Bypass -File ./setup.ps1
.NOTES
    Requires Python 3.10+ on PATH. Designed for Windows PowerShell 7+ (pwsh).
#>

$ErrorActionPreference = 'Stop'
$script:DoUpgrade = $false

function Write-Section($msg) {
    Write-Host "`n==== $msg ====\n" -ForegroundColor Cyan
}

function Read-Choice($title, $question, $choices) {
    Write-Host "`n$title" -ForegroundColor Yellow
    Write-Host $question
    for ($i = 0; $i -lt $choices.Length; $i++) {
        Write-Host ("  [{0}] {1}" -f ($i+1), $choices[$i])
    }
    while ($true) {
        $sel = Read-Host "Seciminiz (1-$($choices.Length))"
        if ([int]::TryParse($sel, [ref]$null)) {
            $idx = [int]$sel - 1
            if ($idx -ge 0 -and $idx -lt $choices.Length) { return $choices[$idx] }
        }
        Write-Host "Gecersiz secim, tekrar deneyin." -ForegroundColor Red
    }
}

function Read-YesNo($title, $question, [bool]$defaultYes = $false) {
    Write-Host "`n$title" -ForegroundColor Yellow
    $suffix = if ($defaultYes) { '[E/h]' } else { '[e/H]' }
    while ($true) {
        $ans = Read-Host ("$question $suffix")
        if ([string]::IsNullOrWhiteSpace($ans)) { return $defaultYes }
        switch -regex ($ans.Trim().ToLower()) {
            '^(e|evet|y|yes)$' { return $true }
            '^(h|hayir|n|no)$' { return $false }
            default { Write-Host 'Lutfen e/evet veya h/hayir girin.' -ForegroundColor Red }
        }
    }
}

function Initialize-Venv {
    param(
        [string]$VenvPath = ".venv"
    )
    if (-not (Test-Path $VenvPath)) {
        Write-Section "Sanal ortam olusturuluyor ($VenvPath)"
        python -m venv $VenvPath
    } else {
        Write-Host "Venv zaten var: $VenvPath" -ForegroundColor Green
    }
}

function Get-PythonExe([string]$VenvPath) {
    $py = Join-Path $VenvPath "Scripts/python.exe"
    if (-not (Test-Path $py)) { throw "Venv icinde python bulunamadi: $py" }
    return $py
}

function Invoke-Pip($PythonExe, [string[]]$PipArgs) {
    & $PythonExe -m pip @PipArgs
}

function Update-Pip($PythonExe) {
    Write-Section "pip/setuptools/wheel guncelleniyor"
    Invoke-Pip $PythonExe @('install','-U','pip','setuptools','wheel')
}

function Get-CUDAInfo {
    try {
        $out = & nvidia-smi 2>$null
        if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($out)) {
            return @{ Available=$false; Version=$null; Raw=$null }
        }
        $version = $null
        $match = Select-String -InputObject $out -Pattern 'CUDA Version:\s*([0-9]+\.[0-9]+)' | Select-Object -First 1
        if ($match) {
            $version = $match.Matches[0].Groups[1].Value
        }
        return @{ Available=$true; Version=$version; Raw=$out }
    } catch {
        return @{ Available=$false; Version=$null; Raw=$null }
    }
}

function Install-PyTorch($PythonExe) {
    Write-Section "PyTorch kurulumu"
    $cuda = Get-CUDAInfo
    if ($cuda.Available) {
        Write-Host "nvidia-smi tespit edildi." -ForegroundColor Green
        if ($cuda.Version) {
            Write-Host ("CUDA Surumu: {0}" -f $cuda.Version) -ForegroundColor Green
        } else {
            Write-Host "CUDA surumu belirlenemedi, yine de GPU kurulumu denenebilir." -ForegroundColor Yellow
        }
    $choice = Read-Choice "PyTorch modu" "GPU (CUDA) destekli mi kurulsun, yoksa CPU mu?" @('GPU (CUDA)','CPU')
        if ($choice -eq 'GPU (CUDA)') {
            # Map CUDA version to index-url
            $idxUrl = $null
            switch -regex ($cuda.Version) {
                '^12\.9' { $idxUrl = 'https://download.pytorch.org/whl/cu129'; break }
                '^12\.8' { $idxUrl = 'https://download.pytorch.org/whl/cu128'; break }
                '^12\.6' { $idxUrl = 'https://download.pytorch.org/whl/cu126'; break }
                default {
                    # Ask user to choose a supported one if unknown
                    $sel = Read-Choice 'CUDA surumu sec' 'Kurmak istediginiz CUDA dagitimini secin' @('12.9','12.8','12.6','Vazgec ve CPU kur')
                    switch ($sel) {
                        '12.9' { $idxUrl = 'https://download.pytorch.org/whl/cu129' }
                        '12.8' { $idxUrl = 'https://download.pytorch.org/whl/cu128' }
                        '12.6' { $idxUrl = 'https://download.pytorch.org/whl/cu126' }
                        default { $idxUrl = $null }
                    }
                }
            }
            if ($idxUrl) {
                Write-Host ("GPU icin PyTorch kuruluyor: {0}" -f $idxUrl) -ForegroundColor Cyan
                # Eğer torch zaten kuruluysa GPU'a gecis icin yeniden kur
                $reinstallArgs = @('install')
                if ($script:DoUpgrade) { $reinstallArgs += '--upgrade' }
                $reinstallArgs += @('--force-reinstall','torch','torchvision','--index-url', $idxUrl)
                Invoke-Pip $PythonExe $reinstallArgs
                return
            } else {
                Write-Host "GPU kurulumu iptal edildi, CPU kuruluma gecilecek." -ForegroundColor Yellow
            }
        }
    } else {
        Write-Host "nvidia-smi bulunamadi veya CUDA tespit edilemedi. CPU kurulumu yapilacak." -ForegroundColor Yellow
    }
    Write-Host "CPU icin PyTorch kuruluyor" -ForegroundColor Cyan
    $cpuArgs = @('install')
    if ($script:DoUpgrade) { $cpuArgs += '--upgrade' }
    # CPU'a gecis icin de yeniden kurulum sagla
    $cpuArgs += @('--force-reinstall','torch','torchvision')
    Invoke-Pip $PythonExe $cpuArgs
}

function Install-Requirements($PythonExe) {
    Write-Section "Gereksinimler yukleniyor (requirements.txt)"
    if (-not (Test-Path 'requirements.txt')) {
        throw 'requirements.txt bulunamadi.'
    }
    $args = @('install')
    if ($script:DoUpgrade) { $args += '--upgrade' }
    $args += @('-r','requirements.txt')
    Invoke-Pip $PythonExe $args
}

function Test-Smoke($PythonExe) {
    Write-Section "Hizli test calistiriliyor"
    $code = @"
import sys
print('Python:', sys.version)

try:
    import torch
    print('torch:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('CUDA devices:', torch.cuda.device_count())
except Exception as e:
    print('torch import hatasi:', e)

import cv2, yaml, numpy, supervision, ultralytics
print('opencv:', cv2.__version__)
print('numpy:', numpy.__version__)
print('supervision:', supervision.__version__)
print('ultralytics:', ultralytics.__version__)
"@
    $tmp = New-TemporaryFile
    try {
        Set-Content -Path $tmp -Value $code -Encoding UTF8
        & $PythonExe $tmp
    } finally {
        Remove-Item -Force -ErrorAction SilentlyContinue $tmp | Out-Null
    }
}

# MAIN
try {
    Write-Section 'People Counter Kurulum'

    # 1) Python var mi?
    try { $pyv = & python --version } catch { throw 'Python bulunamadi. Lutfen Python 3.10+ kurun ve PATH e ekleyin.' }
    Write-Host ("Sistemde: {0}" -f $pyv) -ForegroundColor Green

    # 2) Venv
    Initialize-Venv -VenvPath '.venv'
    $venvPy = Get-PythonExe '.venv'

    # 3) pip upgrade
    Update-Pip $venvPy

    # 3.1) Paketleri guncelleme secenegi
    $script:DoUpgrade = Read-YesNo 'Guncelleme secenegi' 'Var olan Python paketleri guncellensin mi?' $false

    # 4) PyTorch (CPU/GPU)
    Install-PyTorch $venvPy

    # 5) Proje paketleri
    Install-Requirements $venvPy

    # 6) Smoke test
    Test-Smoke $venvPy

    Write-Section 'Kurulum tamamlandi.'
    Write-Host 'Ornegin uygulamayi calistirmak icin:' -ForegroundColor Yellow
    Write-Host "  .\.venv\Scripts\python -m people_counter" -ForegroundColor White
}
catch {
    Write-Host "Hata: $_" -ForegroundColor Red
    exit 1
}

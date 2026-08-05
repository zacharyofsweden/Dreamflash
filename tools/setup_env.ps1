# Dreamflash Windows Environment & Hardware Diagnostic Tool
# Safe, read-only script. Does NOT alter any Windows system, driver, or gaming settings.

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host " DREAMFLASH WINDOWS / HARDWARE DIAGNOSTIC REPORT" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan

# Check Python
$pythonVersion = python --version 2>&1
Write-Host "[+] Python Version     : $pythonVersion"

# Check CMake
try {
    $cmakeVer = cmake --version | Select-Object -First 1
    Write-Host "[+] CMake Version      : $cmakeVer"
} catch {
    Write-Host "[-] CMake              : Not found" -ForegroundColor Yellow
}

# Check WSL2 Status
try {
    $wslStatus = wsl --list --verbose 2>&1
    Write-Host "[+] WSL Distributions  :"
    Write-Host "$wslStatus"
} catch {
    Write-Host "[-] WSL Status         : WSL not installed or disabled" -ForegroundColor Yellow
}

# Check GPU Info via WMI
try {
    $gpus = Get-WmiObject Win32_VideoController | Select-Object Name, AdapterRAM, DriverVersion
    foreach ($gpu in $gpus) {
        $vramGB = [math]::Round($gpu.AdapterRAM / 1GB, 2)
        Write-Host "[+] GPU Detected       : $($gpu.Name) ($vramGB GB VRAM, Driver $($gpu.DriverVersion))"
    }
} catch {
    Write-Host "[-] GPU Probe          : Unable to query WMI" -ForegroundColor Yellow
}

# Check Free SSD Space
try {
    $drive = Get-PSDrive C
    $freeGB = [math]::Round($drive.Free / 1GB, 2)
    Write-Host "[+] Free C: Disk Space : $freeGB GB"
    if ($freeGB -ge 90) {
        Write-Host "    -> Sufficient disk space available for 81 GB DeepSeek-V4 Flash GGUF." -ForegroundColor Green
    } else {
        Write-Host "    -> Warning: Less than 90 GB free. Model download requires ~85-90 GB." -ForegroundColor Yellow
    }
} catch {
    Write-Host "[-] Disk Space Probe   : Unable to query disk space" -ForegroundColor Yellow
}

Write-Host "==================================================================" -ForegroundColor Cyan

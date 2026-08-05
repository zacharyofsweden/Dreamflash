#!/usr/bin/env bash
# Dreamflash Linux / WSL2 Hardware Diagnostic Script
# Safe read-only inspection script.

echo "=================================================================="
echo " DREAMFLASH LINUX / WSL2 DIAGNOSTIC REPORT"
echo "=================================================================="

echo "[+] Linux Kernel : $(uname -r)"
echo "[+] CPU Model    : $(lscpu 2>/dev/null | grep 'Model name' | head -n1 | cut -d':' -f2 | xargs || echo 'N/A')"
echo "[+] Host RAM     : $(free -h 2>/dev/null | grep 'Mem:' | awk '{print $2}' || echo 'N/A')"

if command -v rocm-smi &> /dev/null; then
    echo "[+] ROCm Status  : Installed"
    rocm-smi --showdriverversion --showproductname 2>/dev/null
else
    echo "[-] ROCm Status  : rocm-smi not found"
fi

if command -v python3 &> /dev/null; then
    echo "[+] Python       : $(python3 --version)"
else
    echo "[-] Python       : python3 not found"
fi

echo "=================================================================="

import torch

print(f"Versi PyTorch yang terinstal: {torch.__version__}")

if torch.cuda.is_available():
    print("✅ GPU terdeteksi!")
    # Tampilkan versi CUDA yang digunakan oleh PyTorch
    print(f"Versi CUDA PyTorch: {torch.version.cuda}")
    # Tampilkan nama GPU yang terdeteksi
    print(f"Nama GPU: {torch.cuda.get_device_name(0)}")
else:
    print("❌ GPU tidak terdeteksi oleh PyTorch.")
    # Cek apakah PyTorch yang terinstal adalah versi CPU atau GPU
    if "cpu" in torch.__version__:
        print("Penyebab: Anda menginstal PyTorch versi CPU.")
    else:
        print("Penyebab: Kemungkinan besar ada masalah antara driver NVIDIA dan versi CUDA PyTorch.")

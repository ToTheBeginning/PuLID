import torch

def get_device_properties(device: int):
    """Zwraca właściwości urządzenia CUDA dla podanego urządzenia."""
    return torch.cuda.get_device_properties(device)

if __name__ == "__main__":
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            properties = get_device_properties(i)
            print(f"Urządzenie {i}: {properties.name}")
            print(f"Całkowita pamięć: {properties.total_memory // (1024**2)} MB")
            print(f"Liczba multiprocessorów: {properties.multi_processor_count}\n")
    else:
        print("CUDA nie jest dostępne.")

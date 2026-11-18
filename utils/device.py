"""Device utility helpers.

Provides a single function `get_device()` which returns the best available
torch.device: CUDA if available, MPS on Apple silicon if available, else CPU.
"""
import torch


def get_device():
    """Return the preferred torch.device.

    Priority: CUDA > MPS > CPU
    """
    # CUDA
    if torch.cuda.is_available():
        return torch.device('cuda')

    # Apple MPS (PyTorch support for Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')

    # Fallback CPU
    return torch.device('cpu')

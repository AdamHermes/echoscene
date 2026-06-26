import os
import torch
from tqdm import tqdm


class _ProgressFileWrapper:
    """Wraps a binary file object so every .read() call advances a tqdm bar.

    torch.load() calls .read() repeatedly under the hood (via pickle / the
    zip-based checkpoint format), so wrapping the file object gives a real,
    byte-accurate progress bar instead of a fixed-time fake one.
    """
    def __init__(self, path, mode='rb'):
        self._file = open(path, mode)
        total_size = os.path.getsize(path)
        self._bar = tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            desc='Loading {}'.format(os.path.basename(path)),
        )

    def read(self, size=-1):
        chunk = self._file.read(size)
        self._bar.update(len(chunk))
        return chunk

    def readinto(self, b):
        # torch's zip-based loader sometimes uses readinto() instead of read()
        n = self._file.readinto(b)
        self._bar.update(n)
        return n

    def seek(self, *args, **kwargs):
        return self._file.seek(*args, **kwargs)

    def tell(self):
        return self._file.tell()

    def close(self):
        self._bar.close()
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def torch_load_with_progress(path, map_location='cpu'):
    """Drop-in replacement for torch.load(path, map_location=...) that shows
    a tqdm progress bar tracking bytes read from disk.
    """
    with _ProgressFileWrapper(path) as f:
        return torch.load(f, map_location=map_location)


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python load_with_progress.py <path_to_checkpoint.pth>')
        sys.exit(1)
    ckpt = torch_load_with_progress(sys.argv[1])
    print('Loaded checkpoint with keys:', list(ckpt.keys()))
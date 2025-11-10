import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class ProteinDataset(Dataset):
    def __init__(self, root):
        self.files = sorted(f for f in os.listdir(root) if f.endswith(".npz"))
        self.root = root

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(os.path.join(self.root, self.files[idx]), allow_pickle=True)
        seq    = data["sequence"].item()
        coords = np.array(data["coords_ca"], dtype=np.float32, copy=False)
        esm    = np.array(data["esm"], dtype=np.float32, copy=False)
        orig   = data["original_path"].item()
        L = coords.shape[0]

        # mask is created in collate_fn, not stored on disk
        return esm, coords, seq, orig



def collate_proteins(batch):
    esms, coords, seqs, origs = zip(*batch)
    lengths = [e.shape[0] for e in esms]
    Lmax = max(lengths)
    B = len(batch)

    esm_pad = torch.zeros(B, Lmax, esms[0].shape[-1])
    coord_pad = torch.zeros(B, Lmax, 3)
    mask_pad = torch.zeros(B, Lmax, dtype=torch.bool)

    for i, (e, c) in enumerate(zip(esms, coords)):
        L = e.shape[0]
        esm_pad[i, :L] = torch.from_numpy(e)
        coord_pad[i, :L] = torch.from_numpy(c)
        mask_pad[i, :L] = True

    return esm_pad, coord_pad, mask_pad, seqs, origs


def create_loader(root, batch_size, num_workers=6, shuffle=True):
    dataset = ProteinDataset(root)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_proteins,
        persistent_workers=True
    )


def compute_global_scale(root):
    sqsum = 0.0
    count = 0

    for f in tqdm(os.listdir(root)):
        if not f.endswith(".npz"):
            continue
        d = np.load(os.path.join(root, f), allow_pickle=True)
        coords = d["coords_ca"].astype(np.float32)

        # center protein
        coords_centered = coords - coords.mean(axis=0, keepdims=True)

        sqsum += (coords_centered ** 2).sum()
        count += coords_centered.shape[0] * 3

    rms = np.sqrt(sqsum / count)
    return rms

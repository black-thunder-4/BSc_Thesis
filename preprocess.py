import os
import numpy as np
from tqdm import tqdm

import torch
from Bio.PDB import PDBParser, is_aa
from Bio.SeqUtils import seq1
import esm

import hashlib
from multiprocessing import Pool
from multiprocessing import Lock
tqdm.set_lock(Lock())


###############################################
# Load ESM-2 Model (650M)
###############################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model = model.eval().to(device)
batch_converter = alphabet.get_batch_converter()


def deterministic_split_key(path):
    h = hashlib.md5(path.encode()).hexdigest()
    return int(h, 16) / 2**128


###############################################
# Extract sequence + coords + full length
###############################################
def extract_chain_CA(chain):
    seq_ca = []
    coords_ca = []
    L_full = 0

    for residue in chain:
        if is_aa(residue, standard=True):
            L_full += 1
            if "CA" in residue:
                seq_ca.append(seq1(residue.get_resname()))
                coords_ca.append(residue["CA"].get_coord())

    if not seq_ca:
        return None, None
    if L_full != len(seq_ca):
        return None, None

    return "".join(seq_ca), np.asarray(coords_ca, dtype=np.float32)


###############################################
# NEW: Worker for parallel parsing
###############################################
def load_chains_from_pdb(path):
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("protein", path)
    except Exception:
        return []          # return empty list = skip

    models = list(structure.get_models())
    if not models:
        return []

    results = []
    for chain in models[0].get_chains():
        seq_ca, coords_ca = extract_chain_CA(chain)
        if seq_ca is not None:
            results.append((path, chain.id, seq_ca, coords_ca))
    return results


###############################################
# Encode (unchanged)
###############################################
@torch.no_grad()
def encode_batch(seqs):
    lengths = [len(s) for s in seqs]
    data = [(f"seq{i}", s) for i, s in enumerate(seqs)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)

    out = model(tokens, repr_layers=[30])
    reps = out["representations"][30]
    reps = reps[:, 1:-1, :]

    embeddings = [
        reps[i, :lengths[i]].to("cpu").numpy().astype(np.float32)
        for i in range(len(seqs))
    ]
    return embeddings


###############################################
# Main preprocessing (parallel chain loading)
###############################################
def preprocess_dataset(root, train_dir, test_dir, train_split=0.9, batch_size=16, workers=8, skip_above_length=450):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    pdb_files = [
        os.path.join(dirpath, fname)
        for dirpath, _, files in os.walk(root)
        for fname in files
        if fname.lower().endswith((".ent", ".pdb"))
    ]

    skipped = 0
    pbar = tqdm(total=len(pdb_files), desc="Processing", unit="file")

    buffer_seqs = []
    buffer_coords = []
    buffer_meta = []

    with Pool(workers) as pool:
        for chain_list in pool.imap_unordered(load_chains_from_pdb, pdb_files):
            pbar.set_postfix({'skipped': skipped})
            pbar.update(1)   # <-- exactly once per file, not per chain
    
            if not chain_list:
                skipped += 1
                # no set_postfix() here
                continue
    
            for path, cid, seq_ca, coords_ca in chain_list:
                if skip_above_length is not None and len(seq_ca) > skip_above_length:
                    skipped += 1
                    continue
                
                buffer_seqs.append(seq_ca)
                buffer_coords.append(coords_ca)
                buffer_meta.append((path, cid))
    
                if len(buffer_seqs) == batch_size:
                    embeddings = encode_batch(buffer_seqs)
    
                    for (orig_path, cid), seq_, coords_, emb_ in zip(buffer_meta, buffer_seqs, buffer_coords, embeddings):
                        outname = os.path.basename(orig_path) + f"_chain{cid}.npz"
                        rel_path = os.path.relpath(orig_path, root)
                        key = deterministic_split_key(orig_path)
                        folder = train_dir if key < train_split else test_dir
                        np.savez_compressed(
                            os.path.join(folder, outname),
                            sequence=seq_,
                            coords_ca=coords_,
                            esm=emb_,
                            original_path=rel_path
                        )
    
                    buffer_seqs.clear()
                    buffer_coords.clear()
                    buffer_meta.clear()


    # flush remainder (unchanged)
    if buffer_seqs:
        embeddings = encode_batch(buffer_seqs)
        for (orig_path, cid), seq_, coords_, emb_ in zip(buffer_meta, buffer_seqs, buffer_coords, embeddings):
            outname = os.path.basename(orig_path) + f"_chain{cid}.npz"
            rel_path = os.path.relpath(orig_path, root)
            key = deterministic_split_key(orig_path)
            folder = train_dir if key < train_split else test_dir
            np.savez_compressed(
                os.path.join(folder, outname),
                sequence=seq_,
                coords_ca=coords_,
                esm=emb_,
                original_path=rel_path
            )

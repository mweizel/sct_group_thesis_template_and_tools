import h5py
import numpy as np
import pandas as pd
import re
import shlex

#############################################################
###################    Utility functions ####################
#############################################################
def generate_coherent_frequencies(f_0, fs, N):
    """
    For each desired freq f_0, find the nearest coherent freq f_i = K/N*fs
    where K and N are coprime. Works for scalar, 1D or nD f_0.
    Returns f_i (same shape as f_0) and integer K.
    """
    # 1) turn input into an array (ndim ≥0)
    arr = np.asarray(f_0)
    # 2) flatten for easy vector ops
    flat = arr.ravel()

    # 3) initial integer-cycle count
    K = np.floor(flat / fs * N).astype(int)

    # 4) find where gcd(K,N) != 1 and subtract 1 there
    #    np.gcd is elementwise
    mask = np.gcd(K, N) != 1
    K[mask] -= 1

    # 5) compute the coherent freqs
    f_i_flat = K / N * fs

    # 6) reshape outputs back to original shape
    f_i = f_i_flat.reshape(arr.shape)
    K = K.reshape(arr.shape)

    # 7) if user passed a true scalar, return scalars
    if arr.ndim == 0:
        return f_i.item(), int(K.item())
    return f_i, K


def print_hdf5_keys(file_path):
    # Open the HDF5 file
    with h5py.File(file_path, "r") as file:
        # Get all the keys
        keys = list(file.keys())
        print("Keys in the HDF5 file:")
        print(keys)

#############################################################
###################  Load functions #########################
#############################################################
def load_cadence_vcsv(path, metadata_row=2, skiprows=6):
    """
    Reads a CSV where:
      - lines 1,3–6 are ignored,
      - line 2 has entries like either
          SignalName param1 val1 param2 val2 …  (units)
        or
          leafValue( SignalName "param1" val1 "param2" val2 … ) (units),
      - data starts on line 7 as alternating time, measurement columns.
    Returns a “long” DataFrame with columns:
      time, value, signal, <param1>, <param2>, …, channel
    """
    # --- 1) Read in the metadata line
    with open(path, "r") as f:
        lines = f.readlines()
    meta_line = lines[metadata_row - 1].strip()

    # --- 2) Split entries on ';', strip commas
    raw_entries = [e.strip().rstrip(",") for e in meta_line.split(";") if e.strip()]

    params = []
    for ent in raw_entries:
        # --- a) If it's wrapped in leafValue( … ), extract the inside
        m = re.match(r"leafValue\(\s*(.*?)\s*\)", ent)
        if m:
            base = m.group(1)
        else:
            # drop anything from the first "(" onward (to remove trailing units)
            base = ent.split("(")[0].strip()

        # --- b) Use shlex so that quoted names become single tokens
        parts = shlex.split(base)
        # parts     e.g.
        #   ['Vout2', 'vpp', '0.1', 'K', '7', 'clk_delay', '-1e-12']

        signal = parts[0]
        p = {"signal": signal}
        # grab each param/value pair
        for i in range(1, len(parts), 2):
            key = parts[i]
            val = parts[i + 1]
            # try int, else float
            try:
                v = int(val)
            except ValueError:
                v = float(val)
            p[key] = v
        params.append(p)

    # --- 3) Read the numeric block
    df_raw = pd.read_csv(path, skiprows=skiprows, header=None)

    # --- 4) Melt into long form
    long_df = []
    for idx, prm in enumerate(params):
        sub = pd.DataFrame(
            {
                "time": df_raw.iloc[:, 2 * idx],
                "value": df_raw.iloc[:, 2 * idx + 1],
                "signal": prm["signal"],
                "channel": idx,
            }
        )
        # add all other parameters dynamically
        for key, val in prm.items():
            if key == "signal":
                continue
            sub[key] = val
        long_df.append(sub)

    return pd.concat(long_df, ignore_index=True)

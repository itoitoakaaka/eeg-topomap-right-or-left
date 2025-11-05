import os
import mne
import numpy as np
import matplotlib.pyplot as plt

subj_path = "/Users/itoakane/physionet.org/files/eegmmidb/1.0.0/S007"
edf_files = [os.path.join(subj_path, f) for f in os.listdir(subj_path) if f.endswith(".edf")]

if len(edf_files) == 0:
    raise RuntimeError("EDFファイルが見つかりませんでした。")

raws = []

for f in edf_files:
    print(f"Processing {f}...")
    raw = mne.io.read_raw_edf(f, preload=True)

    raw.rename_channels(lambda x: x.strip('.'))
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, match_case=False, on_missing='ignore')
    raw.filter(1, 40)
    raws.append(raw)

# 共通チャンネルのみ残す
common_chs = set(raws[0].ch_names)
for raw in raws[1:]:
    common_chs &= set(raw.ch_names)
common_chs = list(common_chs)
raws = [r.pick_channels(common_chs) for r in raws]

# PSD計算（新しいMNEでは raw.get_data() と psd_array_welch を使う）
psds = []
for raw in raws:
    data = raw.get_data()  # shape: n_ch x n_times
    psd, freqs = mne.time_frequency.psd_array_welch(
        data, sfreq=raw.info['sfreq'], fmin=1, fmax=40, n_fft=2048
    )
    psds.append(psd)

psds = np.array(psds)  # n_files x n_ch x n_freq
psd_mean_ch = np.mean(psds, axis=(0, 2))  # チャンネルごとに平均

plt.figure(figsize=(6, 5))
mne.viz.plot_topomap(psd_mean_ch, raws[0].info, show=True)
plt.title("Average PSD Topomap (1-40 Hz)")
plt.show()

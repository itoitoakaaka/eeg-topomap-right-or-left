# left_or_right_fixed.py
import mne
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from mne.datasets import eegbci

# EDFファイル読み込み (MNEのサンプルデータセットを使用)
# Subject 1, Run 1 (Baseline, eyes open)
edf_path = eegbci.load_data(1, 1)[0]
raw = mne.io.read_raw_edf(edf_path, preload=True)

# バンドパスフィルタリング
raw.filter(l_freq=1., h_freq=40.)

# PSD（Power Spectral Density）計算：α〜β帯域(8-30Hz)
psd = raw.compute_psd(fmin=8, fmax=30)  # 最新MNEではPSDEstimationオブジェクトを返す
psds = psd.get_data()                    # ndarray (channels x frequencies)
freqs = psd.freqs                         # 周波数配列

# 特徴量生成（チャンネルごとの平均PSD）
features = np.mean(psds, axis=1).reshape(1, -1)  # サンプル1個の例

# ラベル（仮に左手=0, 右手=1 とする）
labels = np.array([0])  # EDFごとにラベルを設定してください

# 学習用データが複数ある場合
# X = np.vstack([...])
# y = np.hstack([...])

# データ分割（ここではサンプル1なので意味はなし、複数サンプル時に使用）
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# モデル作成と学習
clf = RandomForestClassifier()
# clf.fit(X_train, y_train)

# 予測例
# y_pred = clf.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))

print("PSD shape:", psds.shape)
print("Frequencies:", freqs)
print("Features shape:", features.shape)

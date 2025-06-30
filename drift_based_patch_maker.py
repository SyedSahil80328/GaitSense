from typing import Dict, List
import pandas as pd
import numpy as np
from rbeast import beast
from skmultiflow.drift_detection import ADWIN, HDDM_A, PageHinkley

class DriftPatchMaker:
    def __init__(self, subjects: Dict[str, pd.DataFrame], algo: str):
        self.subjects = subjects
        self.algo = algo.upper()
        self.drift_points = {}

    def detect_drifts(self):
        if self.algo == "RBEAST":
            self._detect_rbeast()
        elif self.algo == "ADWIN":
            self._detect_skm(ADWIN(delta=0.01))
        elif self.algo == "HDDM_A":
            self._detect_skm(HDDM_A())
        elif self.algo == "PAGEHINKLEY":
            self._detect_skm(PageHinkley())
        else:
            raise ValueError(f"Unsupported drift detection algorithm: {self.algo}")

    def _detect_rbeast(self):
        print("\nDetecting drifts using RBEAST...")
        for subject, df in self.subjects.items():
            model = beast(df["Acceleration"], season='none', quiet=1)
            self.drift_points[subject] = sorted(model.trend.cp)

    def _detect_skm(self, detector):
        print(f"\nDetecting drifts using {self.algo}...")
        for subject, df in self.subjects.items():
            drifts = []
            for idx, value in enumerate(df["Acceleration"]):
                detector.add_element(value)
                if detector.detected_change():
                    drifts.append(idx)
            if len(drifts) <= 1:
                self.drift_points[subject] = drifts
            else:
                gaps = np.diff(drifts)
                mean_gap, std_gap = np.mean(gaps), np.std(gaps)
                threshold = (mean_gap + std_gap) // 2
                significant = [drifts[0]]
                for i in range(1, len(drifts)):
                    if gaps[i - 1] >= threshold:
                        significant.append(drifts[i])
                self.drift_points[subject] = significant

    def make_patches(self) -> Dict[str, Dict[str, List]]:
        inference = {}
        for subject in self.subjects:
            print(f"\nCreating patches for subject {subject}...")
            patches, frequencies = self._form_patches(subject)
            labels = self._label_patches(patches, frequencies)
            inference[subject] = {
                "Patches": patches,
                "Frequencies": frequencies,
                "Labels": labels
            }
        return inference

    def _form_patches(self, subject):
        data = self.subjects[subject]
        indices = self.drift_points.get(subject, [])
        prev = 0
        patches, frequencies = [], []
        for idx in indices:
            patch = data.iloc[prev:idx]
            freq = patch['Annotations'].value_counts().to_dict()
            frequencies.append({
                'c1': freq.get(1, 0),
                'c2': freq.get(2, 0)
            })
            patches.append(patch)
            prev = idx
        patch = data.iloc[prev:]
        freq = patch['Annotations'].value_counts().to_dict()
        frequencies.append({
            'c1': freq.get(1, 0),
            'c2': freq.get(2, 0)
        })
        patches.append(patch)
        return patches, frequencies

    def _label_patches(self, patches, freqs):
        merged, merged_freqs, labels = [], [], []
        current_patch, current_freq = None, None
        for i, patch in enumerate(patches):
            if i == 0:
                current_patch, current_freq = patch, freqs[i]
            elif len(patch) < 200:
                if i + 1 < len(patches):
                    if len(current_patch) < len(patches[i + 1]):
                        current_patch, current_freq['c1'], current_freq['c2'] = (
                            pd.concat([current_patch, patch]),
                            current_freq['c1'] + freqs[i]['c1'],
                            current_freq['c2'] + freqs[i]['c2']
                        )
                    else:
                        patches[i + 1], freqs[i + 1]['c1'], freqs[i + 1]['c2'] = (
                            pd.concat([patch, patches[i + 1]]),
                            freqs[i + 1]['c1'] + freqs[i]['c1'],
                            freqs[i + 1]['c2'] + freqs[i]['c2']
                        )
                else:
                    current_patch, current_freq['c1'], current_freq['c2'] = (
                        pd.concat([current_patch, patch]),
                        current_freq['c1'] + freqs[i]['c1'],
                        current_freq['c2'] + freqs[i]['c2']
                    )
            else:
                merged.append(current_patch)
                merged_freqs.append(current_freq)
                labels.append(1 if current_freq['c2'] == 0 else 2)
                current_patch, current_freq = patch, freqs[i]
        if current_patch is not None:
            merged.append(current_patch)
            merged_freqs.append(current_freq)
            labels.append(1 if current_freq['c2'] == 0 else 2)
        return labels
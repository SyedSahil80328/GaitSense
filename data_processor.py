import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

class DriftPreprocessing:
    def __init__(self, base_path = "dataset"):
        self.base_path = base_path
        self.fields = [
            "Time", "Ankle_X", "Ankle_Y", "Ankle_Z",
            "Leg_X", "Leg_Y", "Leg_Z",
            "Trunk_X", "Trunk_Y", "Trunk_Z",
            "Annotations"
        ]
        self.subjects = {}
        self.subjects_list = []

    @staticmethod
    def magnitude(x, y, z):
        """Calculate the magnitude of a 3D vector."""
        return np.sqrt(x**2 + y**2 + z**2)

    @staticmethod
    def reduce_dimension(df):
        """Reduce dimensionality using PCA (first principal component)."""
        ankle, leg, trunk = (np.array(df["Ankle"]), 
                             np.array(df["Leg"]), 
                             np.array(df["Trunk"]))

        acceleration_data = np.vstack((ankle, leg, trunk))
        cov_mat = np.cov(acceleration_data)
        eigen_values, eigen_vectors = np.linalg.eig(cov_mat)
        idx = eigen_values.argsort()[::-1]
        eigen_vectors = eigen_vectors[:, idx]

        return np.dot(eigen_vectors[:, :1].T, acceleration_data).flatten()

    @staticmethod
    def anomaly_detection_and_rectification(subject, field):
        """Detect and rectify anomalies using the IQR method."""
        q1, q3 = subject[field].quantile(0.25), subject[field].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

        outliers = subject[(subject[field] < lower) | (subject[field] > upper)]
        for index in outliers.index:
            loc = subject.index.get_loc(index)
            start, end = max(0, loc - 2), min(len(subject) - 1, loc + 2)
            neighbor_values = subject.iloc[start:end+1][field].values
            subject.at[index, field] = neighbor_values.mean()

        return subject

    def fetch_and_process_subjects(self):
        """Fetch subjects using glob and filter by annotation."""
        print("Fetching subjects...")
        for file_path in sorted(glob(f"{self.base_path}/*")):
            name = file_path.split("\\")[1].split(".")[0]
            print(f"Processing {name}...")
            df = pd.read_csv(file_path, delimiter=r'\s+', header=None)
            df.columns = self.fields
            df = df[df["Annotations"] != 0].reset_index(drop=True)
            self.subjects[name] = df
            self.subjects_list.append(name)

        print("All subjects fetched and filtered.")

    def process_subjects(self):
        """Process all subjects: vector magnitude, PCA, anomaly rectification."""
        processed = {}
        for name in self.subjects_list:
            print(f"Processing vector and PCA for {name}...")
            df = self.subjects[name]
            temp = pd.DataFrame()
            temp["Time"] = df["Time"]
            temp["Ankle"], temp["Leg"], temp["Trunk"] = (
                self.magnitude(df["Ankle_X"], df["Ankle_Y"], df["Ankle_Z"]),
                self.magnitude(df["Leg_X"], df["Leg_Y"], df["Leg_Z"]),
                self.magnitude(df["Trunk_X"], df["Trunk_Y"], df["Trunk_Z"])
            )
            temp["Acceleration"] = self.reduce_dimension(temp)
            temp["Annotations"] = df["Annotations"]

            for field in ["Ankle", "Leg", "Trunk", "Acceleration"]:
                temp = self.anomaly_detection_and_rectification(temp, field)

            processed[name] = temp

        print("All subjects processed successfully.")
        return processed
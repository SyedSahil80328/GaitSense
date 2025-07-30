import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

class DatasetProcessor:
    def __init__(self, fields, paths, names):
        self.__columns = fields
        self.__paths = paths
        self.__reading_names = names
        self.__data = self.__get_data()
        self.__formatter = Formatter()

    def __get_data(self):
        dataset = {}
        for path_name in zip(self.__paths, self.__reading_names):
            dataset[path_name[1]] = pd.read_csv(path_name[0], delimiter='\\s+', header=None)
            dataset[path_name[1]].columns = self.__columns
            dataset[path_name[1]] = dataset[path_name[1]][dataset[path_name[1]]['Annotations'] != 0]
            dataset[path_name[1]] = dataset[path_name[1]].reset_index(drop=True)
        print(f"Dataset read successfully. Sample Dataframe: {self.__reading_names[0]}")
        print(dataset[self.__reading_names[0]].head())
        return dataset

    def __vector_summation(self):
        for subject in self.__reading_names:
            df = pd.DataFrame()
            df['Time'] = self.__data[subject]['Time']
            (df['Ankle'], df['Leg'], df['Trunk']) = (
                np.sqrt(self.__data[subject]["Ankle_X"]*self.__data[subject]["Ankle_X"] + self.__data[subject]["Ankle_Y"]*self.__data[subject]["Ankle_Y"] + self.__data[subject]["Ankle_Z"]*self.__data[subject]["Ankle_Z"]),
                np.sqrt(self.__data[subject]["Leg_X"]*self.__data[subject]["Leg_X"] + self.__data[subject]["Leg_Y"]*self.__data[subject]["Leg_Y"] + self.__data[subject]["Leg_Z"]*self.__data[subject]["Leg_Z"]),
                np.sqrt(self.__data[subject]["Trunk_X"]*self.__data[subject]["Trunk_X"] + self.__data[subject]["Trunk_Y"]*self.__data[subject]["Trunk_Y"] + self.__data[subject]["Trunk_Z"]*self.__data[subject]["Trunk_Z"])
            )
            df['Annotations'] = self.__data[subject]['Annotations']
            self.__data[subject] = df

    def __dimensionality_reduction(self):
        for subject in self.__reading_names:
            ankle, leg, trunk = np.array(self.__data[subject]['Ankle']), np.array(self.__data[subject]['Leg']), np.array(self.__data[subject]['Trunk'])
            acceleration_data = np.vstack((ankle, leg, trunk))

            covariance_matrix = np.cov(acceleration_data)
            eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

            idx = eigenvalues.argsort()[::-1]
            eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:,idx]
            self.__data[subject]['Acceleration'] = np.dot(eigenvectors[:, :1].T, acceleration_data).flatten()

            cols = list(self.__data[subject].columns)
            cols.append(cols.pop(-2))
            self.__data[subject] = self.__data[subject][cols]

    def __outlier_rectification(self, required_fields):
        print()
        for subject in self.__reading_names:
            for field in required_fields:
                q1, q3 = self.__data[subject][field].quantile([0.25, 0.75])
                iqr = q3 - q1
                lower_bound, upper_bound = q1 - 1.5*iqr, q3 + 1.5*iqr

                outliers = self.__data[subject][(self.__data[subject][field] < lower_bound) | (self.__data[subject][field] > upper_bound)]
                print(f"Imputing the obtained outliers in {field} for {subject}. No of outliers: {outliers.shape[0]}")
                for index, row in outliers.iterrows():
                  outlier_index = self.__data[subject].index.get_loc(index)

                  start_index, end_index = max(0, outlier_index - 2), min(len(self.__data[subject]) - 1, outlier_index + 2)
                  neighbor_values = self.__data[subject].loc[start_index:end_index, field].values
                  imputed_value = neighbor_values.mean()
                  self.__data[subject].at[index, field] = imputed_value

    def process_data(self):
        print(f"\n{self.__formatter.return_formatted_string('VECTOR SUMMATION')}")
        print("Processing...")
        self.__vector_summation()
        print("\nVector Summation Completed.")
        print("Result of Vector Summation:")
        print(self.__data[self.__reading_names[0]].head())

        print(f"\n{self.__formatter.return_formatted_string('DIMENSIONALITY REDUCTION')}")
        print("Processing...")
        self.__dimensionality_reduction()
        print("\nDimensionality Reduction Completed.")
        print("Result of Dimensionality Reduction:")
        print(self.__data[self.__reading_names[0]].head())

        print(f"\n{self.__formatter.return_formatted_string('OUTLIER RECTIFICATION')}")
        print("Processing...")
        self.__outlier_rectification(["Ankle", "Leg", "Trunk", "Acceleration"])
        print("\nOutlier Rectification Completed.")
        print("Result of Outlier Rectification:")
        print(self.__data[self.__reading_names[0]].head())

        print("\nData Processing Completed.\n")
        return (self.__data, self.__reading_names)
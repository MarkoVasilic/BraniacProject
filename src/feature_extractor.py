import pandas as pd
from typing import List, Tuple

class FeatureExtractor:
    def __init__(self):
        """
        Initialize the FeatureExtractor.
        """

    def generate_feature_list(self, df: pd.DataFrame, feature_name: str) -> List[str]:
        """
        Create a list of all unique feature values.

        Args:
            df (pd.DataFrame): Input DataFrame.
            feature_name (str): Name of the feature column.

        Returns:
            list: List of feature names, sorted by average ratings.
        """
        # Step 1: Track all ratings associated with each feature in a dictionary
        feature_dict = {}
        for index, row in df.iterrows():
            for sub_feat in row[feature_name]:
                if sub_feat not in feature_dict:
                    feature_dict[sub_feat] = (df['vote_average'][index], 1)
                else:
                    feature_dict[sub_feat] = (
                        feature_dict[sub_feat][0] + (df['vote_average'][index]),
                        feature_dict[sub_feat][1] + 1
                    )

        # Step 2: Calculate average ratings for each feature
        for key in feature_dict:
            feature_dict[key] = feature_dict[key][0] / feature_dict[key][1]

        # Step 3: Create and sort a list of tuples (average rating, feature name)
        lst = list()
        for name in feature_dict:
            lst.append((feature_dict[name], name))
        lst = sorted(lst)

        # Step 4: Create a list of only the feature names, from lowest rating to highest rating
        feature_list = list()
        for element in lst:
            feature_list.append(element[1])

        return feature_list
    
    def calculate_bin_array(self, one_feature_list: List[str], all_features_lists: List[str]) -> List[int]:
        """
        Calculate a binary array based on the presence of elements in the input list.

        Args:
            one_feature_list (list): Input list.
            all_features_lists (list): List of all possible features.

        Returns:
            list: Binary array indicating the presence of elements in the input list.
        """
        bin_list = []
        for element in all_features_lists:
            if element in one_feature_list:
                bin_list.append(1)
            else:
                bin_list.append(0)
        return bin_list
    
    def split_arr(self, arr: List[int], n_splits: int) -> List[List[int]]:
        """
        Split an array into batches of specified size.

        Args:
            arr (list): Input array.
            n_splits (int): Number of elements in each batch.

        Yields:
            list: Batched subarrays.
        """
        for i in range(0, len(arr), n_splits):
            yield arr[i:i + n_splits]

    def find_concentration(self, arr: List[int], n: int = 80) -> List[Tuple[float, int]]:
        """
        Find concentration points in an array.

        Args:
            arr (list): Input array.
            n (int): Number of concentration points to find.

        Returns:
            list: List of concentration points and the number of ones in each batch.
        """
        batches = list(self.split_arr(arr, int(len(arr) / n)))
        concentrations = []
        for i in range(len(batches)):
            point = 0
            num_ones = 0
            for j in range(len(batches[i])):
                if batches[i][j] == 1:
                    point += j + (i * int(len(arr) / n)) # adding correction for batches
                    num_ones += 1
            if num_ones > 0:
                point = point/num_ones
                concentrations.append((point,num_ones))
        return concentrations
    
    def create_concentrations(self, column: pd.Series) -> pd.Series:
        """
        Convert binary feature columns to concentration points.

        Args:
            column (pd.Series): Input Series.

        Returns:
            pd.Series: Series with concentration points for binary feature column.
        """
        column = column.apply(lambda x: self.find_concentration(x))
        return column
    
    def weighted_average(self, arr: List[Tuple[float, int]]) -> float:
        """
        Calculate the weighted average of a list of (position, weight) pairs.

        Args:
            arr (list): List of (position, weight) pairs.

        Returns:
            float: Weighted average.
        """
        total_weight = 0
        weighted_sum = 0

        for position, weight in arr:
            weighted_sum += position * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight != 0 else 0

    def to_weighted_average(self, column: pd.Series) -> pd.Series:
        """
        Convert feature column to weighted averages.

        Args:
            column (pd.Series): Input Series.

        Returns:
            pd.Series: Series with feature column replaced by weighted averages.
        """
        return column.apply(self.weighted_average)
    
    def process_column(self, column: pd.Series, all_features_lists: List[str]) -> pd.Series:
        column = column.apply(lambda x: self.calculate_bin_array(x, all_features_lists))
        column = self.create_concentrations(column)
        column = self.to_weighted_average(column)
        return column

        
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os



class Dataset:
    """
    A general-purpose dataset class for loading and managing model data
    for Bayesian model combination workflows.
    
    Supports .h5 and .csv files, and provides data splitting functionality.
    """

    def __init__(self, data_source=None):
        """
        Initialize the Dataset object.

        :param data_source: Path to the data file (.h5 or .csv).
        """
        self.data_source = data_source
        self.data = {}  # Dictionary of model to DataFrame

    def load_data(self, models, keys=None, domain_keys=None, model_column='model'):
        """
        Load data for specified models from a file and synchronize their domains.

        :param models: List of model names (for HDF5 keys or filtering CSV).
        :param keys: List of columns to extract (optional).
        :param domain_keys: List of columns used to define the common domain (ex: ['N', 'Z']).
        :param model_column: Name of the column in the CSV that identifies which model each row belongs to. Only used when loading from a CSV; ignored for HDF5 files.
        :return: Dictionary with model names as keys and synchronized DataFrames as values.
        """
        if self.data_source is None:
            raise ValueError("Data source must be specified.")

        if not os.path.exists(self.data_source):
            raise FileNotFoundError(f"Data source '{self.data_source}' not found.")

        if domain_keys is None:
            domain_keys = []

        data_dict = {}

        if self.data_source.endswith('.h5'):
            for model in models:
                df = pd.read_hdf(self.data_source, key=model)
                data_dict[model] = df[keys] if keys else df

        elif self.data_source.endswith('.csv'):
            df = pd.read_csv(self.data_source)
            for model in models:
                if model_column not in df.columns:
                    raise ValueError(f"Expected column '{model_column}' not found in CSV.")

                model_df = df[df[model_column] == model]
                data_dict[model] = model_df[keys] if keys else model_df

        else:
            raise ValueError("Unsupported file format. Only .h5 and .csv are supported.")
            
        if domain_keys:
            # Build the intersection of domains from all models
            domain_sets = [df[domain_keys].drop_duplicates() for df in data_dict.values()] # Cconstructs a list of DataFrames — one for each model — that only contain the unique domain points

            # Assume domain_sets is a list of DataFrames already prepared
            common_domain = domain_sets[0]  # Start with the first model's domain
            for next_df in domain_sets[1:]:
                common_domain = pd.merge(common_domain, next_df, on=domain_keys, how='inner')

            for model, df in data_dict.items():
                data_dict[model] = df.merge(common_domain, on=domain_keys, how='inner')
        
        if common_domain.empty:
            print("Warning: No shared domain across models.")
        self.data = data_dict

        # Combine into single DataFrame with 'model' column 
        combined = []
        for model, df in data_dict.items():
            df_copy = df.copy()
            df_copy["model"] = model
            combined.append(df_copy)

        return pd.concat(combined, ignore_index=True)

    def separate_points_distance_allSets(self, list1, list2, distance1, distance2):
            """
            Separates points in list1 into three groups based on their proximity to any point in list2.

            :param list1: List of (x, y) tuples.
            :param list2: List of (x, y) tuples.
            :param distance: The threshold distance to determine proximity.
            :return: Two lists - close_points and distant_points.
            """
            train = []
            validation=[]
            test = []

            train_list_coordinates=[]
            validation_list_coordinates=[]
            test_list_coordinates=[]

            for i in range(len(list1)):
                point1=list1[i]
                close = False
                for point2 in list2:
                    if np.linalg.norm(np.array(point1) - np.array(point2)) <= distance1:
                        close = True
                        break
                if close:
                    train.append(point1)
                    train_list_coordinates.append(i)
                else:
                    close2=False
                    for point2 in list2:
                        if np.linalg.norm(np.array(point1) - np.array(point2)) <= distance2:
                            close2 = True
                            break
                    if close2==True:
                        validation.append(point1)
                        validation_list_coordinates.append(i)
                    else:
                        test.append(point1)
                        test_list_coordinates.append(i)                

            return train_list_coordinates, validation_list_coordinates, test_list_coordinates
    
    def split_data(self, data, splitting_algorithm="random", **kwargs):
        """
        Split data into training, validation, and testing sets using random or inside-to-outside logic.

        :param data: DataFrame or ndarray with rows as coordinate points (ex: (x, y) or (N, Z)).
        :param splitting_algorithm: 'random' (default) or 'inside_to_outside'.
        :param kwargs: Additional arguments depending on the chosen algorithm.
            For 'random': train_size, val_size, test_size
            For 'inside_to_outside': stable_points (list of (x, y)), distance1, distance2
        :return: Tuple of train, validation, test datasets as DataFrames or arrays matching input type.
        """
        if isinstance(data, pd.DataFrame):
            indexable_data = data.reset_index(drop=True)
            point_list = list(indexable_data.itertuples(index=False, name=None))
        elif isinstance(data, np.ndarray):
            indexable_data = pd.DataFrame(data)
            point_list = [tuple(row) for row in data]
        else:
            raise TypeError("Input data must be a pandas DataFrame or numpy ndarray.")

        if splitting_algorithm == "random":
            required = ['train_size', 'val_size', 'test_size']
            if not all(k in kwargs for k in required):
                raise ValueError(f"Missing required kwargs for 'random': {required}")
            
            train_size = kwargs['train_size']
            val_size = kwargs['val_size']
            test_size = kwargs['test_size']

            if not np.isclose(train_size + val_size + test_size, 1.0):
                raise ValueError("train_size + val_size + test_size must equal 1.0")

            # Random split using indexes
            train_idx, temp_idx = train_test_split(indexable_data.index, train_size=train_size, random_state=1)
            val_rel = val_size / (val_size + test_size)
            val_idx, test_idx = train_test_split(temp_idx, test_size=1 - val_rel, random_state=1)

        elif splitting_algorithm == "inside_to_outside":
            required = ['stable_points', 'distance1', 'distance2']
            if not all(k in kwargs for k in required):
                raise ValueError(f"Missing required kwargs for 'inside_to_outside': {required}")
            
            stable_points = kwargs['stable_points']
            distance1 = kwargs['distance1']
            distance2 = kwargs['distance2']

            train_idx, val_idx, test_idx = self.separate_points_distance_allSets(
                point_list, stable_points, distance1, distance2
            )
        else:
            raise ValueError("splitting_algorithm must be either 'random' or 'inside_to_outside'")

        # Extract subsets using index lists
        train_data = indexable_data.iloc[train_idx].reset_index(drop=True)
        val_data = indexable_data.iloc[val_idx].reset_index(drop=True)
        test_data = indexable_data.iloc[test_idx].reset_index(drop=True)

        # Return in same format as input
        if isinstance(data, np.ndarray):
            return train_data.to_numpy(), val_data.to_numpy(), test_data.to_numpy()
        else:
            return train_data, val_data, test_data


    def get_subset(self, filters=None, models_to_filter=None):
        """
        Return a DataFrame of filtered data across specified models.

        :param filters: Dictionary of filtering rules.
                        - key = column name, value = single value, tuple (min, max), list of values, or callable
                        - special key 'multi' allows row-wise lambda functions
        :param models_to_filter: List of model names to apply filters to (defaults to all in self.data).
        :return: Concatenated filtered DataFrame with a 'model' column.
        """
        if filters is None:
            raise ValueError("Filters must be provided.")

        if models_to_filter is None:
            models_to_filter = list(self.data.keys())
        else:
            missing = [m for m in models_to_filter if m not in self.data]
            if missing:
                raise ValueError(f"Models not found in data: {missing}")

        def apply_filters(df):
            result = df.copy()
            for column, condition in filters.items():
                if column == 'multi' and callable(condition):
                    result = result[result.apply(condition, axis=1)]
                elif callable(condition):
                    result = result[condition(result[column])]
                elif isinstance(condition, tuple) and len(condition) == 2:
                    result = result[(result[column] >= condition[0]) & (result[column] <= condition[1])]
                elif isinstance(condition, list):
                    result = result[result[column].isin(condition)]
                else:
                    result = result[result[column] == condition]
            return result

        # Apply filters and add a 'model' column 
        filtered_frames = []
        for model_name in models_to_filter:
            df = self.data[model_name].copy()
            filtered = apply_filters(df)
            filtered["model"] = model_name
            filtered_frames.append(filtered)

        return pd.concat(filtered_frames, ignore_index=True)

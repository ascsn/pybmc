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
        Load data for each property and return a dictionary of synchronized DataFrames.
        Each DataFrame has columns: domain_keys + one column per model for that property.

        Parameters:
            models (list): List of model names (for HDF5 keys or filtering CSV).
            keys (list): List of property names to extract (each will be a key in the output dict).
            domain_keys (list, optional): List of columns used to define the common domain (default ['N', 'Z']).
            model_column (str, optional): Name of the column in the CSV that identifies which model each row belongs to.
                                          Only used for CSV files; ignored for HDF5 files.

        Returns:
            dict: Dictionary where each key is a property name and each value is a DataFrame with columns:
                  domain_keys + one column per model for that property.
                  The DataFrames are synchronized to the intersection of the domains for all models.

        Supports both .h5 and .csv files.
        """
        self.domain_keys = domain_keys 

        if self.data_source is None:
            raise ValueError("Data source must be specified.")
        if not os.path.exists(self.data_source):
            raise FileNotFoundError(f"Data source '{self.data_source}' not found.")
        if keys is None:
            raise ValueError("You must specify which properties to extract via 'keys'.")

        result = {}

        for prop in keys:
            dfs = []
            skipped_models = []

            if self.data_source.endswith('.h5'):
                for model in models:
                    df = pd.read_hdf(self.data_source, key=model)
                    # Check required columns
                    missing_cols = [col for col in domain_keys + [prop] if col not in df.columns]
                    if missing_cols:
                        print(f"[Skipped] Model '{model}' missing columns {missing_cols} for property '{prop}'.")
                        skipped_models.append(model)
                        continue
                    temp = df[domain_keys + [prop]].copy()
                    temp.rename(columns={prop: model}, inplace=True) # type: ignore
                    dfs.append(temp)
            elif self.data_source.endswith('.csv'):
                df = pd.read_csv(self.data_source)
                for model in models:
                    if model_column not in df.columns:
                        raise ValueError(f"Expected column '{model_column}' not found in CSV.")
                    model_df = df[df[model_column] == model]
                    missing_cols = [col for col in domain_keys + [prop] if col not in model_df.columns]
                    if missing_cols:
                        print(f"[Skipped] Model '{model}' missing columns {missing_cols} for key '{prop}'.")
                        skipped_models.append(model)
                        continue
                    temp = model_df[domain_keys + [prop]].copy()
                    temp.rename(columns={prop: model}, inplace=True)
                    dfs.append(temp)
            else:
                raise ValueError("Unsupported file format. Only .h5 and .csv are supported.")

            if not dfs:
                print(f"[Warning] No models with property '{prop}'. Resulting DataFrame will be empty.")
                result[prop] = pd.DataFrame(columns=domain_keys + [m for m in models if m not in skipped_models])
                continue

            # Intersect domain for this property
            common_df = dfs[0]
            for other_df in dfs[1:]:
                common_df = pd.merge(common_df, other_df, on=domain_keys, how="inner")
            # Drop rows with NaN in any required column
            common_df = common_df.dropna()
            result[prop] = common_df
            self.data = result
        return result
    
    def view_data(self, property_name=None, model_name=None):
        """
        View data flexibly based on input parameters.

        - No arguments: returns available property names and model names.
        - property_name only: returns the full DataFrame for that property.
        - model_name only: Return model values across all properties.
        - property_name + model_name: returns a Series of values for the model.

        :param property_name: Optional property name 
        :param model_name: Optional model name 
        :return: dict, DataFrame, or Series depending on input.
        """

        if not self.data:
            raise RuntimeError("No data loaded. Run `load_data(...)` first.")

        if property_name is None and model_name is None:
            props = list(self.data.keys())
            models = sorted(set(col for prop_df in self.data.values() for col in prop_df.columns if col not in self.domain_keys))

            return {
                "available_properties": props,
                "available_models": models
            }

        if model_name is not None and property_name is None:
            # Return a dictionary: {property: Series of model values}
            result = {}
            for prop, df in self.data.items():
                if model_name in df.columns:
                    cols = self.domain_keys + [model_name]
                    result[prop] = df[cols]
                else:
                    result[prop] = f"[Model '{model_name}' not available]"
            return result

        if property_name is not None:
            if property_name not in self.data:
                raise KeyError(f"Property '{property_name}' not found.")

            df = self.data[property_name]

            if model_name is None:
                return df  # Full property DataFrame

            if model_name not in df.columns:
                raise KeyError(f"Model '{model_name}' not found in property '{property_name}'.")

            return df[model_name]


    
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
    
    def split_data(self, data_dict, property_name, splitting_algorithm="random", **kwargs):
        """
        Split data into training, validation, and testing sets using random or inside-to-outside logic.

        :param data_dict: Dictionary output from `load_data`, where keys are property names and values are DataFrames.
        :param property_name: The key in `data_dict` specifying which DataFrame to use for splitting.
        :param splitting_algorithm: 'random' (default) or 'inside_to_outside'.
        :param kwargs: Additional arguments depending on the chosen algorithm.
            For 'random': train_size, val_size, test_size
            For 'inside_to_outside': stable_points (list of (x, y)), distance1, distance2
        :return: Tuple of train, validation, test datasets as DataFrames.
        """
        if property_name not in data_dict:
            raise ValueError(f"Property '{property_name}' not found in the provided data dictionary.")

        data = data_dict[property_name]

        if isinstance(data, pd.DataFrame):
            indexable_data = data.reset_index(drop=True)
            point_list = list(indexable_data.itertuples(index=False, name=None))
        else:
            raise TypeError("Data for the specified property must be a pandas DataFrame.")

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

        train_data = indexable_data.iloc[train_idx]
        val_data = indexable_data.iloc[val_idx]
        test_data = indexable_data.iloc[test_idx]
        
        return train_data, val_data, test_data


    def get_subset(self, property_name, filters=None, models_to_include=None):
        """
        Return a filtered, wide-format DataFrame for a given property.

        :param property_name: Name of the property (e.g., "BE", "ChRad").
        :param filters: Dictionary of filtering rules applied to the domain columns (e.g., {"Z": (26, 28)}).
        :param models_to_include: Optional list of model names to retain in the output.
                                If None, all model columns are retained.
        :return: Filtered wide-format DataFrame with columns: domain keys + model columns.
        """
        if property_name not in self.data:
            raise ValueError(f"Property '{property_name}' not found in dataset.")

        df = self.data[property_name].copy()

        # Apply row-level filters (domain-based)
        if filters:
            for column, condition in filters.items():
                if column == 'multi' and callable(condition):
                    df = df[df.apply(condition, axis=1)]
                elif callable(condition):
                    df = df[condition(df[column])]
                elif isinstance(condition, tuple) and len(condition) == 2:
                    df = df[(df[column] >= condition[0]) & (df[column] <= condition[1])]
                elif isinstance(condition, list):
                    df = df[df[column].isin(condition)]
                else:
                    df = df[df[column] == condition]

        # Optionally restrict to a subset of models
        if models_to_include is not None:
            domain_keys = [col for col in ['N', 'Z'] if col in df.columns]
            allowed_cols = domain_keys + [m for m in models_to_include if m in df.columns]
            df = df[allowed_cols]

        return df

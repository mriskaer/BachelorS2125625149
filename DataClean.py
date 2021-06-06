import pandas as pd

class DataClean:
    # Init dataclean
    def __init__(self, data_set):
        self.data_set = data_set

    # Combines data set from init with a parsed dataset - finds key itself
    def combine_data_sets(self, merge_file,del_after):
        self.data_set = pd.merge(self.data_set, merge_file)
        del self.data_set[del_after]
        return self.data_set

    # Removes column from dataset - takes list input pass array of strings
    def keep_columns(self, col):
        for i in self.data_set.head():
            if i not in col:
                del self.data_set[i]
        return self.data_set

    # Drops all rows with N/A (Wont delete oneHotEncodes because we havent split by the time we use this function)
    def drop_na_rows(self):
        self.data_set = self.data_set.dropna()
        return self.data_set

    #Splitting and onehot encoding the string values into integers.
    def one_hot(self, encode_column):
        data_split = self.data_set[encode_column].str.split(", ").apply(pd.Series)
        df = (pd.get_dummies(data_split[0])
              .add(pd.get_dummies(data_split[1]), fill_value=0)
              .astype(int))
        self.data_set=self.data_set.join(df)
        del self.data_set[encode_column]
        return self.data_set

    #Splits column into a list and creates a new column with the length of that list
    def get_length(self,length_column):
        data_split = self.data_set[length_column].str.split(", ")
        df = data_split.str.len()
        del self.data_set[length_column]
        self.data_set = self.data_set.join(df)
        return self.data_set

    # Drops all columns that are non-numeric
    def drop_non_numeric(self):
        for i in self.data_set.head(1):
            self.data_set = self.data_set[self.data_set[i].apply(lambda x: x.isnumeric())]
            return self.data_set

    # Makes all values numeric in the dataset
    def make_all_numeric(self):
        for i in self.data_set.columns:
            self.data_set[i] = pd.to_numeric(self.data_set[i])

    # Function for full clean at the data set.
    def full_clean(self, merge_file, keep_col_list, encode_col,del_afterlife,length_column):
        self.combine_data_sets(merge_file,del_afterlife)
        self.keep_columns(keep_col_list)
        self.drop_na_rows()
        self.one_hot(encode_col)
        self.get_length(length_column)
        self.drop_non_numeric()
        self.make_all_numeric()
        return self.data_set
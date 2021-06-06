# Import libraries and our other classes
from DataClean import *
from DashboardController import *

def main():

    # Import dataset(s) from the directory Datasets
    data_set_movies = pd.read_csv(r"Datasets/IMDb_movies.csv", low_memory=False)
    data_set_ratings = pd.read_csv(r"Datasets/IMDb_ratings.csv", low_memory=False)

    # Calling DataClean, in this case on a subset of the data so code does not take a long time to run.
    data_clean = DataClean(data_set_movies.head(2000))
    keep_col_list = ['imdb_title_id', 'year', 'genre', 'duration', 'country', 'total_votes', 'avg_vote']
    df = data_clean.full_clean(merge_file=data_set_ratings.head(2000),
                               keep_col_list=keep_col_list,
                               encode_col='genre',
                               del_afterlife='imdb_title_id',
                               length_column='country')

    # Creating the DashboardController and calling its dash_frontend method to launch the dashboard
    dashboard = DashboardController(df)
    dashboard.dash_frontend()

if __name__ == "__main__":
    main()

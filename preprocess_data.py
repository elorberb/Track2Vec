import pandas as pd

def save_rows_to_csv(input_file_path, output_file_path, headers, n=None, skip_existing_headers=False):
    """
    Reads the first n rows of a large text file, ignoring existing headers if specified, 
    adds new headers, and saves them to a CSV file using Pandas.
    
    Parameters:
    - input_file_path: Path to the input text file.
    - output_file_path: Path where the output CSV file will be saved.
    - headers: Headers for the CSV file, with hyphens replaced by underscores.
    - n: Number of rows to read and save, or None to save all rows.
    - skip_existing_headers: Boolean indicating whether to skip the first row (existing headers).
    """
    skiprows = 1 if skip_existing_headers else None
    df = pd.read_csv(input_file_path, sep='\t', names=headers, nrows=n, skiprows=skiprows, on_bad_lines='skip') if n else pd.read_csv(input_file_path, sep='\t', names=headers, skiprows=skiprows, on_bad_lines='skip')
    df.to_csv(output_file_path, index=False)
    
    if n:
        print(f"Saved the first {n} rows to {output_file_path}")
    else:
        print(f"Saved all rows to {output_file_path}")


def create_training_dataset(events_filepath, users_filepath, output_filepath):
    # Load events data
    events_df = pd.read_csv(events_filepath)

    # Count the number of times each user listened to each track
    user_track_count_df = events_df.groupby(['user_id', 'track_id']).size().reset_index(name='user_track_count')

    # Load the users data
    users_df = pd.read_csv(users_filepath)

    # Merge the count with the users data
    users_with_counts_df = users_df.merge(user_track_count_df, on='user_id', how='left')

    # Create a list of track IDs for each user
    users_with_counts_df['track_ids'] = users_with_counts_df.groupby('user_id')['track_id'].transform(lambda x: list(x.unique()))

    # Drop duplicates to avoid repetition for each user
    final_df = users_with_counts_df.drop_duplicates(subset='user_id')

    # Select the columns of interest
    final_df = final_df[['user_id', 'track_ids', 'playcount', 'gender', 'user_track_count']]

    # Save the final dataframe to a CSV file
    final_df.to_csv(output_filepath, index=False)

    return final_df


if __name__ == "__main__":
    saving_data_path = "/home/etaylor/code_projects/track2vec/data"
    evalrs_events_raw = "/home/etaylor/.cache/evalrs/LFM-1b_LEs.txt"
    evalrs_tracks_raw = "/home/etaylor/.cache/evalrs/LFM-1b_tracks.txt"
    evalrs_users_raw = "/home/etaylor/.cache/evalrs/LFM-1b_users.txt"
    
    events_headers = ['user_id', 'artist_id', 'album_id', 'track_id', 'timestamp']
    tracks_headers = ['track_id', 'track_name', 'artist_id']
    users_headers = ['user_id', 'country', 'age', 'gender', 'playcount', 'timestamp']

    # print("Start preprocessing data")
    # print("Preprocessing events data")
    # save_rows_to_csv(evalrs_events_raw, f"{saving_data_path}/evalrs_events.csv", events_headers, 10000)
    # print("Preprocessing tracks data")
    # save_rows_to_csv(evalrs_tracks_raw, f"{saving_data_path}/evalrs_tracks.csv", tracks_headers, 10000)
    # print("Preprocessing users data")
    # save_rows_to_csv(evalrs_users_raw, f"{saving_data_path}/evalrs_users.csv", users_headers, skip_existing_headers=True)
    
    # Define file paths
    users_filepath = f"{saving_data_path}/evalrs_users.csv"
    tracks_filepath = f"{saving_data_path}/evalrs_tracks.csv"
    events_filepath = f"{saving_data_path}/evalrs_events.csv"
    output_filepath = f'{saving_data_path}/final_training_dataset.csv'

    # Call the function to create the dataset
    final_dataset = create_training_dataset(events_filepath, users_filepath, output_filepath)
    print(final_dataset.head())

import requests
from io import StringIO
import pandas as pd


def getDataFromWindsor():
    # Get the Windsor.ai API endpoint from windsor.ai platform with query parameters to get ads data:
    # - api_key: your API key
    # - date_preset: fetches data from last year
    # - fields: specifies which fields to fetch from ads data
    api_url = "[your_ads_api_url]"

    # Make the GET request to the API
    response = requests.get(api_url)

    # Extract the 'data' portion of the JSON response
    data = response.json()['data']

    # Load the data into a pandas DataFrame
    df = pd.DataFrame(data)

    # Get conversions data from Windsor.ai. We are using GA4 data to get conversions from Windsor.
    # You can replace this with your own conversions data. And perform operations according to your data needs.
    conversions_api_url = "[your_conversions_api_url]"
    conversions_response = requests.get(conversions_api_url)
    conversions_data = conversions_response.json()['data']
    conversions_df = pd.DataFrame(conversions_data)
    conversions_df = conversions_df[['date', 'event_count']]
    conversions_df.rename(columns={'event_count': 'conversions'}, inplace=True)
    conversions_df = conversions_df.groupby(['date']).sum().reset_index()

    # Keep only relevant columns depending on your analysis needs
    df = df[['date', 'source', 'impressions', 'clicks', 'spend']]

    # Group the data by date and sum numerical metrics across all campaigns/accounts for each day.
    # Make sure you have fields aggregated on which you want to run the model
    # Meridian requires one row per day for time series modeling
    df = df.groupby(['date', 'source']).sum().reset_index()

    # Reshape the data to have separate impressions, clicks, conversions, spend for different sources
    df = df.pivot(index='date', columns='source',
                                values=['impressions', 'spend', 'clicks'])

    # Flatten MultiIndex columns
    df.columns = [f"{col[1]}_{col[0]}" for col in df.columns]
    df = df.reset_index()

    # Merge pivoted data with conversions
    df = df.merge(conversions_df, on='date')

    # Convert the 'date' column to datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Sort the data by date
    df = df.sort_values(by=['date'])

    # This step is necessary if you dont have evenly spaced time series data required by meridian
    # Resample the data to have one row per day (daily frequency)
    # This fills in missing dates with NaNs (important for time series modeling)
    df = df.set_index('date').resample('D').asfreq()

    # Fill missing values with zeros you can fill with any values suited to your analysis (assumes no activity on missing dates)
    for suffix in ['clicks', 'impressions', 'conversions', 'spend']:
        cols_to_fill = df.columns[df.columns.str.endswith(suffix)]
        df[cols_to_fill] = df[cols_to_fill].fillna(0)

    # Reset the index to move 'date' back into a column
    df = df.reset_index()

    # Convert the final DataFrame to a CSV string in memory using StringIO
    csv_data = StringIO()
    df.to_csv(csv_data, index=False)
    csv_data.seek(0)  # Move to the start of the stream

    # Return the in-memory CSV file-like object for use in other parts of the pipeline
    return csv_data

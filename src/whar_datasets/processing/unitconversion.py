from typing import Dict
import pandas as pd
from whar_datasets.core.config import WHARConfig
from tqdm.auto import tqdm

# Standard gravity constant for converting g to m/s^2
G_TO_MS2 = 9.80665

def convert_units(sessions: Dict[int, pd.DataFrame], config: WHARConfig) -> Dict[int, pd.DataFrame]:
    """
    Converts sensor data units to a standard format based on the dataset configuration.

    Args:
        sessions (Dict[int, pd.DataFrame]): A dictionary of session DataFrames.
        config (WHARConfig): The dataset configuration object.

    Returns:
        Dict[int, pd.DataFrame]: The sessions with converted units.
    """
    if not config.unit_conversion:
        return sessions

    print("Converting units for sessions...")
    for session_id, session_df in tqdm(sessions.items()):
        for sensor_channel in config.sensor_channels:
            if sensor_channel.name in session_df.columns:
                if sensor_channel.unit == "g":
                    session_df[sensor_channel.name] *= G_TO_MS2
                # Add other unit conversions here as needed
                # e.g., if sensor_channel.unit == 'rad/s':
                #           session_df[sensor_channel.name] = ...

        sessions[session_id] = session_df

    return sessions

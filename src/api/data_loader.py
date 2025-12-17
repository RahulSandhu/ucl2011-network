import os
import warnings

from statsbombpy import sb

# Configuration
warnings.filterwarnings("ignore")
base_dir = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
data_dir = os.path.join(base_dir, "../../data")
file_path = os.path.join(data_dir, "barca_manutd_2011_events.csv")

# Find Competition & Season
comps = sb.competitions()
target = comps[
    (comps["competition_name"] == "Champions League")
    & (comps["season_name"] == "2010/2011")
]

# Handle not found
if target.empty:
    raise ValueError("Champions League 2010/2011 not found in StatsBomb.")

# Get IDs
comp_id = target.iloc[0]["competition_id"]
season_id = target.iloc[0]["season_id"]

# Find the Match
matches = sb.matches(competition_id=comp_id, season_id=season_id)

# Filter for Barca vs Man Utd
final = matches[
    (
        (matches["home_team"].str.contains("Barcelona"))
        | (matches["away_team"].str.contains("Barcelona"))
    )
    & (
        (matches["home_team"].str.contains("Manchester United"))
        | (matches["away_team"].str.contains("Manchester United"))
    )
]

# Handle not found
if final.empty:
    raise ValueError("Match not found.")

# Download and save
match_id = final.iloc[0]["match_id"]
events = sb.events(match_id=match_id)

print(f"Match Found with ID: {match_id}. Downloading events...")

# Save data
os.makedirs(data_dir, exist_ok=True)
events.to_csv(file_path, index=False)

print(f"Data saved.")

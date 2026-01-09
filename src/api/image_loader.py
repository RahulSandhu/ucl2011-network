# https://fcpython.com/blog/scraping-lists-transfermarkt-saving-images
# https://stackoverflow.com/questions/51710082/what-does-unicodedata-normalize-do-in-python

import os
import re
import time
import unicodedata

import requests
from bs4 import BeautifulSoup

# To fix relative paths problem
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
IMAGES_DIR = ROOT / "images"

# Configuration
teams = [
    {
        "name": "FC Barcelona (2010-11)",
        "url": "https://www.transfermarkt.com/fc-barcelona/kader/verein/131/saison_id/2010",
        "save_folder": IMAGES_DIR / "fcb",
    },
    {
        "name": "Manchester United (2010-11)",
        "url": "https://www.transfermarkt.com/manchester-united/kader/verein/985/saison_id/2010",
        "save_folder": IMAGES_DIR / "man_utd",
    },
]

# HTTP headers to mimic a browser
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}


# Function to clean filenames
def clean_filename(name):
    # Normalize to NFKD and remove diacritics
    nfkd_form = unicodedata.normalize("NFKD", name)
    name_ascii = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    name_lower = name_ascii.lower()

    # Replace spaces with underscores and remove invalid characters
    name_underscored = name_lower.replace(" ", "_")
    final_name = re.sub(r"[^a-z0-9_]", "", name_underscored)

    return final_name


# Function to download team faces
def download_team_faces(team_config):
    # Extract team details
    name = team_config["name"]
    url = team_config["url"]
    folder = team_config["save_folder"]

    # Create folder if it doesn't exist #CHANGED
    folder.mkdir(parents=True, exist_ok=True)


    # Fetch and parse the team page
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")
    players_table = soup.find("table", class_="items")

    # Find all player rows
    player_rows = players_table.find_all("tr", class_=["odd", "even"])

    # Loop through each player row
    for row in player_rows:
        # Find the image tag
        img_tag = row.find("img", class_="bilderrahmen-fixed")

        # If image tag is found
        if img_tag:
            # Extract player name and image URL
            raw_name = img_tag.get("title") or img_tag.get("alt")
            img_url = img_tag.get("data-src") or img_tag.get("src")

            # Validate extracted data
            if not raw_name or not img_url:
                continue

            # Clean filename and prepare path
            safe_name = clean_filename(raw_name)
            filename = folder / f"{safe_name}.jpg"

            # Skip if already exists
            if os.path.exists(filename):
                continue

            # Filter out default placeholder images
            if "default.jpg" in img_url or "portrait_small.png" in img_url:
                continue

            # Download
            img_data = requests.get(img_url, headers=headers).content

            # Save to file
            with open(filename, "wb") as handler:
                handler.write(img_data)

            # Small pause
            time.sleep(0.5)


# Run the downloader
for team in teams:
    download_team_faces(team)
    time.sleep(2)

from splits import DATASET_PATH
import requests
import time
import json
import os
import re
import tqdm
from ratelimiter import RateLimiter

# Register an application with last.fm and
# save keys as environment variables
API_KEY = os.environ["LASTFM_API_KEY"]
USER_AGENT = os.environ["LASTFM_USER_AGENT"]
BASE_URL = "http://ws.audioscrobbler.com/2.0/"
DATASET_PATH = os.environ["DATASET_PATH"]


def get_top_tags():
    """
    Retrieves the 50 most used tags (genres)

    Retries when rate limited (WARNING: UNSAFE RECURSION).

    Returns:
    JSON Response
    """
    headers = {"user-agent": USER_AGENT}

    payload = {
        "api_key": API_KEY,
        "method": "tag.gettoptags",
        "format": "json",
    }

    response = requests.get(BASE_URL, headers=headers, params=payload)

    if response.status_code == 200:
        page = response.json()
        return page
    elif response.status_code == 429:
        time.sleep(60)
        return get_top_tags()
    else:
        raise Exception(f"Bad response: {response.status_code}\n{response.content}")


@RateLimiter(max_calls=4, period=1)
def get_page(tag, page=1):
    """
    Gets a single page of albums for a tag.

    Fetches a maximum of 1000 results. 
    Rate limited to 4 calls per second.
    Retries when rate limited (WARNING: UNSAFE RECURSION).

    Parameters:
        tag (str):  The genre to search for
        page (int): The page number

    Returns:
        JSON Response
    """
    headers = {"user-agent": USER_AGENT}

    payload = {
        "api_key": API_KEY,
        "method": "tag.gettopalbums",
        "tag": tag,
        "limit": 1000,
        "page": page,
        "format": "json",
    }

    response = requests.get(BASE_URL, headers=headers, params=payload)

    if response.status_code == 200:
        page = response.json()
        return page
    elif response.status_code == 429:
        print("RATE LIMITED")
        time.sleep(60)
        return get_page(tag, page)
    else:
        raise Exception(f"Bad response: {response.status_code}\n{response.content}")


def convert_album(album):
    """
    Extracts cover image id and album metadata from a single album result

    Illegal characters are replaced in filename.
    Filenames are appended with `.png`

    Parameters:
        album (dict):   Album response in last.fm format

    Returns:
        dict: {"id": (str), "filename": (str)}
    """
    artist_name = album["artist"]["name"]
    album_name = album["name"]
    filename = f"{artist_name} - {album_name}"
    filename = re.sub("[\./\?<>:\*\|]", "", filename) + ".png"

    img_url = album["image"][0]["#text"]
    img_id = img_url.split("/")[-1].split(".")[0]

    return {"id": img_id, "filename": filename}


def convert_albums(page):
    """
    Converts page into list of album cover dicts

    Parameters:
        page (dict):    last.fm `getTopAlbums` response in JSON format

    Returns:
        list(dict)
    """
    covers = list(map(convert_album, page["albums"]["album"]))
    return covers


def save_img(img_id, filepath, size=174):
    """
    Fetches a cover image from last.fm and saves to disk

    Rate limited to 4 calls per second.
    Retries when rate limited (WARNING: UNSAFE RECURSION).
    Fails gracefully on 404

    Parameters:
        img_id (str):   ID of cover image
        filepath (str): Full path to save to
        size (int):     Image resolution. Can be one of 34, 64, 174.

    Returns:
        None
    """
    response = download_img(
        f"https://lastfm.freetls.fastly.net/i/u/{size}s/{img_id}.png"
    )

    if response.status_code == 200:
        with open(filepath, "wb") as f:
            f.write(response.content)
    elif response.status_code == 429:
        print("RATE LIMITED")
        time.sleep(60)
        return save_img(img_id, filepath, size)
    elif response.status_code == 404:
        print("NOT FOUND: " + filepath + " " + img_id)
    else:
        raise Exception(f"Bad response: {response.status_code}\n{response.content}")


@RateLimiter(max_calls=4, period=1)
def download_img(path):
    """
    Wrapper for a get request.

    Rate limited to 4 calls per second.
    Times out after 5 seconds, retries once after 30 seconds.

    Parameters:
        path (str): Request URL

    Returns:
        `requests` Response object
    """
    try:
        return requests.get(path, timeout=5)
    except:
        time.sleep(30)
        return requests.get(path, timeout=5)


def get_total_pages(page):
    """
    Gets number of pages for a tag

    Parameters:
        page (dict):    last.fm `getTopAlbums` response in JSON format

    Returns:
        int:    Total number of pages
    """
    return int(page["albums"]["@attr"]["totalPages"])


def paginate_tag(tag, max_results=1000):
    """
    Fetches pages for a tag from last.fm.

    Fetches new pages until `max_results` is reached or no results remain.
    May return more than `max_results` images if `max_results % 1000 != 0`.

    Parameters:
        tag (str):          Genre tag to search for.
        max_results (int):  When to stop paginating.

    Returns:
        list(dict): List of album cover metadata
    """
    pages = []

    max_page = round(max_results / 1000)

    page = get_page(tag, page=1)
    pages.append(page)

    total_pages = get_total_pages(page)
    if max_page < total_pages:
        total_pages = max_page

    for i in range(2, total_pages + 1):
        pages.append(get_page(tag, i))

    return pages


if __name__ == "__main__":
    ### 21 tags taken from top 50 tags, with some changes
    tags = [
        "classic rock",
        "hard rock",
        "alternative rock",
        "indie",
        "death metal",
        "heavy metal",
        "metalcore",
        "jazz",
        "folk",
        "punk",
        "Hip-Hop",
        "rap",
        "soul",
        "blues",
        "house",
        "dance",
        "trance",
        "reggae",
        "country",
        "k-pop",
        "ambient",
    ]

    base_path = DATASET_PATH

    # iterate through tags
    for tag in tags:
        # fetch flat map of album cover data
        pages = paginate_tag(tag, max_results=1000)
        covers = []
        for page in pages:
            covers.extend(convert_albums(page))

        data = {"tag": tag, "covers": covers}
        print(f"For {tag}: {len(covers)} covers")

        if not os.path.exists(base_path + tag):
            os.makedirs(base_path + tag)

        # save filenames and ids to json
        with open(f"{base_path}/{tag}.json", "w") as f:
            json.dump(data, f)

        # download and save images
        for cover in tqdm.tqdm(covers):
            filepath = base_path + tag + "/" + cover["filename"]
            if not os.path.exists(filepath):
                save_img(img_id=cover["id"], filepath=filepath)

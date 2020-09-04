import requests
import argparse
import os
import shutil

"""
    Google Drive Downloader
    Adapted from https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
"""

def download_from_google_drive(file_id, dest, file_name):
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id})
    token = get_warning_token(response)
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, dest, file_name)

def get_warning_token(response):
    for k, v in response.cookies.items():
        if k.startswith('download_warning'):
            return v
    return None

def renew_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_response_content(response, dest, file_name):
    renew_path(dest)
    CHUNK_SIZE = 32768
    with open(os.path.join(dest, file_name), "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--id')
    parser.add_argument('-f', '--name')
    parser.add_argument('-d', '--dest', default="dataset")
    args = parser.parse_args()
    print(vars(args))
    download_from_google_drive(args.id, args.dest, args.name)
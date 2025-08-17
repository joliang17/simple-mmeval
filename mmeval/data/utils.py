import requests

def download_tsv(file_url: str, save_path: str) -> str:
    """Download TSV file from URL to
    
    Parameters
    ----------
    file_url : str
        URL to download TSV file from
    save_path : str
        Path to save the downloaded file
    """
    file_dir = os.path.dirname(save_path)
    os.makedirs(file_dir, exist_ok=True)
    
    response = requests.get(file_url, stream=True)
    response.raise_for_status()
    
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=100000):
            f.write(chunk)
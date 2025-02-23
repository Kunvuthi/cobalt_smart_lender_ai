import gdown

data_url = "https://drive.google.com/drive/folders/1I1QSqJOSrkC4rGYvFKQsHxxDh7zUGcV_?usp=drive_link"
output = "data/all_data.zip"
gdown.download(data_url, output, quiet=False)
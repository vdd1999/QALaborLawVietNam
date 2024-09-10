# QA Labor law Vietnam website

## Step run web
* py -V:3.11 -m venv venv
* .\venv\Scripts\Active.ps1
* pip install -r requirements.txt
* cd models/
* gdown https://drive.google.com/drive/folders/1d0qemblTqHrBHmnz4fgS21gAI9frC8YP -O phoBert_model --folder
* cd ../
* flask run --debug


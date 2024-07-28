sudo dnf install python3.11
sudo dnf install git
sudo dnf install pip
git clone https://github.com/vdd1999/QALaborLawVietNam.git
cd QALaborLawVietNam
git checkout website
export TMPDIR=$HOME/tmp
mkdir -p $TMPDIR
pip install virtualenv
virtualenv -p python3.11 venv
source venv/bin/activate
pip install -r requirements.txt
cd models/
gdown https://drive.google.com/drive/folders/1d0qemblTqHrBHmnz4fgS21gAI9frC8YP -O phoBert_model --folder
cd ../
flask run -h 0.0.0.0 --debug
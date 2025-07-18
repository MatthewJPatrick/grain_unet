# Due to dependencies, this project only runs on Python 3.11
brew install pyenv
pyenv install 3.11.9
pyenv local 3.11.9

# Create a new virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
# COVID_Retweet_Prediction_Backend
Backend Flask server for COVID retweet prediction model GUI. Made for SUTD Term 8 AI Project. 

Steps to boot up the local server:

1. Create a new virtual environment to prevent clashing dependencies later.
2. Clone this repository.
3. Open your terminal, activate your virtual environment and cd into this repository.
4. Run `git lfs install` and `git lfs checkout` to download the required CSV files.
5. Run `pip install -r requirements.txt` in the terminal and wait until all required packages are installed.
6. Run `python flask_backend.py` and wait until the local server boots up. Verify that it is running at port `5000`, otherwise kill any processes running in your port `5000` and re-run the command.

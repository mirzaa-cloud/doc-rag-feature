py -3.10 -m venv kyenv

.\kyenv\Scripts\Activate

python.exe -m pip install --upgrade pip


pip install -r requirements.txt

uvicorn app.main:app --reload
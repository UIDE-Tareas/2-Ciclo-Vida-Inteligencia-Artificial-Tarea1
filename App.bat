python -m App --Create
type nul > ".venv\.nosync"
call ".venv\Scripts\activate.bat"
python -m App --Install
python -m App --Info
python -m App

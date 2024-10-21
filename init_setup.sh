echo [$(date)]: "START"
echo [$(date)]: "Create conda evironment with python"

conda create -p ./venv python=3.8 -y

echo [$(date)]: "Activate evironment"
conda activate ./venv

echo [$(date)]: "Install requirements"
pip install -r requirements.txt

echo [$(date)]: "End"
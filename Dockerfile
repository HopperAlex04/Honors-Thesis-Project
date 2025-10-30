FROM python:3.11.4

WORKDIR /wdir 

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install torch

COPY . .

CMD ["python", "Main.py"]
FROM python:3.7

COPY . /work
WORKDIR /work

RUN pip install -r requirements.txt
ENTRYPOINT ["python", "run.py"]

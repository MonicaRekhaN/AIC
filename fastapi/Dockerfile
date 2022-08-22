FROM tiangolo/uvicorn-gunicorn:python3.7

RUN mkdir /fastapi

COPY requirements1.txt /fastapi

WORKDIR /fastapi

# RUN pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install -r requirements1.txt --no-cache-dir
COPY . /fastapi

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
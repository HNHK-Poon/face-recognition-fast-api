FROM python:3.8-slim

WORKDIR /usr/src
COPY ./ ./
ADD requirementss.txt .

# RUN apt-get update \
#     && apt-get install --yes --no-install-recommends \
#         gcc g++ libffi-dev 

RUN apt-get update -y && \
    apt-get install build-essential cmake pkg-config tcl tk -y

RUN pip install -r requirementss.txt --no-cache-dir
#RUN pip install --trusted-host pypi.python.org -r requirementss.txt

WORKDIR /usr/src/app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]


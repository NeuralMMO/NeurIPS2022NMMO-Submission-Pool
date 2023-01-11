FROM neurips2022nmmo/submission-runtime:latest

ARG submission
# ARG PIP_SOURCE=https://mirrors.cloud.tencent.com/pypi/simple

WORKDIR /home/neurips2022nmmo-submission-pool
COPY evaluate.py evaluate.py
COPY ${submission} ${submission}
RUN if [ -f ${submission}/requirements.txt ]; then pip3 install --no-cache-dir -r ${submission}/requirements.txt; fi;


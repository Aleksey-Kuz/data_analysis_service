FROM apache/airflow:2.10.5-python3.9

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends gosu \
    && apt-get autoremove -yqq --purge \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/airflow/scripts \
    && chown -R airflow:root /opt/airflow/scripts \
    && chmod -R 775 /opt/airflow/scripts

COPY ./airflow/scripts/ /opt/airflow/scripts/

USER airflow

COPY ./requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && rm requirements.txt
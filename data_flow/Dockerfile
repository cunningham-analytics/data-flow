FROM python:3.12

RUN apt-get update \
    && apt-get install -y --no-install-recommends

WORKDIR data_flow
COPY . .

RUN pip3 install -r requirements.txt
RUN dbt deps

EXPOSE 8080
CMD        dbt build --profiles-dir profiles && \
           dbt docs generate --profiles-dir profiles && \
           dbt docs serve --profiles-dir profiles
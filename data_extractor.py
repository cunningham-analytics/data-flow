

import yfinance as yahooFinance
from datetime import date, timedelta
import psycopg2
import os
import csv
import pandas as pd

dbUser = 'admin' # os.environ["POSTGRES_USER"]
dbPW = 'root' # os.environ["POSTGRES_PW"]

connStr = f"host=localhost dbname=analytics user={dbUser} password={dbPW}"
conn = psycopg2.connect(connStr)

dateRange = timedelta(days=180)

ticker = "GME"
endDate = date.today()
startDate = endDate - dateRange
stockInfo = yahooFinance.Ticker(ticker)

df = stockInfo.history(start=startDate, end=endDate)
df.reset_index(inplace=True)
df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
df.drop(['Dividends','Stock Splits'], inplace=True, axis=1)
df.to_dict(orient='records')

incrementalData = pd.DataFrame(df)
incrementalData['ticker'] = ticker

fileName = f"{ticker}__{startDate}_{endDate}".replace('-', '')
filePath = f"inc_data_repo/{fileName}.csv"

incrementalData.to_csv(filePath, index=False)


cur = conn.cursor()

cur.execute(f"drop table if exists raw_store.{fileName}")
conn.commit()

cur.execute(
    f"""
    create table raw_store.{fileName} (
        date varchar(12),
        open float,
        high float,
        low float,
        close float,
        volume float,
        ticker varchar(20)
    )"""
)
conn.commit()

cur.execute("set schema 'raw_store' ")
conn.commit()

with open(filePath, 'r') as f:
    next(f) # Skip the header row.
    cur.copy_expert(f"COPY raw_store.{fileName} FROM STDIN WITH (FORMAT csv)", f)
conn.commit()


cur.execute(
    f"""
    create table if not exists raw_store.raw_{ticker} (
        date varchar(12),
        open float,
        high float,
        low float,
        close float,
        volume float,
        ticker varchar(20)
    )"""
)

conn.commit()



cur.execute(
    f"""
    delete from raw_store.raw_{ticker}
    where date::date between '{startDate}'::date and '{endDate}'::date
    """
)

conn.commit()


cur.execute(
    f"""
    insert into raw_store.raw_{ticker} select * from raw_store.{fileName}
    """
)

conn.commit()
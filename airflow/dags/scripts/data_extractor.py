

import yfinance as yahooFinance
from datetime import date, timedelta
import psycopg2
import os
import csv
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

tickers = ["GME", "AAPL", "MSFT", "FFIE", "GOOGL"] 

def load_stock_to_db(ticker, conn, startDate, endDate):

    logger = logging.getLogger()
    stockInfo = yahooFinance.Ticker(ticker)

    df = stockInfo.history(start=startDate, end=endDate)
    df.reset_index(inplace=True)
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    df.drop(['Dividends','Stock Splits'], inplace=True, axis=1)
    df.to_dict(orient='records')

    incrementalData = pd.DataFrame(df)
    incrementalData['ticker'] = ticker

    stocks_root = os.environ['DBT_STOCKS_HOME']
    fileName = f"{ticker}__{startDate}_{endDate}".replace('-', '')
    filePath = f"{stocks_root}/inc_data_repo/{fileName}.csv"

    incrementalData.to_csv(filePath, index=False)

    cur = conn.cursor()

    logger.info(f'Creating raw_store table {fileName}')
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


    logger.info(f'Starting updating raw_store table raw_{ticker}')
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

    logger.info(f'Completed updating raw_store table raw_{ticker}')


def run_data_extractor(tickers):

    logger = logging.getLogger()

    dateRange = timedelta(days=500)

    endDate = date.today()
    startDate = endDate - dateRange

    dbUser = 'admin' # os.environ["POSTGRES_USER"]
    dbPW = 'root' # os.environ["POSTGRES_PW"]
    connStr = f"host=localhost dbname=analytics user={dbUser} password={dbPW}"
    conn = psycopg2.connect(connStr)

    for ticker in tickers:

        logger.info(f"Beginning data extraction for {ticker}")
        load_stock_to_db(ticker, conn, startDate=startDate, endDate=endDate)

    return None

if __name__ == '__main__':
    run_data_extractor(tickers=tickers)
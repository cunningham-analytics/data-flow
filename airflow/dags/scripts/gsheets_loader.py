
import pygsheets
import pandas as pd
import psycopg2
import os
import logging

logging.basicConfig(level=logging.INFO)


def clear_existing_data(gsheets_client, gsheet_key, data_interval_start, data_interval_end, sheet_name):

    logger = logging.getLogger()

    wb = gsheets_client.open_by_key(gsheet_key)
    tab = wb.worksheet_by_title(sheet_name)
    values = tab.get_all_values(value_render="FORMATTED_VALUE")
    searchCol = 2

    logger.info(f'Finding existing data to clear and replace for date interval {data_interval_start} to {data_interval_end}')
    deleteRows = [i for i, r in enumerate(values) if r[searchCol - 1] >= data_interval_start and r[searchCol - 1] <= data_interval_end]
    if deleteRows == []:
        logger.info(f'No data to clear for date interval {data_interval_start} to {data_interval_end}')
        return None
    reqs = [
        {
            "deleteDimension": {
                "range": {
                    "sheetId": tab.id,
                    "startIndex": e,
                    "endIndex": e + 1,
                    "dimension": "ROWS",
                }
            }
        }
        for e in deleteRows
    ]
    reqs.reverse()

    logger.info(f'Clearing data for date interval {data_interval_start} to {data_interval_end}')
    gsheets_client.sheet.batch_update(gsheet_key, reqs)

    return None


def load_new_data(gsheets_client, gsheet_key, data_interval_start, data_interval_end, sheet_name):

    logger = logging.getLogger()

    gc = gsheets_client
    dbUser = os.environ["POSTGRES_USER"]
    dbPW = os.environ["POSTGRES_PW"]

    connStr = f"host=localhost dbname=analytics user={dbUser} password={dbPW}"
    conn = psycopg2.connect(connStr)

    cur = conn.cursor()

    logger.info(f'Querying data for date interval {data_interval_start} to {data_interval_end}')
    cur.execute(
        f"""
        select *
        from production_dw.stocks__closing_metrics
        where report_date between '{data_interval_start} 'and '{data_interval_end}'
        order by ticker, report_date
        """
    )
    conn.commit()

    df = pd.DataFrame(cur.fetchall())

    logger.info(f'Creating csv for testing for date interval {data_interval_start} to {data_interval_end}')

    wb = gc.open_by_key(gsheet_key)

    #select the first sheet 
    tab = wb.worksheet_by_title(sheet_name)
    # tab = wb[0]

    #update the first sheet with df, starting at cell B2. 
    logger.info(f'Sending data to report for date interval {data_interval_start} to {data_interval_end}')
    tab.set_dataframe(df,(0,0))

    return None


def run_gsheets_loader(gsheets_client, gsheet_key, data_interval_start, data_interval_end, sheet_name):

    clear_existing_data(gsheets_client=gsheets_client,
                        gsheet_key=gsheet_key,
                        data_interval_start=data_interval_start,
                        data_interval_end=data_interval_end,
                        sheet_name=sheet_name)

    load_new_data(gsheets_client=gsheets_client,
                    gsheet_key=gsheet_key,
                    data_interval_start=data_interval_start,
                    data_interval_end=data_interval_end,
                    sheet_name=sheet_name)

    return None


if __name__ == '__main__':
    gsheet_key = '1hoWafJ0FAs0cpZ2r0ZVT4dmmg6DIXbWjByFNuA9FxQ4'
    data_interval_start = "2023-01-01" 
    data_interval_end = "2024-06-30"
    scrt = os.environ["DBT_STOCKS_HOME"] + '/gsheetsconfig.json'
    gsheets_client = pygsheets.authorize(client_secret=scrt)
    sheet_name = "Input Data"

    run_gsheets_loader(gsheets_client=gsheets_client,
                    gsheet_key=gsheet_key,
                    data_interval_start=data_interval_start,
                    data_interval_end=data_interval_end,
                    sheet_name=sheet_name)

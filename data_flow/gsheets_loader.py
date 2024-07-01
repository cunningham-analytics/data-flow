
import pygsheets
import pandas as pd
import psycopg2

#authorization
gc = pygsheets.authorize(client_secret='gsheetsconfig.json')

dbUser = 'admin' # os.environ["POSTGRES_USER"]
dbPW = 'root' # os.environ["POSTGRES_PW"]

connStr = f"host=localhost dbname=analytics user={dbUser} password={dbPW}"
conn = psycopg2.connect(connStr)

cur = conn.cursor()

cur.execute(
    f"""
    select * from production_staging.int_gme__closing_calculations order by report_date
    """
)
conn.commit()

df = pd.DataFrame(cur.fetchall())

sh = gc.open_by_key('1hoWafJ0FAs0cpZ2r0ZVT4dmmg6DIXbWjByFNuA9FxQ4')

#select the first sheet 
wks = sh[0]

#update the first sheet with df, starting at cell B2. 
wks.set_dataframe(df,(1,1))
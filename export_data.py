import pandas as pd
import psycopg2

# 1. Konek ke Database Lokal
conn = psycopg2.connect(
    host="localhost",
    port="5433",
    database="f1_datawarehouse",
    user="warehouse_user",
    password="warehouse_password"
)

# 2. Ambil Data Utama
df_advanced = pd.read_sql("SELECT * FROM lap_telemetry_advanced", conn)

# 3. Simpan jadi File
df_advanced.to_csv('dashboard/f1_data_static.csv', index=False)

print("Data berhasil di-export! Cek folder dashboard kamu.")
conn.close()
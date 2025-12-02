from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit, coalesce # <-- COALESCE ditambahkan!
from pyspark.sql.window import Window
import sys

def run_etl():
    # 1. Konfigurasi
    packages = "org.mongodb.spark:mongo-spark-connector_2.12:10.4.0,org.postgresql:postgresql:42.6.0"
    spark = SparkSession.builder \
        .appName("F1_Deep_Analytics_Automated") \
        .config("spark.jars.packages", packages) \
        .master("spark://spark-master:7077") \
        .getOrCreate()

    print("Spark Session Terhubung!")

    # 2. Extract
    print("Extracting Raw Data")
    mongo_uri = "mongodb://admin:password@mongo:27017/?authSource=admin"
    df = spark.read.format("mongodb") \
        .option("spark.mongodb.read.connection.uri", mongo_uri) \
        .option("database", "f1_datalake") \
        .option("collection", "telemetry_raw") \
        .load()

    # 3. CLEANING & TAMBAL DATA KOSONG (IMPUTATION)
    print("Cleaning & Applying COALESCE logic")

    # Filter Valid Laps
    df = df.filter(col("LapTime_Sec") > 0).filter(col("LapTime_Sec").isNotNull())

    # Kita paksakan nilai default 0 atau 30 agar rumus dibawah tidak error NaN
    df = df.na.fill({
        "TrackTemp": 30.0,
        "Avg_Corner_Speed": 100.0,
        "Avg_Brake_Pressure": 0.0,
        "Full_Throttle_Pct": 0.0,
        "Max_Speed_Tel": 0.0,
        "Avg_RPM": 0.0
    })

    # FEATURE ENGINEERING (Menggunakan COALESCE di Rumus)
    
    # 1. Tire Stress Index (TIDAK ADA LAGI NaN DISINI)
    df_features = df.withColumn("Tire_Stress_Index", 
                                (coalesce(col("TrackTemp"), lit(30.0)) * col("Avg_Corner_Speed")) / 100)

    # 2. Driver Aggression
    df_features = df_features.withColumn("Driver_Aggression", 
                                         (col("Full_Throttle_Pct") * 0.7) + (col("Avg_Brake_Pressure") * 0.3))

    # 3. Weather Condition
    df_features = df_features.withColumn("Weather_Condition", 
                                         when(col("TrackTemp") > 40, "Hot")
                                         .when(col("TrackTemp") < 25, "Cold")
                                         .otherwise("Optimal"))

    # 4. LOAD KE WAREHOUSE
    print("Menyimpan Data ke PostgreSQL...")
    
    postgres_url = "jdbc:postgresql://postgres-warehouse:5432/f1_datawarehouse"
    props = {
        "user": "warehouse_user",
        "password": "warehouse_password",
        "driver": "org.postgresql.Driver"
    }

    df_features.write.jdbc(
        url=postgres_url,
        table="lap_telemetry_advanced",
        mode="overwrite",
        properties=props
    )

    print("Data sudah ditambal dengan COALESCE.")
    spark.stop()

if __name__ == "__main__":
    run_etl()
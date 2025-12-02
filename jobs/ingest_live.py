import fastf1
import pandas as pd
import numpy as np
from pymongo import MongoClient
from datetime import datetime, timedelta
import logging
import sys

# CONFIGURATION
MONGO_URI = "mongodb://admin:password@mongo:27017/?authSource=admin"
DB_NAME = "f1_datalake"
COLLECTION_NAME = "telemetry_raw"
CURRENT_YEAR = datetime.now().year

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'/opt/airflow/logs/live_ingestion_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# HELPER FUNCTIONS

def get_existing_rounds(collection, year):
    """Get list of rounds already in MongoDB for given year"""
    try:
        existing = collection.distinct("RoundNumber", {"Year": year})
        return set(existing)
    except Exception as e:
        logger.error(f"Error reading existing rounds: {e}")
        return set()

def extract_lap_data_live(lap, session_info):
    """
    Extract lap data with FULL SCHEMA (28 fields)
    Compatible with historical data structure
    """
    try:
        # Get telemetry
        tel = lap.get_telemetry()
        if tel is None or len(tel) == 0:
            return None
        
        # Calculate telemetry features (same as historical)
        try:
            full_throttle_pct = (tel[tel['Throttle'] >= 95].shape[0] / tel.shape[0]) * 100
        except:
            full_throttle_pct = 0
        
        try:
            brake_data = tel[tel['Brake'] > 0]['Brake']
            avg_brake = float(brake_data.mean()) if len(brake_data) > 0 else 0
        except:
            avg_brake = 0
        
        try:
            avg_corner = float(tel[tel['Throttle'] < 95]['Speed'].mean())
        except:
            avg_corner = 0
        
        try:
            gear_changes = int((tel['nGear'].diff() != 0).sum())
        except:
            gear_changes = 0
        
        try:
            avg_rpm = float(tel['RPM'].mean())
        except:
            avg_rpm = 0
        
        try:
            max_speed = float(tel['Speed'].max())
        except:
            max_speed = 0
        
        # Get weather
        w_air = np.nan
        w_track = np.nan
        w_humid = np.nan
        
        try:
            weather = lap.get_weather_data()
            if weather is not None and len(weather) > 0:
                w_dict = weather.iloc[-1].to_dict()
                w_air = float(w_dict.get('AirTemp', np.nan))
                w_track = float(w_dict.get('TrackTemp', np.nan))
                w_humid = float(w_dict.get('Humidity', np.nan))
        except:
            pass
        
        # Build row with FULL schema (28 fields - same as historical)
        row = {
            'Year': int(session_info['year']),
            'RoundNumber': int(session_info['round']),
            'Circuit': str(session_info['circuit']),
            'EventName': str(session_info['event_name']),
            'Driver': str(lap['Driver']),
            'Team': str(lap['Team']),
            'LapTime_Sec': float(lap['LapTime'].total_seconds()),
            'Sector1_Sec': float(lap['Sector1Time'].total_seconds()) if pd.notna(lap['Sector1Time']) else np.nan,
            'Sector2_Sec': float(lap['Sector2Time'].total_seconds()) if pd.notna(lap['Sector2Time']) else np.nan,
            'Sector3_Sec': float(lap['Sector3Time'].total_seconds()) if pd.notna(lap['Sector3Time']) else np.nan,
            'SpeedI1': float(lap['SpeedI1']) if pd.notna(lap['SpeedI1']) else np.nan,
            'SpeedI2': float(lap['SpeedI2']) if pd.notna(lap['SpeedI2']) else np.nan,
            'SpeedFL': float(lap['SpeedFL']) if pd.notna(lap['SpeedFL']) else np.nan,
            'SpeedST': float(lap['SpeedST']) if pd.notna(lap['SpeedST']) else np.nan,
            'Compound': str(lap['Compound']) if pd.notna(lap['Compound']) else 'UNKNOWN',
            'TyreLife': int(lap['TyreLife']) if pd.notna(lap['TyreLife']) else 0,
            'Full_Throttle_Pct': full_throttle_pct,
            'Avg_Brake_Pressure': avg_brake,
            'Avg_Corner_Speed': avg_corner,
            'Gear_Changes': gear_changes,
            'Avg_RPM': avg_rpm,
            'Max_Speed_Tel': max_speed,
            'TrackTemp': w_track,
            'AirTemp': w_air,
            'Humidity': w_humid,
            'SessionTime': float(lap['Time'].total_seconds())
        }
        
        return row
        
    except Exception as e:
        logger.warning(f"Failed to extract lap data: {str(e)[:100]}")
        return None

def is_race_completed(event_date):
    """Check if race is completed (at least 2 days ago to ensure data available)"""
    cutoff = datetime.now() - timedelta(days=2)
    return event_date < cutoff

# MAIN INGESTION LOGIC

def ingest_new_races():
    """Main function to check and ingest new F1 races"""
    
    logger.info(f"STARTING LIVE INGESTION CHECK FOR {CURRENT_YEAR}")    
    try:
        # Connect to MongoDB
        client = MongoClient(MONGO_URI)
        collection = client[DB_NAME][COLLECTION_NAME]
        logger.info(f"Connected to MongoDB: {DB_NAME}.{COLLECTION_NAME}")
        
        # Get existing rounds
        existing_rounds = get_existing_rounds(collection, CURRENT_YEAR)
        logger.info(f"Existing rounds in DB: {sorted(existing_rounds)}")
        
        # Get F1 schedule
        logger.info(f"Fetching F1 schedule for {CURRENT_YEAR}")
        schedule = fastf1.get_event_schedule(CURRENT_YEAR)
        
        # Filter completed races
        completed_races = schedule[
            (schedule['EventFormat'] != 'testing') & 
            (schedule['Session5DateUtc'].apply(is_race_completed))  # Use qualifying date
        ]
        
        logger.info(f"Found {len(completed_races)} completed races")
        
        # Find new races
        new_races = []
        for _, event in completed_races.iterrows():
            round_num = event['RoundNumber']
            if round_num not in existing_rounds:
                new_races.append(event)
        
        if not new_races:
            logger.info("No new races to download. System up-to-date!")
            return
        
        logger.info(f"Found {len(new_races)} NEW races to download:")
        for event in new_races:
            logger.info(f"   - Round {event['RoundNumber']}: {event['EventName']}")
        
        # Process each new race
        total_laps_inserted = 0
        
        for event in new_races:
            round_num = event['RoundNumber']
            event_name = event['EventName']
            
            logger.info(f"\n{'='*70}")
            logger.info(f"DOWNLOADING: Round {round_num} - {event_name}")
            logger.info(f"{'='*70}")
            
            try:
                # Load session (Qualifying)
                session = fastf1.get_session(CURRENT_YEAR, round_num, 'Q')
                logger.info("   Loading telemetry & weather data...")
                session.load(telemetry=True, weather=True, messages=False)
                
                laps = session.laps
                logger.info(f"   Raw laps: {len(laps)}")
                
                # Filter valid laps (same logic as historical)
                wet_compounds = ['INTERMEDIATE', 'WET']
                laps_dry = laps[~laps['Compound'].isin(wet_compounds)].copy()
                laps_valid = laps_dry[laps_dry['LapTime'].notna()].copy()
                
                logger.info(f"     Dry laps: {len(laps_dry)}")
                logger.info(f"    Valid laps: {len(laps_valid)}")
                
                if len(laps_valid) == 0:
                    logger.warning(f"   No valid laps found for {event_name}")
                    continue
                
                # Extract lap data
                session_info = {
                    'year': CURRENT_YEAR,
                    'round': round_num,
                    'circuit': event['Location'],
                    'event_name': event_name
                }
                
                extracted_rows = []
                errors = 0
                
                logger.info(f"   Processing {len(laps_valid)} laps")
                
                for idx, lap in laps_valid.iterrows():
                    row = extract_lap_data_live(lap, session_info)
                    if row is not None:
                        extracted_rows.append(row)
                    else:
                        errors += 1
                
                logger.info(f"   Extracted: {len(extracted_rows)} laps")
                if errors > 0:
                    logger.info(f"     Skipped: {errors} laps (telemetry errors)")
                
                # Insert to MongoDB
                if extracted_rows:
                    # Check for duplicates before insert
                    result = collection.insert_many(extracted_rows, ordered=False)
                    inserted_count = len(result.inserted_ids)
                    total_laps_inserted += inserted_count
                    
                    logger.info(f"    Inserted {inserted_count} laps to MongoDB!")
                else:
                    logger.warning(f"    No laps to insert for {event_name}")
                
            except Exception as e:
                logger.error(f"    Failed to download {event_name}: {str(e)}")
                continue
        
        # Final summary
        logger.info(f" INGESTION SUMMARY")
        logger.info(f" New races processed: {len(new_races)}")
        logger.info(f" Total laps inserted: {total_laps_inserted:,}")
        
        client.close()
        
    except Exception as e:
        logger.error(f" FATAL ERROR: {str(e)}")
        raise

# EXECUTE

if __name__ == "__main__":
    try:
        ingest_new_races()
        logger.info(" Live ingestion completed successfully!")
        sys.exit(0)
    except Exception as e:
        logger.error(f" Live ingestion failed: {e}")
        sys.exit(1)
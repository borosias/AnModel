# one_time_trends_loader.py
import os
import pandas as pd
from datetime import datetime, timedelta
import json
from serpapi import GoogleSearch
from pathlib import Path
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _parse_results(results: dict, queries: list, region: str) -> list:
    """Parse results from SerpAPI with correct date format"""
    trends = []

    def _parse_date(date_str_raw: str):
        """
        Support formats:
          - 'Jun 4, 2025'
          - 'Dec 1–7, 2024'
          - 'Dec 29, 2024–Jan 4, 2025'
        Take the first date of the interval.
        """
        if not date_str_raw:
            return None

        # Normalize strange unicode characters (narrow space, long dash, etc.)
        s = (
            date_str_raw
            .replace('\u2009', ' ')  # narrow no‑break space
            .replace('\u2011', '-')  # non‑breaking hyphen
            .replace('\u2012', '-')  # figure dash
            .replace('\u2013', '-')  # en dash
            .replace('\u2014', '-')  # em dash
            .strip()
        )

        # 1) Try as a regular date 'Jun 4, 2025'
        try:
            return datetime.strptime(s, '%b %d, %Y').date()
        except ValueError:
            pass

        # 2) Popular format 'Dec 1-7, 2024'
        #    Take the first part before the hyphen, plus the year on the right.
        try:
            # s can be 'Dec 1-7, 2024' or 'Dec 29, 2024-Jan 4, 2025'
            # Take the year from the end of the string (last 4 digits)
            year = s[-4:]
            if not year.isdigit():
                raise ValueError

            # Part before the comma
            if ',' in s:
                left_part = s.split(',', 1)[0]  # 'Dec 1-7' or 'Dec 29, 2024-Jan 4'
            else:
                left_part = s

            # If there is a hyphen, take everything to the left of it ('Dec 1')
            if '-' in left_part:
                left_part = left_part.split('-', 1)[0].strip()

            candidate = f"{left_part}, {year}"  # 'Dec 1, 2024'
            return datetime.strptime(candidate, '%b %d, %Y').date()
        except Exception:
            logger.warning(f"⚠️ Cannot parse date: {date_str_raw}")
            return None

    try:
        timeline_data = results.get("interest_over_time", {}).get("timeline_data", [])

        print(f"📅 Found {len(timeline_data)} timeline entries")

        for day_data in timeline_data:
            date_str = day_data.get("date")
            if not date_str:
                continue

            date = _parse_date(date_str)
            if date is None:
                continue

            # For each query in the batch
            values = day_data.get("values", [])
            print(f"📈 Date: {date}, Values count: {len(values)}")

            for i, query in enumerate(queries):
                if i < len(values):
                    popularity = values[i].get("value", 0)
                    formatted_value = values[i].get("formattedValue", "0")
                else:
                    popularity = 0
                    formatted_value = "0"

                trends.append({
                    'query': query,
                    'region': region,
                    'date': date,
                    'popularity': int(popularity) if popularity else 0,
                    'formatted_value': formatted_value,
                    'loaded_at': datetime.now()
                })

    except Exception as e:
        logger.error(f"❌ Parse error for {queries} in {region}: {e}")
        import traceback
        traceback.print_exc()

    return trends


class OneTimeTrendsLoader:
    def __init__(self, storage_path: str = "./trends_data"):
        self.api_key = os.getenv("SERPAPI_KEY", "e0905b78baa8db75444b707477b51b353782a44bd35d7fde188817b55fb89d45")
        if not self.api_key:
            raise ValueError("SERPAPI_KEY not found!")

        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)

        self.queries = ["shoes", "blanket", "phone", "charger", "jacket", "backpack", "watch"]
        self.regions = ["UA-30", "UA-40", "UA-50"]

    def load_all_trends_once(self, days_back: int = 365):
        """Load all trends once and save to parquet"""

        # Check if already loaded
        master_file = self.storage_path / "trends_master.parquet"
        if master_file.exists():
            logger.info("Trends already loaded! Reading from existing file...")
            return pd.read_parquet(master_file)

        logger.info(f"Loading trends for {days_back} days...")

        all_trends = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        date_range = f"{start_date.strftime('%Y-%m-%d')} {end_date.strftime('%Y-%m-%d')}"

        for region in self.regions:
            for i in range(0, len(self.queries), 5):  # Group by 5 queries
                batch_queries = self.queries[i:i + 5]
                queries_str = ",".join(batch_queries)

                logger.info(f"Loading {queries_str} for {region}...")

                try:
                    params = {
                        "engine": "google_trends",
                        "q": queries_str,
                        "geo": region,
                        "data_type": "TIMESERIES",
                        "date": date_range,
                        "api_key": self.api_key
                    }

                    search = GoogleSearch(params)
                    results = search.get_dict()

                    print(f"📊 Raw API response keys: {results.keys()}")

                    trends_data = _parse_results(results, batch_queries, region)
                    all_trends.extend(trends_data)

                    logger.info(f"✅ Got {len(trends_data)} records for this batch")

                    # Pause between queries
                    time.sleep(2)

                except Exception as e:
                    logger.error(f"Failed for {queries_str} in {region}: {e}")
                    continue

        # Save to parquet
        if all_trends:
            df = pd.DataFrame(all_trends)
            df = self._add_features(df)

            # NORMALIZE SCHEMA FOR parquet
            df['date'] = pd.to_datetime(df['date']).dt.normalize()
            df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce').fillna(0).astype(int)

            # Ensure base columns are present
            base_cols = ['date', 'query', 'region', 'popularity']
            for col in base_cols:
                if col not in df.columns:
                    logger.warning(f"Missing column '{col}' in trends data — adding empty")
                    df[col] = None

            master_file = self.storage_path / "trends_master.parquet"
            df.to_parquet(master_file, index=False)
            logger.info(f"✅ Saved {len(df)} records to {master_file}")

            return df
        else:
            logger.error("❌ No data loaded!")
            return pd.DataFrame()

    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ML features"""
        if df.empty:
            return df

        # Sort for window functions
        df = df.sort_values(['query', 'region', 'date'])

        result_dfs = []

        for (query, region), group in df.groupby(['query', 'region']):
            group = group.copy().sort_values('date')

            # Moving averages
            group['popularity_7d_avg'] = group['popularity'].rolling(7, min_periods=1).mean()
            group['popularity_30d_avg'] = group['popularity'].rolling(30, min_periods=1).mean()

            # Changes
            group['popularity_change_1d'] = group['popularity'].pct_change().fillna(0)
            group['popularity_change_7d'] = group['popularity'].pct_change(7).fillna(0)

            # Simple trend (difference between first and last)
            if len(group) > 1:
                group['trend_slope'] = (group['popularity'].iloc[-1] - group['popularity'].iloc[0]) / len(group)
            else:
                group['trend_slope'] = 0

            result_dfs.append(group)

        return pd.concat(result_dfs, ignore_index=True)

    def get_trends_data(self) -> pd.DataFrame:
        """Just return data from parquet"""
        master_file = self.storage_path / "trends_master.parquet"

        if not master_file.exists():
            logger.warning("No trends data found! Run load_all_trends_once() first.")
            return pd.DataFrame()

        return pd.read_parquet(master_file)


# Simple usage
if __name__ == "__main__":
    print("🚀 Starting trends loader...")

    # Then the main load:
    loader = OneTimeTrendsLoader()
    df = loader.load_all_trends_once(days_back=365)

    if not df.empty:
        print(f"✅ Successfully loaded {len(df)} trend records")
        print(f"📊 Data shape: {df.shape}")
        print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"🔍 Queries: {df['query'].unique().tolist()}")
        print(f"🌍 Regions: {df['region'].unique().tolist()}")

        # Show data sample
        print("\n📋 Sample data:")
        print(df.head(10))

        # Save statistics
        stats_file = Path("./trends_data/loading_stats.json")
        stats = {
            'loaded_at': datetime.now().isoformat(),
            'total_records': len(df),
            'date_range': {
                'start': df['date'].min().isoformat(),
                'end': df['date'].max().isoformat()
            },
            'queries': df['query'].unique().tolist(),
            'regions': df['region'].unique().tolist()
        }

        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        print("📊 Stats saved to loading_stats.json")
    else:
        print("❌ Failed to load trends data")
        print("💡 Try running debug_load() first to see what's happening")

from src.data_access.email_data import EmailData
from src.data_access.load_data import LoadData

from src.utils.logger import logger


logger.info("🚀 Demo started")


# ─────────────────────────────────────────────
# Step 1 → Insert CSV into PostgreSQL
# ─────────────────────────────────────────────
logger.info(
    "📂 Starting CSV insertion process"
)

email_obj = EmailData()

email_obj.insert_csv_to_postgres(
    "data/raw/archive/enron_spam_data.csv"
)

logger.info(
    "✅ CSV insertion process completed"
)


# ─────────────────────────────────────────────
# Step 2 → Fetch data from PostgreSQL
# ─────────────────────────────────────────────
logger.info(
    "📥 Starting data loading process"
)

load_obj = LoadData()

df = load_obj.fetch_data()

logger.info(
    "✅ Data fetched successfully"
)

logger.info("🏁 Demo completed")
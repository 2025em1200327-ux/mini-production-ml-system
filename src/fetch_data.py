"""Download the public IBM Telco churn CSV and verify its checksum, so every run uses identical data."""
import hashlib
import logging
import sys
import urllib.request

from src.config import DATA_DIR, DATA_SHA256, DATA_URL, RAW_FILE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fetch_data")


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not RAW_FILE.exists():
        log.info("Downloading %s", DATA_URL)
        urllib.request.urlretrieve(DATA_URL, RAW_FILE)
    digest = sha256(RAW_FILE)
    if digest != DATA_SHA256:
        log.error("Checksum mismatch: expected %s, got %s", DATA_SHA256, digest)
        return 1
    log.info("Data ready at %s (sha256 verified)", RAW_FILE)
    return 0


if __name__ == "__main__":
    sys.exit(main())

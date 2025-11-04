import os
from pathlib import Path
from dotenv import load_dotenv
import logging

log = logging.getLogger(__name__)

# loading variables from .env file
load_dotenv(override=True)


def get_persistent_storage():
    """This function returns a path to the persistent storage directory."""
    if "PERSISTENT_STORE" in os.environ:
        path = Path(os.environ["PERSISTENT_STORE"])
        if not path.exists():
            raise FileNotFoundError(f"Path '{path}' does not exist.")
    else:
        path = os.path.join(os.getcwd(), "storage")
        os.makedirs(path, exist_ok=True)

    log.info(f"Using persistent storage directory: {path}")
    return path

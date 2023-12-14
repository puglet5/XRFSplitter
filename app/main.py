import logging
import logging.config

from app.ui import UserInterface

logging.config.fileConfig(fname=r"logging.conf", disable_existing_loggers=False)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    main = UserInterface()
    main.start()

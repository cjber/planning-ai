from loguru import logger

logger.add("logs/model.log", rotation="1 MB", retention="10 days", level="DEBUG")

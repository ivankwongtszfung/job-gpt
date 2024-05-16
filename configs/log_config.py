import logging.config


def setup_logging():
    logging.config.fileConfig(
        fname="configs/logging.ini", disable_existing_loggers=False
    )


def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Logging is configured and the application is starting.")


if __name__ == "__main__":
    main()

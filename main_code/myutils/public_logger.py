import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
directory = os.path.join(Path(__file__).resolve().parent.parent,"climbing_logs")
if not os.path.exists(directory):
    os.makedirs(directory)

logging_reserved_path = os.path.join(directory,"log.txt")
logger = logging.getLogger("PublicLogger")
logger.setLevel(level=logging.INFO)
formatter = logging.Formatter('%(levelname)s %(asctime)s %(filename)s %(funcName)s line=%(lineno)d %(message)s',datefmt="%y-%m-%d %H:%M:%S")
formatter2 = logging.Formatter('%(message)s',datefmt="%y-%m-%d %H:%M:%S")
file_handler = RotatingFileHandler(logging_reserved_path, maxBytes=3*1024*1024, backupCount=10)#3M,10个日志文件
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter2)
file_handler.encoding("utf-8")
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter2)

logger.addHandler(file_handler)
logger.addHandler(console)


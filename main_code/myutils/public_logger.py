import logging
from logging.handlers import RotatingFileHandler
import os
if not os.path.exists("./climbing_logs/"):
    os.mkdir("./climbing_logs")
logging_reserved_path = "./climbing_logs/log.txt"
logger = logging.getLogger("PublicLogger")
logger.setLevel(level=logging.INFO)
formatter = logging.Formatter('%(levelname)s %(asctime)s %(filename)s %(funcName)s line=%(lineno)d %(message)s',datefmt="%y-%m-%d %H:%M:%S")
formatter2 = logging.Formatter('%(message)s',datefmt="%y-%m-%d %H:%M:%S")
file_handler = RotatingFileHandler(logging_reserved_path, maxBytes=3*1024*1024, backupCount=10)#3M,10个日志文件
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter2)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter2)

logger.addHandler(file_handler)
logger.addHandler(console)


from asyncio.log import logger
import logging
import os

class Logger():
    def __init__(self, log_name, log_dir) -> None:
        
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        fh = logging.FileHandler(filename = log_dir + log_name, encoding = 'utf-8', mode = 'a')
        fh.setLevel(logging.INFO)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s-%(name)s-%(filename)s-[line:%(lineno)d]''-%(levelname)s-[context]: %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)    
        pass
    
    def log_insert(self, msg, type):
        if type is logging.DEBUG:
            logger.debug(msg)
        elif type is logging.INFO:
            logger.info(msg)
        elif type is logging.WARNING:
            logger.warning(msg)
        elif type is logging.ERROR:
            logger.error(msg)
        elif type is logging.CRITICAL:
            logger.critical(msg)
import logging
from pathlib import Path
import time


def set_logger(model, dataset, prefix, to_file):
  logger = logging.getLogger()
  logger.setLevel(logging.DEBUG)
  Path("log/{}".format(model)).mkdir(parents=True, exist_ok=True)
  timenow = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
  ch = logging.StreamHandler()
  ch.setLevel(logging.INFO)
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  ch.setFormatter(formatter)
  logger.addHandler(ch)
  if to_file:
    fh = logging.FileHandler('log/{}/{}.log'.format(model, timenow + str(prefix)))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
  logger.info("Model:{}, Dataset:{}".format(model, dataset))
  return logger, timenow


def get_checkpoint_path(model, timenow, task, uml):
  return "log/{}/{}/checkpoints/task{}.pth".format(model, timenow, task), "log/{}/{}/checkpoints/task{}_mem.pth".format(model, timenow, task),"log/{}/{}/checkpoints/task{}_IB.pth".format(model, timenow, task), "log/{}/{}/checkpoints/task{}_PGen.pth".format(model, timenow, task)

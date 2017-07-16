# coding=utf-8

# 日志配置
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

# 数据路径
data_dirpath = "../data/2016E27_V2/data/"
source_dirpath = data_dirpath + "source/"
ere_dirpath = data_dirpath + "ere/"
annotation_dirpath = data_dirpath + "annotation/"
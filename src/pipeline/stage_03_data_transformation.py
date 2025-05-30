from src.config.configuration import ConfigurationManager
from src.components.data_transformation import DataTransformation
from src.my_logging.my_logger import logging

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):    
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.convert_data()
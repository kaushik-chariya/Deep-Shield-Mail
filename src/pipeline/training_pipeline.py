import sys

from src.utils.logger import logger
from src.utils.exception import MyException

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
# from src.components.model_trainer import ModelTrainer
# from src.components.model_evaluation import ModelEvaluation
# from src.components.model_pusher import ModelPusher

from src.entity.config_entity import ( DataIngestionConfig,
                                      DataValidationConfig,
                                      DataTransformationConfig)
#                                       ModelTrainerConfig,
#                                       ModelEvaluationConfig,
#                                       ModelPusherConfig
# )

from src.entity.artifact_entity import ( DataIngestionArtifact,
                                        DataValidationArtifact,
                                        DataTransformationArtifact)
                                        # ModelTrainerArtifact,
                                        # ModelEvaluationArtifact,
                                        # ModelPusherArtifact
# )


class TrainPipeline:

    def __init__(self):

        self.data_ingestion_config = DataIngestionConfig()

        self.data_validation_config = DataValidationConfig()

        self.data_transformation_config = DataTransformationConfig()

        # self.model_trainer_config = ModelTrainerConfig()

        # self.model_evaluation_config = ModelEvaluationConfig()

        # self.model_pusher_config = ModelPusherConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        This method of TrainPipeline class
        is responsible for starting
        data ingestion component
        """

        try:

            logger.info( "Entered start_data_ingestion method of TrainPipeline class" )

            logger.info( "Getting data from PostgreSQL" )

            data_ingestion = DataIngestion( data_ingestion_config=self.data_ingestion_config)

            data_ingestion_artifact = ( data_ingestion. initiate_data_ingestion())

            logger.info( "Got train_set and test_set from PostgreSQL")

            logger.info( "Exited start_data_ingestion method of TrainPipeline class")

            return data_ingestion_artifact

        except Exception as e:
            raise MyException(e, sys) from e
        
    def start_data_validation(
        self,
        data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        """
        This method of TrainPipeline class
        is responsible for starting
        data validation component
        """

        logger.info(
            "Entered start_data_validation method "
            "of TrainPipeline class"
        )

        try:

            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=self.data_validation_config
            )

            data_validation_artifact = (
                data_validation.initiate_data_validation()
            )

            logger.info(
                "Performed data validation successfully"
            )

            logger.info(
                "Exited start_data_validation method "
                "of TrainPipeline class"
            )

            return data_validation_artifact

        except Exception as e:
            raise MyException(e, sys) from e
        

        
    def start_data_transformation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact
    ) -> DataTransformationArtifact:
        """
        This method of TrainPipeline class
        is responsible for starting
        data transformation component
        """

        try:

            data_transformation = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact,
                data_transformation_config=
                self.data_transformation_config
            )

            data_transformation_artifact = (
                data_transformation.initiate_data_transformation()
            )

            logger .info(
                "Performed data transformation successfully"
            )

            return data_transformation_artifact

        except Exception as e:
            raise MyException(e, sys)













    
    def run_pipeline(self) -> None:
        """ This method of TrainPipeline class is responsible for running complete training pipeline """

        try:

            logger.info( "Starting complete training pipeline 🏃🏻‍♂️")

            data_ingestion_artifact = ( self.start_data_ingestion() )

            data_validation_artifact = ( self.start_data_validation( data_ingestion_artifact=data_ingestion_artifact ))

            data_transformation_artifact = ( self.start_data_transformation( data_ingestion_artifact=data_ingestion_artifact,data_validation_artifact=data_validation_artifact))

            # model_trainer_artifact = (self.start_model_trainer(data_transformation_artifact=data_transformation_artifact))

            # model_evaluation_artifact = (self.start_model_evaluation(data_ingestion_artifact=data_ingestion_artifact,
            # model_trainer_artifact=model_trainer_artifact))

            # if not model_evaluation_artifact.is_model_accepted:

            #     logger.info("Model not accepted 😕")
            #     return None

            # self.start_model_pusher(model_evaluation_artifact=model_evaluation_artifact)

            # logger.info("Training pipeline completed successfully ✅")

        except Exception as e:
            raise MyException(e, sys) 
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.trainer import Trainer
from pose_estimation.pl_wrappers import PoseHeatmapLightningModel, Pose3DLightningModel

from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


if __name__ == "__main__":

    LightningCLI()
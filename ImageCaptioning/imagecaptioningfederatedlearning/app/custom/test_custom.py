from unittest.mock import patch

import pytest
from imagecaptiontrainer import ImageCaptionTrainer
from imagecaptionvalidator import ImageCaptionValidator
import appConstants

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.signal import Signal


@pytest.fixture()
def get_cifar_trainer():
    with patch.object(ImageCaptionTrainer, "_save_local_model") as mock_save:
        with patch.object(ImageCaptionTrainer, "_load_local_model") as mock_load:
            yield ImageCaptionTrainer()


class TestImageCaptionTrainer:
    @pytest.mark.parametrize("num_rounds", [1, 3])
    def test_execute(self, get_cifar_trainer, num_rounds):
        trainer = get_cifar_trainer
        # just take first batch
        iterator = iter(trainer._train_loader)
        trainer._train_loader = [next(iterator)]

        dxo = DXO(data_kind=DataKind.WEIGHTS, data=trainer.model.state_dict())
        result = dxo.to_shareable()
        for i in range(num_rounds):
            result = trainer.execute(appConstants.TRAIN_TASK_NAME, shareable=result, fl_ctx=FLContext(), abort_signal=Signal())
            assert result.get_return_code() == ReturnCode.OK

    @patch.object(ImageCaptionTrainer, "_save_local_model")
    @patch.object(ImageCaptionTrainer, "_load_local_model")
    def test_execute_rounds(self, mock_save, mock_load):
        train_task_name = "train"
        trainer = ImageCaptionTrainer()
        # just take first batch
        myitt = iter(trainer.training_setup["train_loader"])
        trainer.training_setup["train_loader"] = [next(myitt)]

        dxo = DXO(data_kind=DataKind.WEIGHTS, data=trainer.training_setup["model"].state_dict())
        result = dxo.to_shareable()
        for i in range(3):
            result = trainer.execute(train_task_name, shareable=result, fl_ctx=FLContext(), abort_signal=Signal())
            assert result.get_return_code() == ReturnCode.OK


class TestImageCaptionValidator:
    def test_execute(self):
        validate_task_name = "validate"
        validator = ImageCaptionValidator()
        # just take first batch
        iterator = iter(validator.validating_setup['val_loader'])
        validator.validating_setup['val_loader'] = [next(iterator)]

        dxo = DXO(data_kind=DataKind.WEIGHTS, data=validator.model.state_dict())
        result = validator.execute(
            validate_task_name, shareable=dxo.to_shareable(), fl_ctx=FLContext(), abort_signal=Signal()
        )
        assert result.get_return_code() == ReturnCode.OK
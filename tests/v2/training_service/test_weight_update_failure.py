# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from areal.v2.training_service.controller.controller import GatewayTrainController


@pytest.mark.parametrize("status", ["ok", "error"])
def test_weight_update_resumes_only_after_success(status):
    controller = GatewayTrainController.__new__(GatewayTrainController)
    controller.rollout = Mock()
    controller._weight_update_ctrl = Mock()
    controller._weight_update_ctrl.update_weights.return_value = SimpleNamespace(
        status=status, error="reload failed", duration_ms=1.0
    )

    if status == "error":
        with pytest.raises(RuntimeError, match="reload failed"):
            controller.update_weights(SimpleNamespace(version=1))
        controller.rollout.continue_generation.assert_not_called()
    else:
        controller.update_weights(SimpleNamespace(version=1))
        controller.rollout.continue_generation.assert_called_once()
    controller.rollout.pause_generation.assert_called_once()

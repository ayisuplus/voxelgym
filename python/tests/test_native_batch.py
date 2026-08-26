import numpy as np
import pytest

import voxelgym_rs as rs


IDLE = (0, 0, 0, 0, 4, 0, 0, 0, 0, 0)


def make_batch():
    return rs.PyWorldBatch([(10, "void"), (11, "flat")])


def test_batch_constructor_and_observation_contracts():
    with pytest.raises(ValueError, match="unknown preset"):
        rs.PyWorldBatch([(1, "missing")])

    batch = make_batch()
    assert batch.len() == 2
    assert len(batch.hashes()) == 2
    assert batch.obs_voxels_batch().shape == (2, 21, 11, 21)
    assert batch.obs_inventory_batch().shape == (2, 36, 2)
    assert batch.obs_pose_batch().shape == (2, 6)
    assert batch.obs_raycast_batch().shape == (2, 2)


def test_batch_tuple_and_numpy_steps_cover_contiguous_and_strided_rows():
    batch = make_batch()

    assert batch.step_batch([IDLE, IDLE]) == [False, False]
    actions = np.zeros((2, 20), dtype=np.uint8)[:, ::2]
    actions[:, 4] = 4
    assert not actions.flags.c_contiguous
    assert batch.step_batch_np(actions) == [False, False]


@pytest.mark.parametrize(
    "actions",
    [
        np.zeros((2, 9), dtype=np.uint8),
        np.zeros((1, 10), dtype=np.uint8),
        np.zeros((3, 10), dtype=np.uint8),
    ],
)
def test_batch_numpy_step_rejects_wrong_shape(actions):
    batch = make_batch()

    with pytest.raises(ValueError, match=r"\(2, 10\)"):
        batch.step_batch_np(actions)


def test_batch_tuple_step_rejects_wrong_world_count():
    batch = make_batch()

    with pytest.raises(ValueError, match="2 actions"):
        batch.step_batch([IDLE])

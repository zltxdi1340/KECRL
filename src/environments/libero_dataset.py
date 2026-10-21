"""Role-restricted, lazy sequence access to pinned LIBERO demonstrations."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset


AccessPhase = Literal["development", "post_method_freeze"]


@dataclass(frozen=True)
class LiberoDemonstrationAccess:
    task_id: int
    task_name: str
    task_role: str
    demonstration_role: str
    demonstration_indices: tuple[int, ...]
    access_phase: AccessPhase
    locked_before_method_freeze: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LiberoSequenceWindow:
    demonstration_index: int
    start: int
    valid_length: int


def resolve_demonstration_access(
    protocol: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
    *,
    task_id: int,
    demonstration_role: str,
    access_phase: AccessPhase = "development",
) -> LiberoDemonstrationAccess:
    """Resolve one allowed demo partition before any HDF5 file is opened."""
    if access_phase not in {"development", "post_method_freeze"}:
        raise ValueError("unknown LIBERO access phase")
    if protocol.get("formal_result") is not False or dataset_manifest.get("formal_result") is not False:
        raise ValueError("role-restricted access requires non-formal candidate manifests")
    dataset_rows = {int(row["task_id"]): row for row in dataset_manifest.get("files", ())}
    if task_id not in dataset_rows:
        raise ValueError("task is absent from the dataset manifest")
    roles = protocol.get("task_roles", {})
    matching_roles = [role for role, task_ids in roles.items() if task_id in task_ids]
    if len(matching_roles) != 1:
        raise ValueError("task must have exactly one transfer-protocol role")
    task_role = matching_roles[0]
    partition = protocol.get("demonstration_indices_by_task_role", {}).get(task_role, {})
    if demonstration_role not in partition:
        raise ValueError("demonstration role is not valid for the task role")
    indices = tuple(int(index) for index in partition[demonstration_role])
    if not indices or len(indices) != len(set(indices)) or any(index < 0 for index in indices):
        raise ValueError("demonstration partition must contain unique non-negative indices")
    expected_count = int(dataset_manifest.get("dataset", {}).get("expected_demo_count_per_task", 0))
    if expected_count < 1 or any(index >= expected_count for index in indices):
        raise ValueError("demonstration partition exceeds the pinned dataset count")
    access_key = f"{task_role}.{demonstration_role}"
    locked = access_key in set(
        protocol.get("locked_access", {}).get("before_method_freeze", ())
    )
    if access_phase == "development" and locked:
        raise PermissionError(f"demonstration partition is locked before method freeze: {access_key}")
    return LiberoDemonstrationAccess(
        task_id=task_id,
        task_name=str(dataset_rows[task_id]["task_name"]),
        task_role=task_role,
        demonstration_role=demonstration_role,
        demonstration_indices=indices,
        access_phase=access_phase,
        locked_before_method_freeze=locked,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class RoleRestrictedLiberoSequenceDataset(Dataset):
    """Lazy fixed-length windows from one protocol-authorized task partition."""

    def __init__(
        self,
        protocol: Mapping[str, Any],
        dataset_manifest: Mapping[str, Any],
        dataset_root: Path,
        *,
        task_id: int,
        demonstration_role: str,
        sequence_length: int,
        stride: int,
        access_phase: AccessPhase = "development",
        verify_sha256: bool = True,
    ) -> None:
        if sequence_length < 1 or stride < 1:
            raise ValueError("sequence length and stride must be positive")
        self.access = resolve_demonstration_access(
            protocol,
            dataset_manifest,
            task_id=task_id,
            demonstration_role=demonstration_role,
            access_phase=access_phase,
        )
        rows = {int(row["task_id"]): row for row in dataset_manifest["files"]}
        row = rows[task_id]
        root = Path(dataset_root).resolve()
        self.path = (root / row["path"]).resolve()
        if not self.path.is_relative_to(root):
            raise ValueError("LIBERO demonstration path escapes the dataset root")
        if self.path.stat().st_size != int(row["size_bytes"]):
            raise ValueError("LIBERO demonstration file size differs from pinned metadata")
        if verify_sha256 and _sha256(self.path) != row["sha256"]:
            raise ValueError("LIBERO demonstration SHA-256 differs from pinned metadata")
        self.sha256_verified = bool(verify_sha256)
        self.sequence_length = int(sequence_length)
        self.stride = int(stride)
        self._handle = None
        self.windows = self._index_windows()
        # Do not carry an HDF5 handle across DataLoader worker forks. Each
        # process reopens the pinned file lazily on its first item access.
        self.close()
        if not self.windows:
            raise ValueError("authorized demonstration partition contains no sequence windows")

    def _open(self):
        import h5py

        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def _index_windows(self) -> tuple[LiberoSequenceWindow, ...]:
        windows = []
        data = self._open()["data"]
        for demonstration_index in self.access.demonstration_indices:
            name = f"demo_{demonstration_index}"
            if name not in data:
                raise ValueError(f"authorized demonstration is absent: {name}")
            length = int(data[name]["actions"].shape[0])
            if length < 1:
                raise ValueError("authorized demonstration must be nonempty")
            for start in range(0, length, self.stride):
                windows.append(
                    LiberoSequenceWindow(
                        demonstration_index=demonstration_index,
                        start=start,
                        valid_length=min(self.sequence_length, length - start),
                    )
                )
        return tuple(windows)

    @property
    def access_record(self) -> dict[str, Any]:
        return {
            **self.access.to_dict(),
            "dataset_path": str(self.path.resolve()),
            "dataset_sha256_verified": self.sha256_verified,
            "sequence_length": self.sequence_length,
            "stride": self.stride,
            "window_count": len(self.windows),
        }

    def __len__(self) -> int:
        return len(self.windows)

    @staticmethod
    def _padded(values: np.ndarray, length: int, valid_length: int) -> np.ndarray:
        result = np.zeros((length, *values.shape[1:]), dtype=values.dtype)
        result[:valid_length] = values
        return result

    def __getitem__(self, index: int) -> dict[str, Any]:
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self):
            raise IndexError("LIBERO sequence window index is out of range")
        window = self.windows[index]
        demo = self._open()["data"][f"demo_{window.demonstration_index}"]
        stop = window.start + window.valid_length
        obs = demo["obs"]
        agentview = np.asarray(obs["agentview_rgb"][window.start:stop])
        eye_in_hand = np.asarray(obs["eye_in_hand_rgb"][window.start:stop])
        joint_states = np.asarray(obs["joint_states"][window.start:stop], dtype=np.float32)
        gripper_states = np.asarray(obs["gripper_states"][window.start:stop], dtype=np.float32)
        actions = np.asarray(demo["actions"][window.start:stop], dtype=np.float32)
        expected = {
            "agentview_rgb": (window.valid_length, 128, 128, 3),
            "eye_in_hand_rgb": (window.valid_length, 128, 128, 3),
            "joint_states": (window.valid_length, 7),
            "gripper_states": (window.valid_length, 2),
            "actions": (window.valid_length, 7),
        }
        actual = {
            "agentview_rgb": agentview.shape,
            "eye_in_hand_rgb": eye_in_hand.shape,
            "joint_states": joint_states.shape,
            "gripper_states": gripper_states.shape,
            "actions": actions.shape,
        }
        if actual != expected:
            raise ValueError("LIBERO sequence fields do not match the Policy interface")
        if agentview.dtype != np.uint8 or eye_in_hand.dtype != np.uint8:
            raise ValueError("LIBERO sequence RGB fields must be uint8")
        if not all(
            np.isfinite(values).all()
            for values in (joint_states, gripper_states, actions)
        ):
            raise ValueError("LIBERO sequence contains non-finite proprioception or actions")
        if np.any(actions < -1.0) or np.any(actions > 1.0):
            raise ValueError("LIBERO sequence action lies outside [-1, 1]")
        agentview = self._padded(agentview, self.sequence_length, window.valid_length)
        eye_in_hand = self._padded(eye_in_hand, self.sequence_length, window.valid_length)
        joint_states = self._padded(joint_states, self.sequence_length, window.valid_length)
        gripper_states = self._padded(gripper_states, self.sequence_length, window.valid_length)
        actions = self._padded(actions, self.sequence_length, window.valid_length)
        valid_mask = np.zeros(self.sequence_length, dtype=np.bool_)
        valid_mask[: window.valid_length] = True
        return {
            "agentview_rgb": torch.from_numpy(agentview).permute(0, 3, 1, 2),
            "eye_in_hand_rgb": torch.from_numpy(eye_in_hand).permute(0, 3, 1, 2),
            "joint_states": torch.from_numpy(joint_states),
            "gripper_states": torch.from_numpy(gripper_states),
            "actions": torch.from_numpy(actions),
            "valid_mask": torch.from_numpy(valid_mask),
            "task_id": self.access.task_id,
            "demonstration_index": window.demonstration_index,
            "start_index": window.start,
        }

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle is not None:
            handle.close()
            self._handle = None

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_handle"] = None
        return state

    def __del__(self) -> None:
        self.close()

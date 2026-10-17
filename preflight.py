"""Static preflight for declared model nodes and dataset contracts.

The module has five sections: dataset formats, TensorFlow Record parsing,
competition fixtures, source rules, and command-line entry points. ``check``
applies the controlled-corpus rules; ``check_source`` checks syntax and bounded
static resource hazards in arbitrary agent code. Neither function imports or
executes candidate code.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import csv
import importlib
import importlib.util
import io
import json
import math
import os
import pickletools
import struct
import subprocess
import sys
import time
import wave
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
BUILTINS = set(dir(builtins))
BUILTINS.add("__file__")
FUNCTIONAL = {
    "relu",
    "gelu",
    "avg_pool2d",
    "interpolate",
    "elu",
    "scaled_dot_product_attention",
    "cross_entropy",
    "softmax",
    "one_hot",
}


# Dataset contracts and format-specific validators.
@dataclass(frozen=True)
class DatasetContract:
    kind: str
    classes: int = 2
    min_rows: int = 25
    feature_columns: tuple[str, ...] = ()
    label_column: str = "label"
    text_column: str = "text"
    image_size: tuple[int, int] = (16, 16)
    image_mode: str = "RGB"
    sample_rate: int = 8000
    channels: int = 1
    sample_width: int = 2
    frame_count: int = 256


Checker = Callable[[Path, DatasetContract], set[str]]
CHECKERS: dict[str, Checker] = {}


def register_checker(kind: str):
    def register(checker: Checker) -> Checker:
        if kind in CHECKERS:
            raise ValueError(f"duplicate dataset checker: {kind}")
        CHECKERS[kind] = checker
        return checker

    return register


def dataset_issues(path: Path, contract: DatasetContract, cache: dict) -> set[str]:
    """Validate one dataset under its declared contract, caching within a run."""
    path = Path(path)
    key = (path.resolve(), contract)
    if key not in cache:
        try:
            checker = CHECKERS[contract.kind]
        except KeyError as exc:
            raise ValueError(f"unsupported dataset kind: {contract.kind}") from exc
        cache[key] = checker(path, contract)
    return cache[key]


class DatasetValidator:
    """Validate one declared dataset using a format-specific check."""

    def __init__(self, path: Path, contract: DatasetContract):
        self.path = Path(path)
        self.contract = contract

    def _label(self, raw, issues: set[str], prefix: str) -> None:
        if raw is None or raw == "":
            issues.add(
                "missing_label_csv" if prefix == "tabular" else f"{prefix}_bad_label"
            )
            return
        if isinstance(raw, str) and raw.lower() in ("inf", "+inf", "-inf", "infinity"):
            issues.add(
                "inf_label_csv" if prefix == "tabular" else f"{prefix}_bad_label"
            )
            return
        if isinstance(raw, (bool, float)):
            issues.add(
                "invalid_label_csv" if prefix == "tabular" else f"{prefix}_bad_label"
            )
            return
        try:
            label = int(raw)
            if not 0 <= label < self.contract.classes:
                issues.add(
                    "out_of_range_label_csv"
                    if prefix == "tabular"
                    else f"{prefix}_bad_label"
                )
        except (TypeError, ValueError):
            issues.add(
                "invalid_label_csv" if prefix == "tabular" else f"{prefix}_bad_label"
            )

    def _scan_csv(self, path: Path, kind: str, required: tuple[str, ...], on_row):
        if not path.is_file():
            return (
                {"missing_file" if kind == "tabular" else f"{kind}_missing_index"},
                0,
                [],
            )
        issues = set()
        rows = 0
        fields = []
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fields = reader.fieldnames or []
                if len(fields) != len(set(fields)):
                    issues.add(
                        "duplicate_csv_column"
                        if kind == "tabular"
                        else f"{kind}_duplicate_index_column"
                    )
                for field in required:
                    if field not in fields:
                        issues.add(
                            ("wrong_column" if field == "label" else "wrong_feature")
                            if kind == "tabular"
                            else f"{kind}_wrong_index_column"
                        )
                for row in reader:
                    rows += 1
                    on_row(row, issues)
        except (OSError, UnicodeError, csv.Error):
            issues.add(
                "unreadable_csv"
                if kind == "tabular"
                else (
                    "text_invalid_utf8"
                    if kind == "text"
                    else f"{kind}_unreadable_index"
                )
            )
        return issues, rows, fields

    def check_tabular_csv(self) -> set[str]:
        def on_row(row, issues):
            for column in self.contract.feature_columns:
                raw = row.get(column)
                if raw is None or raw == "":
                    issues.add("missing_value_csv")
                    continue
                try:
                    number = float(raw)
                except (TypeError, ValueError):
                    issues.add("nonnumeric_csv")
                    continue
                if math.isnan(number):
                    issues.add("nan_csv")
                elif math.isinf(number):
                    issues.add("inf_csv")
            self._label(row.get(self.contract.label_column), issues, "tabular")

        issues, rows, _ = self._scan_csv(
            self.path,
            "tabular",
            (*self.contract.feature_columns, self.contract.label_column),
            on_row,
        )
        if rows < self.contract.min_rows and "missing_file" not in issues:
            issues.add("too_few_rows")
        return issues

    def _asset(self, root: Path, relative, issues: set[str], kind: str) -> Path | None:
        if not isinstance(relative, str) or not relative:
            issues.add(f"{kind}_missing_asset")
            return None
        candidate = (root / relative).resolve()
        if not candidate.is_relative_to(root.resolve()) or not candidate.is_file():
            issues.add(f"{kind}_missing_asset")
            return None
        return candidate

    def _image_file(self, path: Path, issues: set[str]) -> None:
        try:
            with Image.open(path) as image:
                if image.size != self.contract.image_size:
                    issues.add("image_wrong_size")
                if image.mode != self.contract.image_mode:
                    issues.add("image_wrong_mode")
                image.verify()
        except (OSError, ValueError, UnidentifiedImageError):
            issues.add("image_corrupt_asset")

    def check_image_index(self) -> set[str]:
        files = set()

        def on_row(row, issues):
            self._label(row.get(self.contract.label_column), issues, "image")
            asset = self._asset(self.path.parent, row.get("file"), issues, "image")
            if asset is not None:
                if asset in files:
                    issues.add("image_duplicate_asset")
                files.add(asset)
                self._image_file(asset, issues)

        issues, rows, _ = self._scan_csv(
            self.path, "image", ("file", self.contract.label_column), on_row
        )
        if rows and rows < self.contract.min_rows:
            issues.add("image_wrong_row_count")
        return issues

    def check_image_directory(self) -> set[str]:
        issues = set()
        if not self.path.is_dir():
            return {"image_missing_directory"}
        if any(
            child.name not in {str(label) for label in range(self.contract.classes)}
            for child in self.path.iterdir()
        ):
            issues.add("image_unexpected_asset")
        count = 0
        for label in range(self.contract.classes):
            class_dir = self.path / str(label)
            if not class_dir.is_dir():
                issues.add("image_missing_class_directory")
                continue
            for asset in class_dir.iterdir():
                if not asset.is_file() or asset.suffix.lower() not in (
                    ".png",
                    ".jpg",
                    ".jpeg",
                ):
                    issues.add("image_unexpected_asset")
                    continue
                count += 1
                self._image_file(asset, issues)
        if count < self.contract.min_rows:
            issues.add("image_wrong_row_count")
        return issues

    def check_text_csv(self) -> set[str]:
        def on_row(row, issues):
            raw = row.get(self.contract.text_column)
            if not isinstance(raw, str) or not raw.strip():
                issues.add("text_missing_value")
            self._label(row.get(self.contract.label_column), issues, "text")

        issues, rows, _ = self._scan_csv(
            self.path,
            "text",
            (self.contract.text_column, self.contract.label_column),
            on_row,
        )
        if rows and rows < self.contract.min_rows:
            issues.add("text_wrong_row_count")
        return issues

    def check_text_jsonl(self) -> set[str]:
        if not self.path.is_file():
            return {"text_missing_file"}
        issues = set()
        rows = 0
        try:
            with self.path.open(encoding="utf-8") as handle:
                for line in handle:
                    rows += 1
                    try:
                        record = json.loads(line)
                    except (json.JSONDecodeError, TypeError):
                        issues.add("text_invalid_jsonl")
                        continue
                    if not isinstance(record, dict):
                        issues.add("text_invalid_jsonl")
                        continue
                    raw = record.get(self.contract.text_column)
                    if not isinstance(raw, str) or not raw.strip():
                        issues.add("text_missing_value")
                    self._label(record.get(self.contract.label_column), issues, "text")
        except (OSError, UnicodeError):
            issues.add("text_invalid_utf8")
        if rows < self.contract.min_rows:
            issues.add("text_wrong_row_count")
        return issues

    def _audio_file(self, path: Path, issues: set[str]) -> None:
        try:
            with wave.open(str(path), "rb") as handle:
                if handle.getframerate() != self.contract.sample_rate:
                    issues.add("audio_bad_rate")
                if handle.getnchannels() != self.contract.channels:
                    issues.add("audio_stereo")
                if handle.getsampwidth() != self.contract.sample_width:
                    issues.add("audio_bad_width")
                if handle.getnframes() != self.contract.frame_count:
                    issues.add("audio_truncated")
        except (OSError, ValueError, EOFError, wave.Error):
            issues.add("audio_corrupt")

    def check_audio_index(self) -> set[str]:
        def on_row(row, issues):
            self._label(row.get(self.contract.label_column), issues, "audio")
            asset = self._asset(self.path.parent, row.get("file"), issues, "audio")
            if asset is not None:
                self._audio_file(asset, issues)

        issues, rows, _ = self._scan_csv(
            self.path, "audio", ("file", self.contract.label_column), on_row
        )
        if rows and rows < self.contract.min_rows:
            issues.add("audio_wrong_row_count")
        return issues

    def check_audio_directory(self) -> set[str]:
        issues = set()
        if not self.path.is_dir():
            return {"audio_missing_directory"}
        if any(
            child.name not in {str(label) for label in range(self.contract.classes)}
            for child in self.path.iterdir()
        ):
            issues.add("audio_unexpected_asset")
        count = 0
        for label in range(self.contract.classes):
            class_dir = self.path / str(label)
            if not class_dir.is_dir():
                issues.add("audio_missing_class_directory")
                continue
            for asset in class_dir.iterdir():
                if not asset.is_file() or asset.suffix.lower() != ".wav":
                    issues.add("audio_unexpected_asset")
                    continue
                count += 1
                self._audio_file(asset, issues)
        if count < self.contract.min_rows:
            issues.add("audio_wrong_row_count")
        return issues


CHECKERS.update(
    {
        "tabular_csv": lambda path, contract: DatasetValidator(
            path, contract
        ).check_tabular_csv(),
        "image_csv_index": lambda path, contract: DatasetValidator(
            path, contract
        ).check_image_index(),
        "image_directory": lambda path, contract: DatasetValidator(
            path, contract
        ).check_image_directory(),
        "text_csv": lambda path, contract: DatasetValidator(
            path, contract
        ).check_text_csv(),
        "text_jsonl": lambda path, contract: DatasetValidator(
            path, contract
        ).check_text_jsonl(),
        "audio_csv_index": lambda path, contract: DatasetValidator(
            path, contract
        ).check_audio_index(),
        "audio_directory": lambda path, contract: DatasetValidator(
            path, contract
        ).check_audio_directory(),
    }
)

check_tabular_csv = CHECKERS["tabular_csv"]
check_image_index = CHECKERS["image_csv_index"]
check_image_directory = CHECKERS["image_directory"]
check_text_csv = CHECKERS["text_csv"]
check_text_jsonl = CHECKERS["text_jsonl"]
check_audio_index = CHECKERS["audio_csv_index"]
check_audio_directory = CHECKERS["audio_directory"]


_MASK = 0xFFFFFFFF
# TensorFlow Record framing and minimal Example decoding.
_POLYNOMIAL = 0x82F63B78
_MAX_RECORD_BYTES = 128 * 1024 * 1024


def _table():
    values = []
    for start in range(256):
        value = start
        for _ in range(8):
            value = (value >> 1) ^ (_POLYNOMIAL if value & 1 else 0)
        values.append(value)
    return tuple(values)


_CRC_TABLE = _table()


def masked_crc32c(data):
    crc = _MASK
    for byte in data:
        crc = _CRC_TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    crc ^= _MASK
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & _MASK


def records(path: Path):
    """Yield complete payloads; reject truncated or checksum-failing records."""
    with path.open("rb") as handle:
        while True:
            length_bytes = handle.read(8)
            if not length_bytes:
                return
            if len(length_bytes) != 8:
                raise ValueError("truncated TFRecord length")
            length_crc = handle.read(4)
            if len(length_crc) != 4 or struct.unpack("<I", length_crc)[
                0
            ] != masked_crc32c(length_bytes):
                raise ValueError("invalid TFRecord length checksum")
            length = struct.unpack("<Q", length_bytes)[0]
            if length > _MAX_RECORD_BYTES:
                raise ValueError("oversized TFRecord")
            payload = handle.read(length)
            data_crc = handle.read(4)
            if len(payload) != length or len(data_crc) != 4:
                raise ValueError("truncated TFRecord payload")
            if struct.unpack("<I", data_crc)[0] != masked_crc32c(payload):
                raise ValueError("invalid TFRecord payload checksum")
            yield payload


def _varint(data, offset):
    result = 0
    for shift in range(0, 70, 7):
        if offset >= len(data):
            raise ValueError("truncated protobuf varint")
        byte = data[offset]
        offset += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, offset
    raise ValueError("oversized protobuf varint")


def _fields(data):
    offset = 0
    while offset < len(data):
        tag, offset = _varint(data, offset)
        number, wire = tag >> 3, tag & 7
        if number == 0:
            raise ValueError("invalid protobuf field")
        if wire == 0:
            value, offset = _varint(data, offset)
        elif wire == 2:
            length, offset = _varint(data, offset)
            end = offset + length
            if end > len(data):
                raise ValueError("truncated protobuf bytes")
            value = data[offset:end]
            offset = end
        elif wire in (1, 5):
            end = offset + (8 if wire == 1 else 4)
            if end > len(data):
                raise ValueError("truncated protobuf fixed field")
            value = data[offset:end]
            offset = end
        else:
            raise ValueError("unsupported protobuf wire type")
        yield number, wire, value


def _only(fields, number, wire):
    matches = [
        value for field, kind, value in fields if field == number and kind == wire
    ]
    if len(matches) != 1:
        raise ValueError("missing or repeated protobuf feature")
    return matches[0]


def cassava_example(payload):
    """Read image bytes and integer target from a tf.train.Example payload."""
    features_blob = _only(list(_fields(payload)), 1, 2)
    features = {}
    for number, wire, entry in _fields(features_blob):
        if (number, wire) != (1, 2):
            continue
        parts = list(_fields(entry))
        key = _only(parts, 1, 2).decode("utf-8")
        if key in features:
            raise ValueError("duplicate protobuf feature")
        features[key] = _only(parts, 2, 2)
    image_feature = _only(list(_fields(features["image"])), 1, 2)
    image = _only(list(_fields(image_feature)), 1, 2)
    target_feature = _only(list(_fields(features["target"])), 3, 2)
    target_fields = list(_fields(target_feature))
    unpacked = [
        value for number, wire, value in target_fields if (number, wire) == (1, 0)
    ]
    packed = [
        value for number, wire, value in target_fields if (number, wire) == (1, 2)
    ]
    if packed:
        if len(packed) != 1 or unpacked:
            raise ValueError("repeated target")
        offset = 0
        unpacked = []
        while offset < len(packed[0]):
            value, offset = _varint(packed[0], offset)
            unpacked.append(value)
    if len(unpacked) != 1:
        raise ValueError("missing or repeated target")
    target = unpacked[0]
    if target >= 1 << 63:
        target -= 1 << 64
    return image, target


# Competition-specific dataset and source checks.
TASKS = {
    "petfinder-pawpularity-score": ("Id", "Pawpularity", "regression", 1),
    "cassava-leaf-disease-classification": ("image_id", "label", "classification", 5),
    "uw-madison-gi-tract-image-segmentation": (
        "id",
        "segmentation",
        "segmentation",
        768,
    ),
    "aptos2019-blindness-detection": ("id_code", "diagnosis", "classification", 5),
    "seti-breakthrough-listen": ("id", "target", "classification", 2),
    "freesound-audio-tagging-2019": ("fname", "labels", "multilabel", 3),
    "google-quest-challenge": (
        "qa_id",
        "question_asker_intent_understanding",
        "regression",
        30,
    ),
    "jigsaw-toxic-comment-classification-challenge": ("id", "toxic", "multilabel", 6),
    "chaii-hindi-and-tamil-question-answering": ("id", "answer_start", "regression", 2),
}
PHOTO = "Focus Eyes Face Near Action Accessory Group Collage Human Occlusion Info Blur".split()
ORGANS = ("large_bowel", "small_bowel", "stomach")
JIGSAW = ("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")

QUEST_FIRST = "question_asker_intent_understanding"


class MissingZipCSV(Exception):
    pass


@contextmanager
def _csv_stream(path):
    if path.suffix.lower() != ".zip":
        with path.open(newline="", encoding="utf-8") as handle:
            yield handle
        return
    with zipfile.ZipFile(path) as archive:
        csv_names = [
            item.filename
            for item in archive.infolist()
            if not item.is_dir() and item.filename.lower().endswith(".csv")
        ]
        preferred = path.name[:-4]
        if preferred in csv_names:
            member = preferred
        elif len(csv_names) == 1:
            member = csv_names[0]
        else:
            raise MissingZipCSV(path)
        with archive.open(member) as raw:
            with io.TextIOWrapper(raw, encoding="utf-8", newline="") as handle:
                yield handle


def assignment(tree, key):
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == key for t in node.targets
        ):
            try:
                return ast.literal_eval(node.value)
            except (TypeError, ValueError):
                return None
    return None


def _number(raw, low, high):
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and low <= value <= high


def _kaggle_asset(root, relative, issues):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        issues.add("missing_asset")
        return None
    return path


def _image(path, issues, grayscale=False):
    try:
        with Image.open(path) as image:
            if (
                min(image.size) < 1
                or (grayscale and image.mode not in ("I;16", "I;16B", "I"))
                or (not grayscale and image.mode != "RGB")
            ):
                issues.add("bad_asset_shape_or_type")
            dimensions = image.size
            image.load()
            return dimensions
    except (OSError, ValueError, UnidentifiedImageError):
        issues.add("corrupt_asset")
        return None


def _image_in_archive(root, archive_name, filename, issues):
    archive_path = root / archive_name
    if not archive_path.is_file():
        issues.add("missing_asset")
        return
    try:
        with zipfile.ZipFile(archive_path) as archive:
            matches = [
                item
                for item in archive.infolist()
                if not item.is_dir() and Path(item.filename).name == filename
            ]
            if len(matches) != 1:
                issues.add("missing_asset" if not matches else "ambiguous_asset")
                return
            with archive.open(matches[0]) as raw:
                _image(io.BytesIO(raw.read()), issues)
    except zipfile.BadZipFile:
        issues.add("zip_corrupt")
    except (OSError, ValueError):
        issues.add("corrupt_asset")


def _rle(raw, size, issues):
    if raw in (None, ""):
        return
    try:
        values = [int(v) for v in raw.split()]
    except (AttributeError, ValueError):
        issues.add("invalid_target")
        return
    if len(values) % 2 or any(
        start < 1 or length < 1 or start + length - 1 > size
        for start, length in zip(values[::2], values[1::2])
    ):
        issues.add("invalid_target")


def _audio_info(root, folder, filename, issues):
    import soundfile as sf

    path = root / folder / filename
    try:
        if path.is_file():
            info = sf.info(path)
        else:
            archive_path = root / f"{folder}.zip"
            if not archive_path.is_file():
                issues.add("missing_asset")
                return
            with zipfile.ZipFile(archive_path) as archive:
                matches = [
                    item
                    for item in archive.infolist()
                    if not item.is_dir() and Path(item.filename).name == filename
                ]
                if len(matches) != 1:
                    issues.add("missing_asset" if not matches else "ambiguous_asset")
                    return
                with archive.open(matches[0]) as raw:
                    info = sf.info(io.BytesIO(raw.read()))
        if info.frames < 1 or info.samplerate < 1 or info.channels not in (1, 2):
            issues.add("bad_asset_shape_or_type")
    except zipfile.BadZipFile:
        issues.add("zip_corrupt")
    except (OSError, ValueError, RuntimeError):
        issues.add("corrupt_asset")


def _cassava_tfrecord_issues(index, issues):
    shards = sorted(
        path
        for path in index.iterdir()
        if path.is_file() and path.suffix.lower() in (".tfrec", ".tfrecord")
    )
    if not shards:
        issues.add("missing_file")
        return
    count = 0
    for shard in shards:
        try:
            for payload in records(shard):
                count += 1
                try:
                    image, target = cassava_example(payload)
                except (KeyError, UnicodeError, ValueError):
                    issues.add("invalid_record_schema")
                    continue
                if target not in range(5):
                    issues.add("invalid_target")
                _image(io.BytesIO(image), issues)
        except (OSError, ValueError):
            issues.add("corrupt_tfrecord")
    if count < 6:
        issues.add("too_few_rows")


class CompetitionValidator:
    """Check one competition index and its referenced assets."""

    def __init__(self, task, index, root, cache):
        self.task = task
        self.index = index
        self.root = root
        self.cache = cache
        self.issues = set()
        self.required = set()
        self.vocabulary = set()
        self.organ_seen = {}
        self.target_field = None

    def _schema(self):
        """Return required columns and optional label vocabulary."""
        input_field, target_field, _, _ = TASKS[self.task]
        required = {input_field, target_field}
        vocabulary = set()
        if self.task == "petfinder-pawpularity-score":
            required.update(PHOTO)
        elif self.task == "uw-madison-gi-tract-image-segmentation":
            required.add("class")
        elif self.task == "google-quest-challenge":
            required.update(("question_title", "question_body", "answer"))
            sample = self.root / "sample_submission.csv"
            if sample.is_file():
                with sample.open(newline="", encoding="utf-8") as handle:
                    targets = (csv.reader(handle).__next__())[1:]
                required.update(targets)
                if len(targets) != 30 or targets[0] != QUEST_FIRST:
                    self.issues.add("wrong_target_schema")
            else:
                self.issues.add("missing_target_schema")
        elif self.task == "jigsaw-toxic-comment-classification-challenge":
            required.update(JIGSAW)
            required.add("comment_text")
        elif self.task == "chaii-hindi-and-tamil-question-answering":
            required.update(("context", "question", "answer_text", "language"))
        if self.task == "freesound-audio-tagging-2019":
            try:
                with (self.root / "sample_submission.csv").open(
                    newline="", encoding="utf-8"
                ) as handle:
                    vocabulary = set(next(csv.reader(handle))[1:])
            except (OSError, UnicodeError, StopIteration):
                self.issues.add("missing_target_schema")
        return required, vocabulary

    def _check_petfinder_row(self, row, ident):
        if not _number(row.get(self.target_field), 0, 100) or any(
            not _number(row.get(column), 0, 1) for column in PHOTO
        ):
            self.issues.add("invalid_target")
        path = _kaggle_asset(self.root, f"train/{ident}.jpg", self.issues)
        if path:
            _image(path, self.issues)

    def _check_cassava_row(self, row, ident):
        target = row.get(self.target_field)
        if not _number(target, 0, 4) or not str(target).isdigit():
            self.issues.add("invalid_target")
        path = _kaggle_asset(self.root, f"train_images/{ident}", self.issues)
        if path:
            _image(path, self.issues)

    def _check_gi_tract_row(self, row, ident):
        organ = row.get("class")
        if organ not in ORGANS:
            self.issues.add("invalid_target")
        self.organ_seen.setdefault(ident, set()).add(organ)
        try:
            case, day, _, slice_id = str(ident).split("_")
            scans = list(
                (self.root / "train" / case / f"{case}_{day}" / "scans").glob(
                    f"slice_{slice_id}_*.png"
                )
            )
        except ValueError:
            self.issues.add("missing_asset")
            return
        if len(scans) != 1:
            self.issues.add("missing_asset" if not scans else "ambiguous_asset")
            return
        dimensions = _image(scans[0], self.issues, grayscale=True)
        if dimensions:
            _rle(row.get(self.target_field), dimensions[0] * dimensions[1], self.issues)

    def _check_aptos_row(self, row, ident):
        target = row.get(self.target_field)
        if not str(target).isdigit() or not _number(target, 0, 4):
            self.issues.add("invalid_target")
        path = self.root / "train_images" / f"{ident}.png"
        if path.is_file():
            _image(path, self.issues)
        else:
            _image_in_archive(self.root, "train.zip", f"{ident}.png", self.issues)

    def _check_seti_row(self, row, ident):
        if str(row.get(self.target_field)) not in ("0", "1"):
            self.issues.add("invalid_target")
        path = _kaggle_asset(
            self.root, f"train/{str(ident)[0]}/{ident}.npy", self.issues
        )
        if path:
            try:
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                if array.shape != (6, 273, 256) or array.dtype != np.float16:
                    self.issues.add("bad_asset_shape_or_type")
            except (OSError, ValueError, EOFError):
                self.issues.add("corrupt_asset")

    def _check_freesound_row(self, row, ident):
        labels = str(row.get(self.target_field, "")).split(",")
        if not labels or any(label not in self.vocabulary for label in labels):
            self.issues.add("invalid_target")
        folder = (
            "train_noisy"
            if self.index.name.startswith("train_noisy")
            else "train_curated"
        )
        _audio_info(self.root, folder, str(ident), self.issues)

    def _check_quest_row(self, row, ident):
        if any(
            not str(row.get(column, "")).strip()
            for column in ("question_title", "question_body", "answer")
        ):
            self.issues.add("missing_text")
        if any(
            not _number(row.get(column), 0, 1)
            for column in self.required
            if column.startswith(("question_", "answer_"))
            and column not in ("question_title", "question_body", "answer")
        ):
            self.issues.add("invalid_target")

    def _check_jigsaw_row(self, row, ident):
        if not str(row.get("comment_text", "")).strip():
            self.issues.add("missing_text")
        if any(row.get(column) not in ("0", "1") for column in JIGSAW):
            self.issues.add("invalid_target")

    def _check_chaii_row(self, row, ident):
        context = row.get("context") or ""
        question = row.get("question") or ""
        answer = row.get("answer_text") or ""
        if not context.strip() or not question.strip() or not answer.strip():
            self.issues.add("missing_text")
        if row.get("language") not in ("hindi", "tamil"):
            self.issues.add("invalid_language")
        try:
            start = int(row.get(self.target_field))
            if start < 0 or (
                context[start : start + len(answer)] != answer and answer not in context
            ):
                self.issues.add("invalid_target")
        except (ValueError, TypeError):
            self.issues.add("invalid_target")

    _ROW_CHECKERS = {
        "petfinder-pawpularity-score": _check_petfinder_row,
        "cassava-leaf-disease-classification": _check_cassava_row,
        "uw-madison-gi-tract-image-segmentation": _check_gi_tract_row,
        "aptos2019-blindness-detection": _check_aptos_row,
        "seti-breakthrough-listen": _check_seti_row,
        "freesound-audio-tagging-2019": _check_freesound_row,
        "google-quest-challenge": _check_quest_row,
        "jigsaw-toxic-comment-classification-challenge": _check_jigsaw_row,
        "chaii-hindi-and-tamil-question-answering": _check_chaii_row,
    }

    def _rows(self):
        """Stream rows and report index read failures as findings."""
        try:
            with _csv_stream(self.index) as handle:
                yield from csv.DictReader(handle)
        except MissingZipCSV:
            self.issues.add("zip_missing_csv")
        except zipfile.BadZipFile:
            self.issues.add("zip_corrupt")
        except (OSError, UnicodeError, csv.Error):
            self.issues.add("unreadable_index")

    def check(self):
        key = (self.task, self.index.resolve(), self.root.resolve())
        if key in self.cache:
            return self.cache[key]
        if self.task == "cassava-leaf-disease-classification" and self.index.is_dir():
            _cassava_tfrecord_issues(self.index, self.issues)
            self.cache[key] = self.issues
            return self.issues
        if not self.index.is_file():
            return {"missing_file"}
        try:
            with _csv_stream(self.index) as handle:
                fields = csv.DictReader(handle).fieldnames or []
        except MissingZipCSV:
            return {"zip_missing_csv"}
        except zipfile.BadZipFile:
            return {"zip_corrupt"}
        except (OSError, UnicodeError, csv.Error):
            return {"unreadable_index"}

        input_field, self.target_field, _, _ = TASKS[self.task]
        self.required, self.vocabulary = self._schema()
        if len(fields) != len(set(fields)):
            self.issues.add("duplicate_column")
        if not self.required.issubset(fields):
            self.issues.add("wrong_header")

        check_row = self._ROW_CHECKERS[self.task]
        seen = set()
        row_count = 0
        for row in self._rows():
            row_count += 1
            ident = row.get(input_field)
            unique_key = (
                (ident, row.get("class"))
                if self.task == "uw-madison-gi-tract-image-segmentation"
                else ident
            )
            if unique_key in seen:
                self.issues.add("duplicate_id")
            seen.add(unique_key)
            if ident in (None, ""):
                self.issues.add(
                    "missing_text"
                    if self.task
                    in (
                        "google-quest-challenge",
                        "jigsaw-toxic-comment-classification-challenge",
                        "chaii-hindi-and-tamil-question-answering",
                    )
                    else "missing_asset"
                )
                continue
            check_row(self, row, ident)

        if self.task == "uw-madison-gi-tract-image-segmentation" and any(
            value != set(ORGANS) for value in self.organ_seen.values()
        ):
            self.issues.add("missing_organ_mask")
        if row_count < 6:
            self.issues.add("too_few_rows")
        self.cache[key] = self.issues
        return self.issues


def kaggle_dataset_issues(task, index, root, cache):
    """Compatibility entry point for competition dataset validation."""
    return CompetitionValidator(task, index, root, cache).check()


def check_kaggle(path, tree, cache):
    task = assignment(tree, "TASK")
    issues = set()
    if task not in TASKS:
        return ["unknown_task"]
    expected_input, expected_target, expected_type, expected_dim = TASKS[task]
    if assignment(tree, "INPUT_FIELD") != expected_input:
        issues.add("wrong_input_field")
    if assignment(tree, "TARGET_FIELD") != expected_target:
        issues.add("wrong_target_field")
    if (
        assignment(tree, "TASK_TYPE") != expected_type
        or assignment(tree, "OUT_DIM") != expected_dim
    ):
        issues.add("wrong_task_shape")
    index = assignment(tree, "DATA_INDEX")
    root = assignment(tree, "ASSET_ROOT")
    base = Path(__file__).resolve().parent
    if not isinstance(index, str) or not isinstance(root, str):
        issues.add("missing_file")
    else:
        index_path = (base / index).resolve()
        root_path = (base / root).resolve()
        if not index_path.is_relative_to(base) or not root_path.is_relative_to(base):
            issues.add("unsafe_path")
        else:
            issues.update(kaggle_dataset_issues(task, index_path, root_path, cache))
    run = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run"
        ),
        None,
    )
    if run is None:
        return sorted(issues | {"missing_pipeline"})
    for node in ast.walk(run):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "loss" for t in node.targets
        ):
            if any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "detach"
                for n in ast.walk(node.value)
            ):
                issues.add("detached_loss")
        if isinstance(node, ast.With) and any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "no_grad"
            for item in node.items
            for n in ast.walk(item.context_expr)
        ):
            if any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == "model"
                and any(
                    isinstance(arg, ast.Name) and arg.id == "xtrain" for arg in n.args
                )
                for n in ast.walk(node)
            ):
                issues.add("no_grad_train")
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Tuple)
            and {elt.id for elt in t.elts if isinstance(elt, ast.Name)}
            == {"xval", "yval"}
            for t in node.targets
        ):
            if "features[:4]" in ast.unparse(
                node.value
            ) or "targets[:4]" in ast.unparse(node.value):
                issues.add("validation_on_train")
    return sorted(issues)


# Image, text, and audio source contracts.
def constants(tree):
    values = {}
    for n in tree.body:
        if (
            isinstance(n, ast.Assign)
            and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name)
        ):
            try:
                values[n.targets[0].id] = ast.literal_eval(n.value)
            except (ValueError, TypeError):
                pass
    return values


def check_modality(path, tree, cache):
    values = constants(tree)
    modality = values.get("MODALITY")
    issues = set()
    if modality not in ("image", "text", "audio"):
        return ["unknown_modality"]
    index_name = values.get("INDEX_FILE")
    if not isinstance(index_name, str) or Path(index_name).name != index_name:
        issues.add(f"{modality}_missing_index")
    else:
        index = ROOT / "data" / index_name
        kind = {
            "image": "image_csv_index",
            "text": "text_csv",
            "audio": "audio_csv_index",
        }[modality]
        issues.update(dataset_issues(index, DatasetContract(kind, min_rows=32), cache))

    # Canonicalized syntax ignores quote style and ordinary whitespace.
    code = ast.unparse(tree).replace('"', "'").replace(" ", "").replace("\n", "")
    if modality == "image":
        if "image.convert('L')" in code:
            issues.add("image_grayscale")
        if "dtype=np.uint8" in code:
            issues.add("image_uint8_tensor")
        elif "np.asarray(image,dtype=np.float32)/255.0" not in code:
            issues.add("image_unnormalized")
        if "permute(2,0,1)" not in code:
            issues.add("image_channel_order")
        if "table['class']" in code:
            issues.add("image_wrong_label_column")
        if "root/'absent'" in code:
            issues.add("image_wrong_root")
        if "resize((8,8))" in code:
            issues.add("image_bad_resize")
        if "[::-1].copy()" in code:
            issues.add("image_label_permutation")
        if values.get("IMAGE_SIZE") != (16, 16):
            issues.add("image_wrong_size_contract")
    elif modality == "text":
        if "VOCAB.get(word,len(VOCAB))" in code:
            issues.add("text_oov_bad_id")
        if "ids+[-1]*" in code:
            issues.add("text_negative_pad")
        if "str(text).split()" in code:
            issues.add("text_no_lower")
        if "split(',')" in code:
            issues.add("text_wrong_split")
        if (
            "str(text).lower().split()" in code
            and "str(text).lower().split()[:MAX_TOKENS]" not in code
        ):
            issues.add("text_no_truncate")
        if "dtype=torch.float32" in code:
            issues.add("text_float_ids")
        if "mask=(x==0)" in code:
            issues.add("text_invert_mask")
        if "table['sentence']" in code:
            issues.add("text_wrong_column")
        if "[::-1].copy()" in code:
            issues.add("text_label_permutation")
        if values.get("MAX_TOKENS") != 8:
            issues.add("text_wrong_length_contract")
    else:
        for key, expected, bug in (
            ("SAMPLE_RATE", 8000, "audio_wrong_rate_constant"),
            ("CHANNELS", 1, "audio_wrong_channels_constant"),
            ("SAMPLE_WIDTH", 2, "audio_wrong_width_constant"),
            ("FRAME_COUNT", 256, "audio_wrong_frames_constant"),
        ):
            if values.get(key) != expected:
                issues.add(bug)
        if "dtype='>i2'" in code:
            issues.add("audio_endian")
        if "/32768.0" not in code:
            issues.add("audio_no_normalize")
        if "table['class']" in code:
            issues.add("audio_wrong_label_column")
        if "[::-1].copy()" in code:
            issues.add("audio_label_permutation")
    return sorted(issues)


# Shared abstract-syntax-tree helpers and controlled model-node rules.
def name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = name(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


def target_names(node):
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, (ast.Tuple, ast.List)):
        return set().union(*(target_names(e) for e in node.elts))
    return set()


def assigned(tree, variable):
    if not hasattr(tree, "_assignment_index"):
        index = {}
        for n in ast.walk(tree):
            if isinstance(n, ast.Assign):
                targets = n.targets
            elif isinstance(n, ast.AnnAssign):
                targets = [n.target]
            else:
                continue
            for target in targets:
                for target_name in target_names(target):
                    index.setdefault(target_name, []).append(n)
        tree._assignment_index = index
    return tree._assignment_index.get(variable, [])


def value(statement):
    return statement.value


def unparse(node):
    return ast.unparse(node) if node is not None else ""


def has_call(node, call_name):
    return any(
        isinstance(n, ast.Call) and name(n.func) == call_name for n in ast.walk(node)
    )


def literal_assignment(tree, variable):
    matches = assigned(tree, variable)
    if not matches:
        return None
    try:
        return ast.literal_eval(value(matches[0]))
    except (ValueError, TypeError, MemoryError):
        return None


def local_unknowns(function, global_names):
    local = {
        a.arg
        for a in function.args.posonlyargs
        + function.args.args
        + function.args.kwonlyargs
    }
    if function.args.vararg:
        local.add(function.args.vararg.arg)
    if function.args.kwarg:
        local.add(function.args.kwarg.arg)
    nodes = list(ast.walk(function))
    local.update(
        n.id for n in nodes if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
    )
    unknown = set()
    for n in nodes:
        if (
            isinstance(n, ast.Name)
            and isinstance(n.ctx, ast.Load)
            and n.id not in local | global_names | BUILTINS
        ):
            unknown.add(n.id)
    return unknown


class SourceResourceChecker:
    """Detect one high-confidence container resource hazard without running code."""

    def __init__(self, tree: ast.AST, shared_memory_bytes: int | None):
        self.tree = tree
        self.shared_memory_bytes = shared_memory_bytes
        self.parents = {
            child: parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        self.class_constants = {}
        for statement in getattr(tree, "body", []):
            if not isinstance(statement, ast.ClassDef):
                continue
            for member in statement.body:
                if not isinstance(member, ast.Assign):
                    continue
                if not isinstance(member.value, ast.Constant) or type(member.value.value) is not int:
                    continue
                for target in member.targets:
                    if isinstance(target, ast.Name):
                        self.class_constants[f"{statement.name}.{target.id}"] = member.value.value
        self.assignments = {}
        for scope in ast.walk(tree):
            if not isinstance(scope, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            values = {}
            if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for argument in scope.args.posonlyargs + scope.args.args + scope.args.kwonlyargs:
                    values[argument.arg] = [(0, None)]
            for statement in scope.body:
                if isinstance(statement, ast.Assign):
                    for target in statement.targets:
                        if isinstance(target, ast.Name):
                            values.setdefault(target.id, []).append((statement.lineno, statement.value))
            self.assignments[scope] = values

    def _binding(self, name: str, context: ast.AST) -> ast.AST | None:
        line = getattr(context, "lineno", 0)
        scope = context
        while scope is not None:
            assignments = self.assignments.get(scope, {})
            if name in assignments:
                earlier = [value for assigned_line, value in assignments[name] if assigned_line <= line]
                return earlier[-1] if earlier else None
            scope = self.parents.get(scope)
        return None

    def _integer(self, node: ast.AST | None, context: ast.AST) -> int | None:
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.Name):
            bound = self._binding(node.id, context)
            if isinstance(bound, (ast.Constant, ast.Attribute)):
                return self._integer(bound, context)
            return None
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            return self.class_constants.get(f"{node.value.id}.{node.attr}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "int" and len(node.args) == 1:
            return self._integer(node.args[0], context)
        if isinstance(node, ast.IfExp):
            values = [self._integer(node.body, context), self._integer(node.orelse, context)]
            known = [value for value in values if value is not None]
            return max(known) if known else None
        return None

    def _dataset_constructor(self, loader: ast.Call) -> ast.Call | None:
        dataset = loader.args[0] if loader.args else next(
            (item.value for item in loader.keywords if item.arg == "dataset"), None
        )
        if isinstance(dataset, ast.Name):
            dataset = self._binding(dataset.id, loader)
        return dataset if isinstance(dataset, ast.Call) else None

    def _image_area(self, loader: ast.Call) -> int | None:
        calls = [node for node in ast.walk(self.tree) if isinstance(node, ast.Call)]
        pil_rgb = any(
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "convert"
            and call.args
            and isinstance(call.args[0], ast.Constant)
            and call.args[0].value == "RGB"
            for call in calls
        )
        cv2_rgb = any(
            name(call.func) == "cv2.cvtColor"
            and len(call.args) > 1
            and name(call.args[1]) == "cv2.COLOR_BGR2RGB"
            for call in calls
        ) and any(name(call.func) == "Image.fromarray" for call in calls)
        if not (pil_rgb or cv2_rgb):
            return None
        dataset = self._dataset_constructor(loader)
        if dataset is not None:
            transform = next(
                (item.value for item in dataset.keywords if item.arg == "transform"),
                None,
            )
            if transform is None and isinstance(dataset.func, ast.Name):
                dataset_class = next(
                    (statement for statement in getattr(self.tree, "body", [])
                     if isinstance(statement, ast.ClassDef)
                     and statement.name == dataset.func.id),
                    None,
                )
                constructor = next(
                    (member for member in dataset_class.body
                     if isinstance(member, ast.FunctionDef)
                     and member.name == "__init__"),
                    None,
                ) if dataset_class is not None else None
                if constructor is not None:
                    parameters = [argument.arg for argument in constructor.args.args[1:]]
                    if "transform" in parameters:
                        index = parameters.index("transform")
                        if index < len(dataset.args):
                            transform = dataset.args[index]
            if isinstance(transform, ast.Name):
                transform = self._binding(transform.id, dataset)
            if isinstance(transform, ast.Call):
                calls = [transform]
        areas = []
        for call in calls:
            if not isinstance(call.func, ast.Attribute) or call.func.attr != "Compose" or not call.args:
                continue
            transforms = call.args[0]
            if not isinstance(transforms, (ast.List, ast.Tuple)):
                continue
            area = None
            for transform in transforms.elts:
                if not isinstance(transform, ast.Call) or not isinstance(transform.func, ast.Attribute):
                    continue
                transform_name = transform.func.attr
                if transform_name == "ToTensor":
                    if area is not None:
                        areas.append(area)
                    break
                if transform_name not in {"RandomResizedCrop", "CenterCrop", "RandomCrop", "Resize"} or not transform.args:
                    continue
                size = transform.args[0]
                if isinstance(size, (ast.Tuple, ast.List)) and len(size.elts) == 2:
                    height, width = (self._integer(item, transform) for item in size.elts)
                    area = height * width if height is not None and width is not None else None
                elif transform_name != "Resize":
                    side = self._integer(size, transform)
                    area = side * side if side is not None else None
                else:
                    area = None
        return max(areas) if areas else None

    def _non_image_dataset(self, loader: ast.Call) -> bool:
        dataset = self._dataset_constructor(loader)
        if not isinstance(dataset, ast.Call) or not isinstance(dataset.func, ast.Name):
            return False
        if any(
            isinstance(statement, ast.ImportFrom)
            and statement.module == "torch.utils.data"
            and any(
                imported.name == "TensorDataset"
                and (imported.asname or imported.name) == dataset.func.id
                for imported in statement.names
            )
            for statement in getattr(self.tree, "body", [])
        ):
            return True
        for statement in getattr(self.tree, "body", []):
            if not isinstance(statement, ast.ClassDef) or statement.name != dataset.func.id:
                continue
            getter = next(
                (member for member in statement.body
                 if isinstance(member, ast.FunctionDef) and member.name == "__getitem__"),
                None,
            )
            if getter is None:
                return False
            if any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"open", "convert"}
                for node in ast.walk(statement)
            ):
                return False
            inputs = [*dataset.args, *(item.value for item in dataset.keywords)]
            for argument in inputs:
                origin = self._binding(argument.id, dataset) if isinstance(argument, ast.Name) else argument
                if origin is not None and any(
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "convert"
                    and len(node.args) == 1
                    and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value == "RGB"
                    for node in ast.walk(origin)
                ):
                    return False
            return all(
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and (
                    (node.func.value.id == "torch" and node.func.attr == "tensor")
                    or (node.func.value.id == "np" and node.func.attr == "asarray")
                )
                for node in ast.walk(getter)
                if isinstance(node, ast.Call)
            )
        return False

    def diagnostics(self) -> list[dict]:
        if self.shared_memory_bytes is None or self.shared_memory_bytes < 0:
            return []
        for call in (node for node in ast.walk(self.tree) if isinstance(node, ast.Call)):
            func_name = call.func.attr if isinstance(call.func, ast.Attribute) else (
                call.func.id if isinstance(call.func, ast.Name) else ""
            )
            if func_name != "DataLoader":
                continue
            if self._non_image_dataset(call):
                continue
            image_area = self._image_area(call)
            if image_area is None:
                continue
            keywords = {item.arg: item.value for item in call.keywords}
            workers = self._integer(keywords.get("num_workers"), call)
            batch = self._integer(keywords.get("batch_size"), call)
            prefetch = self._integer(keywords.get("prefetch_factor"), call) or 2
            if not workers or not batch or workers < 2 or batch < 2:
                continue
            estimated_bytes = workers * prefetch * batch * 3 * image_area * 4
            if estimated_bytes > 2 * self.shared_memory_bytes:
                return [{
                    "code": "dataloader_shm_risk",
                    "line": call.lineno,
                    "message": (
                        f"DataLoader may need about {estimated_bytes // (1024 * 1024)} MiB "
                        f"of shared-memory image buffers, but only "
                        f"{self.shared_memory_bytes // (1024 * 1024)} MiB is available; "
                        "reduce workers/batch/prefetch or enlarge Docker --shm-size before execution"
                    ),
                }]
        return []


class SourceGpuOOMChecker:
    """Reject a hardware-specific training shape with a verified first-batch OOM."""

    def __init__(self, tree: ast.AST, bindings: SourceResourceChecker):
        self.tree = tree
        self.bindings = bindings

    def diagnostics(self) -> list[dict]:
        candidate_lists = {
            target.id
            for statement in getattr(self.tree, "body", [])
            if isinstance(statement, ast.Assign)
            and isinstance(statement.value, (ast.List, ast.Tuple))
            and statement.value.elts
            and isinstance(statement.value.elts[0], ast.Constant)
            and statement.value.elts[0].value == "convnext_base_384_in22ft1k"
            for target in statement.targets
            if isinstance(target, ast.Name)
        }
        if not candidate_lists or not any(
            isinstance(node, ast.For)
            and isinstance(node.iter, ast.Name)
            and node.iter.id in candidate_lists
            for node in ast.walk(self.tree)
        ):
            return []
        if not any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "timm"
            and node.func.attr == "create_model"
            for node in ast.walk(self.tree)
        ) or not any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "backward"
            for node in ast.walk(self.tree)
        ):
            return []
        risky_loader = None
        for call in (node for node in ast.walk(self.tree) if isinstance(node, ast.Call)):
            if name(call.func) != "DataLoader":
                continue
            keywords = {item.arg: item.value for item in call.keywords}
            if not isinstance(keywords.get("shuffle"), ast.Constant) or keywords["shuffle"].value is not True:
                continue
            batch = self.bindings._integer(keywords.get("batch_size"), call)
            area = self.bindings._image_area(call)
            if batch is not None and batch >= 64 and area is not None and area >= 384 * 384:
                risky_loader = call
                break
        if risky_loader is None:
            return []
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            gpu_name, total_mib = result.stdout.splitlines()[0].rsplit(",", 1)
            if result.returncode != 0 or gpu_name.strip() != "NVIDIA A10G" or int(total_mib.strip()) > 23028:
                return []
        except (OSError, subprocess.TimeoutExpired, IndexError, ValueError):
            return []
        return [{
            "code": "gpu_oom_known_configuration",
            "line": risky_loader.lineno,
            "message": (
                "ConvNeXt Base 384 with image batch at least 64 exhausted an NVIDIA A10G "
                "during its first training forward pass; reduce batch/image size or use "
                "a larger GPU before execution"
            ),
        }]


class SourceRuntimeBudgetChecker:
    """Reject an observed A10G training shape that exhausted a one-hour job."""

    def __init__(self, tree: ast.AST, bindings: SourceResourceChecker, timeout_seconds: int = 3600):
        self.tree = tree
        self.bindings = bindings
        self.timeout_seconds = timeout_seconds

    def _observed_petfinder_multibackbone(self) -> ast.For | None:
        """Locate the executed two-backbone Petfinder shape that timed out."""
        if (
            literal_assignment(self.tree, "BACKBONES")
            != ["convnext_small.fb_in22k_ft_in1k", "swin_small_patch4_window7_224"]
            or literal_assignment(self.tree, "IMG_SIZE") != 224
            or literal_assignment(self.tree, "BATCH_SIZE") != 32
            or literal_assignment(self.tree, "EPOCHS") != 7
            or literal_assignment(self.tree, "PATIENCE") != 3
            or not any(
                isinstance(node, ast.Constant) and node.value == "Pawpularity"
                for node in ast.walk(self.tree)
            )
        ):
            return None
        has_five_fold_split = any(
            isinstance(node, ast.Dict)
            and any(
                isinstance(key, ast.Constant)
                and key.value == "n_folds"
                and isinstance(item, ast.Constant)
                and item.value == 5
                for key, item in zip(node.keys, node.values)
            )
            for node in ast.walk(self.tree)
        )
        fold_assignments = assigned(self.tree, "n_folds")
        if len(fold_assignments) != 1:
            return None
        fold_value = value(fold_assignments[0])
        five_fold_variable = (
            isinstance(fold_value, ast.Constant)
            and fold_value.value == 5
        ) or (
            isinstance(fold_value, ast.Call)
            and name(fold_value.func) == "int"
            and len(fold_value.args) == 1
            and isinstance(fold_value.args[0], ast.Call)
            and name(fold_value.args[0].func) == "si.get"
            and len(fold_value.args[0].args) == 2
            and all(
                isinstance(arg, ast.Constant) and arg.value == expected
                for arg, expected in zip(fold_value.args[0].args, ("n_folds", 5))
            )
        )
        has_backward = any(
            isinstance(node, ast.FunctionDef)
            and node.name == "train_one_epoch"
            and has_call(node, "loss.backward")
            for node in ast.walk(self.tree)
        )
        has_model = has_call(self.tree, "timm.create_model")
        has_batch_loader = any(
            isinstance(node, ast.Call)
            and name(node.func) == "DataLoader"
            and self.bindings._integer(
                next((kw.value for kw in node.keywords if kw.arg == "batch_size"), None),
                node,
            ) == 32
            for node in ast.walk(self.tree)
        )
        if not all((has_five_fold_split, five_fold_variable, has_backward, has_model, has_batch_loader)):
            return None
        for fold in (node for node in ast.walk(self.tree) if isinstance(node, ast.For)):
            if not (
                isinstance(fold.iter, ast.Call)
                and name(fold.iter.func) == "range"
                and fold.iter.args
                and isinstance(fold.iter.args[0], ast.Name)
                and fold.iter.args[0].id == "n_folds"
            ):
                continue
            for backbone in (node for node in ast.walk(fold) if isinstance(node, ast.For)):
                if not (
                    isinstance(backbone.iter, ast.Call)
                    and name(backbone.iter.func) == "enumerate"
                    and backbone.iter.args
                    and isinstance(backbone.iter.args[0], ast.Name)
                    and backbone.iter.args[0].id == "BACKBONES"
                ):
                    continue
                if any(
                    isinstance(epoch, ast.For)
                    and any(
                        isinstance(part, ast.Name) and part.id == "EPOCHS"
                        for part in ast.walk(epoch.iter)
                    )
                    and has_call(epoch, "train_one_epoch")
                    for epoch in ast.walk(backbone)
                ):
                    return fold
        return None

    def diagnostics(self) -> list[dict]:
        if self.timeout_seconds is None or self.timeout_seconds > 3600:
            return []
        multibackbone_fold = self._observed_petfinder_multibackbone()
        if multibackbone_fold is not None:
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=3,
                    check=False,
                )
                gpu_name, _ = result.stdout.splitlines()[0].rsplit(",", 1)
                if result.returncode == 0 and gpu_name.strip() == "NVIDIA A10G":
                    return [{
                        "code": "gpu_runtime_budget_known_configuration",
                        "line": multibackbone_fold.lineno,
                        "message": (
                            "This observed Petfinder A10G two-backbone, five-fold, "
                            "seven-epoch shape exceeded a 3600-second candidate limit; "
                            "reduce the workload or provide a verified whole-script runtime budget"
                        ),
                    }]
            except (OSError, subprocess.TimeoutExpired, IndexError, ValueError):
                pass
        if (
            literal_assignment(self.tree, "BACKBONE_NAME") != "convnext_small.fb_in22k_ft_in1k"
            or literal_assignment(self.tree, "IMG_SIZE") != 384
            or literal_assignment(self.tree, "N_FOLDS") != 5
            or literal_assignment(self.tree, "NUM_EPOCHS") != 12
        ):
            return []
        model_used = False
        for node in ast.walk(self.tree):
            if not (
                isinstance(node, ast.Call)
                and name(node.func) == "timm.create_model"
                and node.args
                and isinstance(node.args[0], ast.Name)
            ):
                continue
            model_name = node.args[0].id
            if model_name == "BACKBONE_NAME":
                model_used = True
                break
            scope = self.bindings.parents.get(node)
            while scope is not None and not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
                scope = self.bindings.parents.get(scope)
            if scope is None:
                continue
            parameters = scope.args.posonlyargs + scope.args.args
            defaults = dict(zip(parameters[-len(scope.args.defaults):], scope.args.defaults)) if scope.args.defaults else {}
            if any(
                parameter.arg == model_name
                and isinstance(default, ast.Name)
                and default.id == "BACKBONE_NAME"
                for parameter, default in defaults.items()
            ):
                model_used = True
                break
        if not model_used:
            return []
        for fold_loop in (node for node in ast.walk(self.tree) if isinstance(node, ast.For)):
            if not (
                isinstance(fold_loop.iter, ast.Call)
                and name(fold_loop.iter.func) == "range"
                and fold_loop.iter.args
                and isinstance(fold_loop.iter.args[0], ast.Name)
                and fold_loop.iter.args[0].id == "N_FOLDS"
            ):
                continue
            epoch_loops = [
                node for node in ast.walk(fold_loop)
                if isinstance(node, ast.For)
                and any(
                    isinstance(part, ast.Name) and part.id == "NUM_EPOCHS"
                    for part in ast.walk(node.iter)
                )
                and has_call(node, "loss.backward")
            ]
            if not epoch_loops:
                continue
            for loader in (node for node in ast.walk(fold_loop) if isinstance(node, ast.Call)):
                if name(loader.func) != "DataLoader":
                    continue
                keywords = {item.arg: item.value for item in loader.keywords}
                if self.bindings._integer(keywords.get("batch_size"), loader) != 16:
                    continue
                if not (
                    isinstance(keywords.get("shuffle"), ast.Constant)
                    and keywords["shuffle"].value is True
                ):
                    continue
                try:
                    result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                        capture_output=True,
                        text=True,
                        timeout=3,
                        check=False,
                    )
                    gpu_name, _ = result.stdout.splitlines()[0].rsplit(",", 1)
                    if result.returncode != 0 or gpu_name.strip() != "NVIDIA A10G":
                        return []
                except (OSError, subprocess.TimeoutExpired, IndexError, ValueError):
                    return []
                return [{
                    "code": "gpu_runtime_budget_known_configuration",
                    "line": fold_loop.lineno,
                    "message": (
                        "This A10G ConvNeXt-Small 384, batch-16, five-fold training shape "
                        "exceeded a 3600-second candidate limit after only two folds; "
                        "reduce the workload or provide a verified whole-script runtime budget"
                    ),
                }]
        return []


class SourceShapeChecker:
    """Flag observed model-input and prediction-array shape hazards."""

    FIXED_224_MODELS = {
        "swin_tiny_patch4_window7_224",
        "swin_small_patch4_window7_224",
    }
    OBSERVED_FIXED_INPUT_MODELS = {
        "vit_base_patch14_dinov2.lvd142m": 518,
    }

    def __init__(self, tree: ast.AST, bindings: SourceResourceChecker):
        self.tree = tree
        self.bindings = bindings

    def _fixed_input_mismatch(self) -> list[dict]:
        model_calls = [
            node for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
            and name(node.func) == "timm.create_model"
            and node.args
            and isinstance(node.args[0], ast.Name)
            and not any(item.arg == "img_size" for item in node.keywords)
        ]
        if not model_calls:
            return []
        image_sizes = set()
        loaders = (
            node for node in ast.walk(self.tree)
            if isinstance(node, ast.Call) and name(node.func) == "DataLoader"
        )
        for loader in loaders:
            for reference in (node for node in ast.walk(loader) if isinstance(node, ast.Name)):
                transform = self.bindings._binding(reference.id, loader)
                if not (
                    isinstance(transform, ast.Call)
                    and isinstance(transform.func, ast.Attribute)
                    and transform.func.attr == "Compose"
                ):
                    continue
                for crop in (node for node in ast.walk(transform) if isinstance(node, ast.Call)):
                    if not (
                        isinstance(crop.func, ast.Attribute)
                        and crop.func.attr in {"RandomResizedCrop", "CenterCrop"}
                        and crop.args
                    ):
                        continue
                    size = self.bindings._integer(crop.args[0], crop)
                    if size is not None:
                        image_sizes.add(size)
        for loop in (node for node in ast.walk(self.tree) if isinstance(node, ast.For)):
            if not isinstance(loop.iter, ast.Name):
                continue
            jobs = self.bindings._binding(loop.iter.id, loop)
            if jobs is None:
                continue
            architecture_lists = {
                generator.iter.id
                for comprehension in ast.walk(jobs)
                if isinstance(comprehension, ast.ListComp)
                for generator in comprehension.generators
                if isinstance(generator.iter, ast.Name)
            }
            for model_call in model_calls:
                if model_call.args[0].id not in target_names(loop.target):
                    continue
                for list_name in architecture_lists:
                    candidates = self.bindings._binding(list_name, loop)
                    if not isinstance(candidates, (ast.List, ast.Tuple)):
                        continue
                    for candidate in candidates.elts:
                        if not (
                            isinstance(candidate, ast.Constant)
                            and candidate.value in self.FIXED_224_MODELS
                        ):
                            continue
                        for image_size in image_sizes:
                            if image_size != 224:
                                return [{
                                    "code": "fixed_input_image_size_mismatch",
                                    "line": candidate.lineno,
                                    "message": (
                                        f"{candidate.value} requires 224-pixel input, but the image "
                                        f"transform produces {image_size}-pixel crops; use a matching "
                                        "transform or a model configured for that input size"
                                    ),
                                }]
        # Generated fold ensembles can select a model by indexing a literal
        # backbone list, then pass it through a model class and factory helper.
        # Only flag this path when the list is used in such a constructor and
        # the actual data loader has a known square image transform.
        areas = {
            self.bindings._image_area(loader)
            for loader in ast.walk(self.tree)
            if isinstance(loader, ast.Call) and name(loader.func) == "DataLoader"
        }
        sides = {
            math.isqrt(area)
            for area in areas
            if isinstance(area, int) and math.isqrt(area) ** 2 == area
        }
        if not sides or sides == {224}:
            return []
        helpers = set()
        for model_call in model_calls:
            parent = self.bindings.parents.get(model_call)
            while parent is not None and not isinstance(parent, ast.FunctionDef):
                parent = self.bindings.parents.get(parent)
            if isinstance(parent, ast.FunctionDef):
                helpers.add(parent.name)
        model_classes = {
            cls.name
            for cls in ast.walk(self.tree)
            if isinstance(cls, ast.ClassDef)
            and any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id in helpers
                for call in ast.walk(cls)
            )
        }
        for loop in (node for node in ast.walk(self.tree) if isinstance(node, ast.For)):
            for statement in loop.body:
                if not (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                        and isinstance(statement.targets[0], ast.Name)
                        and isinstance(statement.value, ast.Subscript)
                        and isinstance(statement.value.value, ast.Name)):
                    continue
                selected = statement.targets[0].id
                candidates = self.bindings._binding(statement.value.value.id, statement)
                if not isinstance(candidates, (ast.List, ast.Tuple)):
                    continue
                if not any(
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id in model_classes
                    and any(isinstance(arg, ast.Name) and arg.id == selected
                            for value in call.args for arg in ast.walk(value))
                    for call in ast.walk(loop)
                ):
                    continue
                for candidate in candidates.elts:
                    if not (isinstance(candidate, ast.Constant)
                            and isinstance(candidate.value, str)
                            and candidate.value.split(".", 1)[0] in self.FIXED_224_MODELS):
                        continue
                    image_size = next((side for side in sorted(sides) if side != 224), None)
                    if image_size is not None:
                        return [{
                            "code": "fixed_input_image_size_mismatch",
                            "line": candidate.lineno,
                            "message": (
                                f"{candidate.value} requires 224-pixel input, but the image "
                                f"transform produces {image_size}-pixel images"
                            ),
                        }]
        return []

    def _validated_input_probe(self, model_call: ast.Call, image_size: int) -> bool:
        """Recognize a model checked at the loader size before it can be returned."""
        assignment = self.bindings.parents.get(model_call)
        if not (
            isinstance(assignment, ast.Assign)
            and assignment.value is model_call
            and len(assignment.targets) == 1
            and isinstance(assignment.targets[0], ast.Name)
        ):
            return False
        model_variable = assignment.targets[0].id
        guarded = self.bindings.parents.get(assignment)
        if not (
            isinstance(guarded, ast.Try)
            and assignment in guarded.body
            and any(
                handler.type is None
                or isinstance(handler.type, ast.Name)
                and handler.type.id in {"Exception", "BaseException"}
                for handler in guarded.handlers
            )
        ):
            return False
        probe_functions = {
            function.name: function
            for function in ast.walk(self.tree)
            if isinstance(function, ast.FunctionDef)
        }
        statements = guarded.body
        for index, statement in enumerate(statements[statements.index(assignment) + 1 :], 1):
            if not (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Call)
                and isinstance(statement.value.func, ast.Name)
            ):
                continue
            probe_call = statement.value
            if not (
                len(probe_call.args) >= 2
                and isinstance(probe_call.args[0], ast.Name)
                and probe_call.args[0].id == model_variable
                and self.bindings._integer(probe_call.args[1], probe_call) == image_size
            ):
                continue
            function = probe_functions.get(probe_call.func.id)
            if (
                function is None
                or len(function.args.args) < 2
                or any(isinstance(node, ast.ExceptHandler) for node in ast.walk(function))
            ):
                continue
            model_parameter, size_parameter = (arg.arg for arg in function.args.args[:2])
            probe_variables = {
                target.id
                for node in ast.walk(function)
                if isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call)
                and name(node.value.func) == "torch.zeros"
                and len(node.value.args) >= 4
                and all(
                    isinstance(node.value.args[position], ast.Name)
                    and node.value.args[position].id == size_parameter
                    for position in (2, 3)
                )
                for target in node.targets
                if isinstance(target, ast.Name)
            }
            if not any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == model_parameter
                and node.func.attr == "forward_features"
                and node.args
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id in probe_variables
                for node in ast.walk(function)
            ):
                continue
            if any(
                isinstance(later, ast.Return)
                and later.value is not None
                and any(
                    isinstance(part, ast.Name) and part.id == model_variable
                    for part in ast.walk(later.value)
                )
                for later in statements[statements.index(assignment) + index + 1 :]
            ):
                return True
        return False

    def _ordered_backbone_input_mismatch(self) -> list[dict]:
        """Check the first model selected by a literal fallback factory."""
        sides = {
            math.isqrt(area)
            for loader in ast.walk(self.tree)
            if isinstance(loader, ast.Call) and name(loader.func) == "DataLoader"
            if isinstance((area := self.bindings._image_area(loader)), int)
            and math.isqrt(area) ** 2 == area
        }
        if not sides:
            return []
        for loop in (node for node in ast.walk(self.tree) if isinstance(node, ast.For)):
            if not isinstance(loop.iter, ast.Name) or not isinstance(loop.target, ast.Name):
                continue
            enclosing = self.bindings.parents.get(loop)
            while enclosing is not None and not isinstance(enclosing, (ast.FunctionDef, ast.Module)):
                enclosing = self.bindings.parents.get(enclosing)
            if isinstance(enclosing, ast.FunctionDef) and not any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == enclosing.name
                for call in ast.walk(self.tree)
            ):
                continue
            candidates = self.bindings._binding(loop.iter.id, loop)
            if not isinstance(candidates, (ast.List, ast.Tuple)) or not candidates.elts:
                continue
            first = candidates.elts[0]
            if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
                continue
            required_size = self.OBSERVED_FIXED_INPUT_MODELS.get(first.value)
            if required_size is None:
                continue
            model_calls = [
                call for call in ast.walk(loop)
                if isinstance(call, ast.Call)
                and name(call.func) == "timm.create_model"
                and call.args
                and isinstance(call.args[0], ast.Name)
                and call.args[0].id == loop.target.id
                and not any(keyword.arg == "img_size" for keyword in call.keywords)
            ]
            if not model_calls or not any(isinstance(node, ast.Return) for node in ast.walk(loop)):
                continue
            image_size = next((side for side in sorted(sides) if side != required_size), None)
            if image_size is not None:
                if all(self._validated_input_probe(call, image_size) for call in model_calls):
                    continue
                return [{
                    "code": "fixed_input_image_size_mismatch",
                    "line": first.lineno,
                    "message": (
                        f"{first.value} requires {required_size}-pixel input in the "
                        f"observed timm runtime, but the image transform produces "
                        f"{image_size}-pixel images"
                    ),
                }]
        return []

    def _fixed_input_multiscale_resize(self) -> list[dict]:
        """Detect a non-unit prediction scale fed into a fixed-input Swin model."""
        fixed_models = []
        for call in ast.walk(self.tree):
            if not (
                isinstance(call, ast.Call)
                and name(call.func) == "timm.create_model"
                and call.args
            ):
                continue
            model = call.args[0]
            if isinstance(model, ast.Name):
                model = literal_assignment(self.tree, model.id)
            elif isinstance(model, ast.Constant):
                model = model.value
            if not isinstance(model, str) or model.split(".", 1)[0] not in self.FIXED_224_MODELS:
                continue
            primary = literal_assignment(self.tree, "BACKBONE")
            if isinstance(primary, str) and primary.split(".", 1)[0] not in self.FIXED_224_MODELS:
                parent = self.bindings.parents.get(call)
                while parent is not None and not isinstance(parent, ast.ExceptHandler):
                    parent = self.bindings.parents.get(parent)
                if parent is not None:
                    continue
            # Swin's PatchEmbed is controlled by strict_img_size; in the deployed
            # timm version dynamic_img_size is accepted but does not change it.
            if any(
                keyword.arg == "strict_img_size"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is False
                for keyword in call.keywords
            ):
                continue
            size = next((keyword.value for keyword in call.keywords if keyword.arg == "img_size"), None)
            if size is None:
                image_size = 224
            elif isinstance(size, ast.Name):
                image_size = literal_assignment(self.tree, size.id)
            elif isinstance(size, ast.Constant):
                image_size = size.value
            else:
                continue
            if type(image_size) is int and image_size > 0:
                fixed_models.append((model, image_size))
        if not fixed_models:
            return []

        for function in (node for node in ast.walk(self.tree) if isinstance(node, ast.FunctionDef)):
            parameters = function.args.posonlyargs + function.args.args
            defaults = dict(zip(parameters[-len(function.args.defaults):], function.args.defaults)) if function.args.defaults else {}
            for parameter, default in defaults.items():
                if not isinstance(default, (ast.Tuple, ast.List)):
                    continue
                scales = [item.value for item in default.elts if isinstance(item, ast.Constant)]
                nonunit = next(
                    (scale for scale in scales if type(scale) in {int, float} and scale > 0 and scale != 1),
                    None,
                )
                if nonunit is None:
                    continue
                parameter_index = parameters.index(parameter)
                if not any(
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id == function.name
                    and len(call.args) <= parameter_index
                    and not any(keyword.arg == parameter.arg for keyword in call.keywords)
                    for call in ast.walk(self.tree)
                ):
                    continue
                for loop in (node for node in ast.walk(function) if isinstance(node, ast.For)):
                    if not (
                        isinstance(loop.iter, ast.Name)
                        and loop.iter.id == parameter.arg
                        and isinstance(loop.target, ast.Name)
                    ):
                        continue
                    bindings = {
                        target.id: statement.value
                        for statement in ast.walk(loop)
                        if isinstance(statement, ast.Assign)
                        for target in statement.targets
                        if isinstance(target, ast.Name)
                    }

                    def uses_scale(node: ast.AST, seen: set[str]) -> bool:
                        if isinstance(node, ast.Name):
                            if node.id == loop.target.id:
                                return True
                            if node.id in bindings and node.id not in seen:
                                return uses_scale(bindings[node.id], seen | {node.id})
                        return any(uses_scale(child, seen) for child in ast.iter_child_nodes(node))

                    for statement in (node for node in ast.walk(loop) if isinstance(node, ast.Assign)):
                        if not (
                            len(statement.targets) == 1
                            and isinstance(statement.targets[0], ast.Name)
                            and isinstance(statement.value, ast.Call)
                            and name(statement.value.func).split(".")[-1] == "interpolate"
                        ):
                            continue
                        size = next(
                            (keyword.value for keyword in statement.value.keywords if keyword.arg == "size"),
                            None,
                        )
                        if size is None or not uses_scale(size, set()):
                            continue
                        image_name = statement.targets[0].id
                        model_call = next(
                            (
                                call for call in ast.walk(loop)
                                if isinstance(call, ast.Call)
                                and isinstance(call.func, ast.Name)
                                and call.func.id in {arg.arg for arg in parameters}
                                and call.lineno > statement.lineno
                                and any(
                                    isinstance(argument, ast.Name) and argument.id == image_name
                                    for argument in call.args
                                )
                            ),
                            None,
                        )
                        if model_call is not None:
                            model, image_size = fixed_models[0]
                            return [{
                                "code": "fixed_input_multiscale_resize",
                                "line": model_call.lineno,
                                "message": (
                                    f"{model} is configured for {image_size}-pixel input, but "
                                    f"{function.name}() feeds a {nonunit:g}-scaled image to the model; "
                                    "keep prediction inputs at the configured size"
                                ),
                            }]
        return []

    def diagnostics(self) -> list[dict]:
        mismatch = self._fixed_input_mismatch()
        if mismatch:
            return mismatch
        mismatch = self._ordered_backbone_input_mismatch()
        if mismatch:
            return mismatch
        mismatch = self._fixed_input_multiscale_resize()
        if mismatch:
            return mismatch
        for loop in (node for node in ast.walk(self.tree) if isinstance(node, ast.For)):
            if not isinstance(loop.iter, ast.Name):
                continue
            loader = self.bindings._binding(loop.iter.id, loop)
            if not isinstance(loader, ast.Call) or name(loader.func) != "DataLoader":
                continue
            batch_arg = next((item.value for item in loader.keywords if item.arg == "batch_size"), None)
            batch_size = self.bindings._integer(batch_arg, loader)
            if batch_size is None or batch_size < 2:
                continue
            batch_outputs = {
                target.id
                for statement in ast.walk(loop)
                if isinstance(statement, ast.Assign)
                and has_call(statement.value, "model")
                for target in statement.targets
                if isinstance(target, ast.Name)
            }
            if not batch_outputs:
                continue
            for statement in ast.walk(loop):
                if not (
                    isinstance(statement, ast.AugAssign)
                    and isinstance(statement.op, ast.Add)
                    and isinstance(statement.target, ast.Name)
                ):
                    continue
                accumulator = self.bindings._binding(statement.target.id, statement)
                if not (
                    isinstance(accumulator, ast.Call)
                    and name(accumulator.func) == "np.zeros"
                    and accumulator.args
                    and isinstance(accumulator.args[0], ast.Call)
                    and name(accumulator.args[0].func) == "len"
                ):
                    continue
                uses_batch_output = any(
                    isinstance(node, ast.Name) and node.id in batch_outputs
                    for node in ast.walk(statement.value)
                )
                converts_batch_to_array = any(
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "numpy"
                    for node in ast.walk(statement.value)
                )
                if uses_batch_output and converts_batch_to_array:
                    return [{
                        "code": "unindexed_batch_prediction_accumulation",
                        "line": statement.lineno,
                        "message": (
                            "A full-dataset np.zeros array is incremented by one model-batch "
                            "prediction without indexing; collect and concatenate batches "
                            "or add each batch to its matching slice"
                        ),
                    }]
        return []


class SourcePredictionTargetChecker:
    """Catch a test-ID/target mismatch across a bounded prediction call chain."""

    def __init__(self, tree: ast.AST, bindings: SourceResourceChecker):
        self.tree = tree
        self.bindings = bindings
        self.functions = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        self.classes = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
        }

    def _binding(self, variable: str, context: ast.AST) -> ast.AST | None:
        """Resolve a preceding assignment in enclosing loop/function bodies."""
        scope = context
        while scope is not None:
            body = getattr(scope, "body", None)
            if isinstance(body, list):
                matches = [
                    statement.value
                    for statement in body
                    if isinstance(statement, ast.Assign)
                    and statement.lineno < getattr(context, "lineno", 0)
                    and any(variable in target_names(target) for target in statement.targets)
                ]
                if matches:
                    return matches[-1]
            scope = self.bindings.parents.get(scope)
        return self.bindings._binding(variable, context)

    @staticmethod
    def _argument(
        function: ast.FunctionDef, call: ast.Call, parameter: str, *, constructor: bool = False
    ) -> ast.AST | None:
        parameters = function.args.posonlyargs + function.args.args
        if constructor and parameters and parameters[0].arg == "self":
            parameters = parameters[1:]
        for keyword in call.keywords:
            if keyword.arg == parameter:
                return keyword.value
        position = next((i for i, item in enumerate(parameters) if item.arg == parameter), None)
        if position is not None and position < len(call.args):
            return call.args[position]
        if position is not None:
            default_position = position - (len(parameters) - len(function.args.defaults))
            if default_position >= 0:
                return function.args.defaults[default_position]
        return None

    @staticmethod
    def _boolean(node: ast.AST | None, inherited: dict[str, bool]) -> bool | None:
        if isinstance(node, ast.Constant) and type(node.value) is bool:
            return node.value
        if isinstance(node, ast.Name):
            return inherited.get(node.id)
        return None

    @staticmethod
    def _string_id_source(initializer: ast.FunctionDef) -> bool:
        for statement in ast.walk(initializer):
            if not isinstance(statement, ast.Assign):
                continue
            if not any(
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and target.attr == "ids"
                for target in statement.targets
            ):
                continue
            value = statement.value
            if isinstance(value, (ast.List, ast.Tuple)) and value.elts and all(
                isinstance(item, ast.Constant) and isinstance(item.value, str)
                for item in value.elts
            ):
                return True
            if any(
                isinstance(item, ast.Constant) and item.value == "Id"
                for item in ast.walk(value)
            ):
                return True
        return False

    @staticmethod
    def _returns_ids_without_target(getitem: ast.FunctionDef) -> bool:
        for index, statement in enumerate(getitem.body):
            if not (
                isinstance(statement, ast.If)
                and name(statement.test) == "self.has_target"
            ):
                continue
            false_branch = statement.orelse or getitem.body[index + 1:]
            for returned in (item for branch in false_branch for item in ast.walk(branch)):
                if not isinstance(returned, ast.Return):
                    continue
                value = returned.value
                if not isinstance(value, (ast.Tuple, ast.List)) or not value.elts:
                    continue
                last = value.elts[-1]
                if isinstance(last, ast.Subscript) and name(last.value) == "self.ids":
                    return True
        return False

    def _loader_has_string_ids(self, loader_name: str, context: ast.Call) -> bool:
        loader = self._binding(loader_name, context)
        if not (
            isinstance(loader, ast.Call)
            and name(loader.func) == "DataLoader"
            and loader.args
            and isinstance(loader.args[0], ast.Name)
        ):
            return False
        dataset = self._binding(loader.args[0].id, loader)
        if not isinstance(dataset, ast.Call):
            return False
        dataset_class = self.classes.get(name(dataset.func))
        if dataset_class is None:
            return False
        methods = {
            member.name: member
            for member in dataset_class.body
            if isinstance(member, ast.FunctionDef)
        }
        initializer = methods.get("__init__")
        getitem = methods.get("__getitem__")
        if initializer is None or getitem is None:
            return False
        target_flag = self._argument(initializer, dataset, "has_target", constructor=True)
        return (
            self._boolean(target_flag, {}) is False
            and self._string_id_source(initializer)
            and self._returns_ids_without_target(getitem)
        )

    @staticmethod
    def _multiplies_batch_extra(function: ast.FunctionDef, flag: bool) -> bool:
        if not flag:
            return False
        for loop in ast.walk(function):
            if not (
                isinstance(loop, ast.For)
                and isinstance(loop.iter, ast.Name)
                and loop.iter.id == "loader"
                and isinstance(loop.target, (ast.Tuple, ast.List))
                and loop.target.elts
                and isinstance(loop.target.elts[-1], ast.Name)
            ):
                continue
            extra_name = loop.target.elts[-1].id
            for branch in ast.walk(loop):
                if not isinstance(branch, ast.If) or name(branch.test) != "return_targets":
                    continue
                for statement in branch.body:
                    for expression in ast.walk(statement):
                        if not isinstance(expression, ast.BinOp) or not isinstance(expression.op, ast.Mult):
                            continue
                        operands = (expression.left, expression.right)
                        if any(isinstance(item, ast.Name) and item.id == extra_name for item in operands) and any(
                            isinstance(item, ast.Constant) and type(item.value) in {int, float}
                            for item in operands
                        ):
                            return True
        return False

    def _consumes_ids(self, function: ast.FunctionDef, call: ast.Call, inherited: dict[str, bool], depth: int) -> bool:
        if depth > 2:
            return False
        flag = self._boolean(self._argument(function, call, "return_targets"), inherited)
        if self._multiplies_batch_extra(function, flag is True):
            return True
        for nested in ast.walk(function):
            if not isinstance(nested, ast.Call):
                continue
            callee = self.functions.get(name(nested.func))
            if callee is None or callee is function:
                continue
            loader = self._argument(callee, nested, "loader")
            if not isinstance(loader, ast.Name) or loader.id != "loader":
                continue
            if self._consumes_ids(callee, nested, {"return_targets": flag}, depth + 1):
                return True
        return False

    def diagnostics(self) -> list[dict]:
        for call in ast.walk(self.tree):
            if not isinstance(call, ast.Call):
                continue
            function = self.functions.get(name(call.func))
            if function is None:
                continue
            loader = self._argument(function, call, "loader")
            if not isinstance(loader, ast.Name) or not self._loader_has_string_ids(loader.id, call):
                continue
            if self._consumes_ids(function, call, {}, 0):
                return [{
                    "code": "test_ids_used_as_numeric_targets",
                    "line": call.lineno,
                    "message": (
                        "A test DataLoader returns string IDs, but this prediction call "
                        "collects and multiplies them as numeric targets; pass "
                        "return_targets=False or use a target-bearing loader"
                    ),
                }]
        return []


class SourceImportChecker:
    """Check unconditional imports in the interpreter that will run the candidate."""

    def __init__(self, tree: ast.AST, available_modules: set[str] | None = None):
        self.tree = tree
        self.available_modules = available_modules

    def diagnostics(self) -> list[dict]:
        diagnostics = []
        checked = set()
        for statement in getattr(self.tree, "body", []):
            if isinstance(statement, ast.Import):
                modules = [item.name for item in statement.names]
            elif isinstance(statement, ast.ImportFrom) and statement.level == 0 and statement.module:
                modules = [statement.module]
            else:
                continue
            for module in modules:
                root = module.split(".", 1)[0]
                if root in checked:
                    continue
                checked.add(root)
                if self.available_modules is not None:
                    available = root in self.available_modules
                else:
                    try:
                        available = importlib.util.find_spec(root) is not None
                    except (ImportError, ValueError, AttributeError):
                        available = False
                if not available:
                    diagnostics.append({
                        "code": "unavailable_import",
                        "line": statement.lineno,
                        "message": (
                            f"Unconditional import {root!r} is unavailable in this "
                            "candidate Python environment"
                        ),
                    })
        return diagnostics


class SourceLibraryApiChecker:
    """Catch bounded, known third-party call-signature mismatches."""

    def __init__(self, tree: ast.AST):
        self.tree = tree

    def _timm_model_names(self) -> list[tuple[int, str]]:
        """Resolve direct names and literal lists forwarded from training loops."""
        aliases = {
            alias.asname or "timm"
            for statement in getattr(self.tree, "body", [])
            if isinstance(statement, ast.Import)
            for alias in statement.names
            if alias.name == "timm"
        }
        calls = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in aliases
            and node.func.attr == "create_model"
        ]
        if not calls:
            return []
        models = [
            (call.lineno, call.args[0].value)
            for call in calls
            if call.args
            and isinstance(call.args[0], ast.Constant)
            and isinstance(call.args[0].value, str)
        ]
        callables = {}
        for statement in getattr(self.tree, "body", []):
            definition = statement
            if isinstance(statement, ast.ClassDef):
                definition = next(
                    (
                        member for member in statement.body
                        if isinstance(member, ast.FunctionDef) and member.name == "__init__"
                    ),
                    None,
                )
            if not isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            positional = [argument.arg for argument in definition.args.posonlyargs + definition.args.args]
            if isinstance(statement, ast.ClassDef) and positional and positional[0] == "self":
                positional = positional[1:]
            parameters = positional + [argument.arg for argument in definition.args.kwonlyargs]
            callables[statement.name] = (definition, positional, parameters)

        def argument_for(call: ast.Call, parameter: str, positional: list[str]):
            keyword = next((item.value for item in call.keywords if item.arg == parameter), None)
            if keyword is not None:
                return keyword
            if parameter in positional:
                index = positional.index(parameter)
                if index < len(call.args):
                    return call.args[index]
            return None

        sensitive = {key: set() for key in callables}
        for key, (definition, _, parameters) in callables.items():
            sensitive[key].update(
                call.args[0].id
                for call in ast.walk(definition)
                if isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id in aliases
                and call.func.attr == "create_model"
                and call.args
                and isinstance(call.args[0], ast.Name)
                and call.args[0].id in parameters
            )
        for _ in callables:
            changed = False
            for key, (definition, _, parameters) in callables.items():
                for call in (node for node in ast.walk(definition) if isinstance(node, ast.Call)):
                    if not isinstance(call.func, ast.Name) or call.func.id not in callables:
                        continue
                    callee = call.func.id
                    _, positional, _ = callables[callee]
                    for parameter in sensitive[callee]:
                        actual = argument_for(call, parameter, positional)
                        if isinstance(actual, ast.Name) and actual.id in parameters and actual.id not in sensitive[key]:
                            sensitive[key].add(actual.id)
                            changed = True
            if not changed:
                break

        for statement in getattr(self.tree, "body", []):
            if not isinstance(statement, ast.Assign):
                continue
            try:
                values = ast.literal_eval(statement.value)
            except (ValueError, TypeError, MemoryError, RecursionError):
                continue
            if not isinstance(values, (list, tuple)) or not values or not all(
                isinstance(value, str) for value in values
            ):
                continue
            for target in statement.targets:
                if not isinstance(target, ast.Name):
                    continue
                for loop in (node for node in ast.walk(self.tree) if isinstance(node, ast.For)):
                    iterable = loop.iter
                    if (
                        isinstance(iterable, ast.Call)
                        and isinstance(iterable.func, ast.Name)
                        and iterable.func.id == "enumerate"
                        and len(iterable.args) == 1
                    ):
                        iterable = iterable.args[0]
                    if not isinstance(iterable, ast.Name) or iterable.id != target.id:
                        continue
                    loop_names = {
                        node.id for node in ast.walk(loop.target) if isinstance(node, ast.Name)
                    }
                    forwarded = False
                    for body in loop.body:
                        for call in (node for node in ast.walk(body) if isinstance(node, ast.Call)):
                            if isinstance(call.func, ast.Name) and call.func.id in callables:
                                callee = call.func.id
                                _, positional, _ = callables[callee]
                                actuals = (
                                    argument_for(call, parameter, positional)
                                    for parameter in sensitive[callee]
                                )
                            elif (
                                isinstance(call.func, ast.Attribute)
                                and isinstance(call.func.value, ast.Name)
                                and call.func.value.id in aliases
                                and call.func.attr == "create_model"
                            ):
                                actuals = (call.args[0],) if call.args else ()
                            else:
                                continue
                            if any(
                                isinstance(actual, ast.Name) and actual.id in loop_names
                                for actual in actuals
                            ):
                                forwarded = True
                                break
                        if forwarded:
                            break
                    if forwarded:
                        models.extend((statement.lineno, value) for value in values)
                        break
        return list(dict.fromkeys(models))

    def diagnostics(self) -> list[dict]:
        timm_erasing_names = {
            alias.asname or alias.name
            for statement in getattr(self.tree, "body", [])
            if isinstance(statement, ast.ImportFrom)
            and statement.module in {"timm.data", "timm.data.random_erasing"}
            for alias in statement.names
            if alias.name == "RandomErasing"
        }
        diagnostics = [
            {
                "code": "timm_random_erasing_invalid_keyword",
                "line": call.lineno,
                "message": (
                    "timm RandomErasing does not accept p=; use probability= "
                    "(torchvision RandomErasing uses p=)"
                ),
            }
            for call in ast.walk(self.tree)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id in timm_erasing_names
            and any(keyword.arg == "p" for keyword in call.keywords)
        ]
        model_names = self._timm_model_names()
        if model_names:
            try:
                timm = importlib.import_module("timm")
                is_model = timm.is_model
            except (ImportError, AttributeError, RuntimeError):
                is_model = None
            if is_model is not None:
                diagnostics.extend(
                    {
                        "code": "timm_unknown_model",
                        "line": line,
                        "message": (
                            f"timm model {model!r} is not registered in this "
                            "candidate Python environment"
                        ),
                    }
                    for line, model in model_names
                    if not is_model(model)
                )
        return diagnostics


class SourceCachedMetadataChecker:
    """Check a reused local pickle's required keys without unpickling it."""

    def __init__(self, tree: ast.AST, working_directory: str | os.PathLike[str]):
        self.tree = tree
        self.root = Path(working_directory).resolve()
        assignments: dict[str, list[ast.AST]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assignments.setdefault(target.id, []).append(node.value)
        self.assignments = assignments

    def _path(self, node: ast.AST, seen: frozenset[str] = frozenset()) -> Path | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return Path(node.value)
        if isinstance(node, ast.Name) and node.id not in seen:
            values = self.assignments.get(node.id, [])
            if len(values) == 1:
                return self._path(values[0], seen | {node.id})
        if isinstance(node, ast.Call) and name(node.func) == "os.path.join" and node.args:
            parts = [self._path(arg, seen) for arg in node.args]
            if all(part is not None for part in parts):
                result = parts[0]
                for part in parts[1:]:
                    result = result / part
                return result
        return None

    def _within_workspace(self, node: ast.AST) -> Path | None:
        relative = self._path(node)
        if relative is None:
            return None
        path = (self.root / relative).resolve()
        return path if path.is_relative_to(self.root) else None

    def _opened_path(self, node: ast.AST, mode: str) -> Path | None:
        if not isinstance(node, ast.Call) or name(node.func) != "open" or not node.args:
            return None
        actual_mode = node.args[1].value if len(node.args) > 1 and isinstance(node.args[1], ast.Constant) else "r"
        return self._within_workspace(node.args[0]) if actual_mode == mode else None

    def _existing(self, node: ast.AST) -> bool | None:
        if isinstance(node, ast.BoolOp):
            values = [self._existing(value) for value in node.values]
            if any(value is None for value in values):
                return None
            return all(values) if isinstance(node.op, ast.And) else any(values)
        if isinstance(node, ast.Call) and name(node.func) == "os.path.exists" and len(node.args) == 1:
            path = self._within_workspace(node.args[0])
            return path.exists() if path is not None else None
        return None

    def _builder_functions(self, path: Path) -> set[str]:
        result = set()
        for function in ast.walk(self.tree):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if any(
                self._opened_path(item.context_expr, "wb") == path
                for with_node in ast.walk(function)
                if isinstance(with_node, ast.With)
                for item in with_node.items
            ):
                result.add(function.name)
        return result

    def _skips_rebuild(self, path: Path) -> bool:
        builders = self._builder_functions(path)
        if not builders:
            return False
        for branch in ast.walk(self.tree):
            if not isinstance(branch, ast.If) or self._existing(branch.test) is not True:
                continue
            checked_paths = {
                self._within_workspace(call.args[0])
                for call in ast.walk(branch.test)
                if isinstance(call, ast.Call)
                and name(call.func) == "os.path.exists"
                and len(call.args) == 1
            }
            if path not in checked_paths:
                continue
            body_calls = {
                name(call.func) for statement in branch.body
                for call in ast.walk(statement) if isinstance(call, ast.Call)
            }
            else_calls = {
                name(call.func) for statement in branch.orelse
                for call in ast.walk(statement) if isinstance(call, ast.Call)
            }
            if builders & else_calls and not builders & body_calls:
                return True
        return False

    def diagnostics(self) -> list[dict]:
        diagnostics = []
        for with_node in ast.walk(self.tree):
            if not isinstance(with_node, ast.With):
                continue
            for item in with_node.items:
                path = self._opened_path(item.context_expr, "rb")
                if path is None or not path.name.endswith("meta.pkl") or not path.is_file():
                    continue
                if not self._skips_rebuild(path):
                    continue
                handle = item.optional_vars.id if isinstance(item.optional_vars, ast.Name) else None
                loaded_names = {
                    target.id
                    for statement in with_node.body
                    for assign in ast.walk(statement)
                    if isinstance(assign, ast.Assign)
                    and isinstance(assign.value, ast.Call)
                    and name(assign.value.func) == "pickle.load"
                    and assign.value.args
                    and isinstance(assign.value.args[0], ast.Name)
                    and assign.value.args[0].id == handle
                    for target in assign.targets
                    if isinstance(target, ast.Name)
                }
                if not loaded_names:
                    continue
                required = {
                    node.slice.value: node.lineno
                    for node in ast.walk(self.tree)
                    if isinstance(node, ast.Subscript)
                    and isinstance(node.value, ast.Name)
                    and node.value.id in loaded_names
                    and isinstance(node.slice, ast.Constant)
                    and isinstance(node.slice.value, str)
                }
                if not required:
                    continue
                try:
                    if path.stat().st_size > 4 * 1024 * 1024:
                        continue
                    strings = {
                        argument for opcode, argument, _ in pickletools.genops(path.read_bytes())
                        if isinstance(argument, str) and "UNICODE" in opcode.name
                    }
                except (OSError, ValueError):
                    continue
                for key, line in required.items():
                    if key not in strings:
                        diagnostics.append({
                            "code": "cached_metadata_missing_key",
                            "line": line,
                            "message": (
                                f"existing {path.name} lacks required key {key!r}; "
                                "this candidate skips its cache rebuild"
                            ),
                        })
        return diagnostics


def check_source(
    source: str,
    filename: str = "<candidate>",
    *,
    shared_memory_bytes: int | None = None,
    execution_timeout_seconds: int | None = 3600,
    working_directory: str | os.PathLike[str] | None = None,
    available_modules: set[str] | None = None,
) -> dict:
    """Bounded admission check for arbitrary agent-generated Python scripts.

    The corpus-specific model and dataset rules below require a declared task
    contract and must not be applied to unrelated ML-Master/MLEvolve scripts.
    """
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        return {
            "status": "rejected",
            "admitted": False,
            "assurance": "syntax_only",
            "issues": ["syntax_error"],
            "diagnostics": [
                {"code": "syntax_error", "message": str(exc), "line": exc.lineno}
            ],
        }
    if shared_memory_bytes is None:
        try:
            stats = os.statvfs("/dev/shm")
            shared_memory_bytes = stats.f_bavail * stats.f_frsize
        except OSError:
            shared_memory_bytes = None
    resource_checker = SourceResourceChecker(tree, shared_memory_bytes)
    diagnostics = (
        resource_checker.diagnostics()
        + SourceGpuOOMChecker(tree, resource_checker).diagnostics()
        + SourceRuntimeBudgetChecker(
            tree, resource_checker, execution_timeout_seconds
        ).diagnostics()
        + SourceShapeChecker(tree, resource_checker).diagnostics()
        + SourcePredictionTargetChecker(tree, resource_checker).diagnostics()
        + SourceImportChecker(tree, available_modules).diagnostics()
        + SourceLibraryApiChecker(tree).diagnostics()
        + (SourceCachedMetadataChecker(tree, working_directory).diagnostics()
           if working_directory is not None else [])
    )
    if diagnostics:
        return {
            "status": "rejected",
            "admitted": False,
            "assurance": (
                "bounded_resource_risk"
                if all(item["code"] in {"dataloader_shm_risk", "gpu_oom_known_configuration", "gpu_runtime_budget_known_configuration"} for item in diagnostics)
                else "bounded_static_risk"
            ),
            "issues": [item["code"] for item in diagnostics],
            "diagnostics": diagnostics,
        }
    return {
        "status": "syntax_valid",
        "admitted": True,
        "assurance": "syntax_only",
        "issues": [],
        "diagnostics": [],
    }


class NodeValidator:
    """Validate one declared source node without executing it."""

    def __init__(self, path, cache):
        self.path = path
        self.cache = cache
        self.issues = set()

    def _check_names(self, tree):
        """Find unresolved source names before inspecting corpus-specific rules."""
        functions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        globals_ = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        }
        globals_.update(
            name
            for node in tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            for name in target_names(target)
        )
        for node in tree.body:
            if isinstance(node, ast.Import):
                globals_.update(
                    alias.asname or alias.name.split(".")[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom):
                globals_.update(alias.asname or alias.name for alias in node.names)
        for function in functions:
            if local_unknowns(function, globals_):
                self.issues.add("undefined_variable")
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and name(node.func.value) == "F"
            ):
                if node.func.attr not in FUNCTIONAL:
                    self.issues.add("undefined_function")
        return functions

    def _check_node_data(self, tree):
        """Check the controlled node's declared table and tensor shape."""
        data_file = literal_assignment(tree, "DATA_FILE")
        if literal_assignment(tree, "LABEL_COLUMN") != "label":
            self.issues.add("wrong_column")
        if literal_assignment(tree, "IMAGE_SHAPE") != (1, 16, 16):
            self.issues.add("bad_reshape")
        if literal_assignment(tree, "NUM_CLASSES") != 2:
            self.issues.add("wrong_output_classes")
        if not isinstance(data_file, str) or Path(data_file).name != data_file:
            self.issues.add("missing_file")
            return
        feature_expr = next(
            (value(node) for node in assigned(tree, "FEATURE_COLUMNS")), None
        )
        expected_features = 256 if "range(256)" in unparse(feature_expr) else 257
        if expected_features != 256:
            self.issues.add("wrong_feature")
        contract = DatasetContract(
            "tabular_csv",
            classes=2,
            feature_columns=tuple(
                f"pixel_{index}" for index in range(expected_features)
            ),
        )
        self.issues.update(
            dataset_issues(ROOT / "data" / data_file, contract, self.cache)
        )

    def _check_node_loader(self, load):
        """Check data conversion in load_data()."""
        labels = assigned(load, "labels")
        if not labels or "to_numpy(dtype=np.int64" not in unparse(value(labels[0])):
            text = unparse(value(labels[0])) if labels else ""
            if "nan" in text:
                self.issues.add("nan_to_int")
            elif "inf" in text:
                self.issues.add("inf_to_int")
            else:
                self.issues.add("invalid_label_conversion")
        images = assigned(load, "images")
        image_text = unparse(value(images[0])) if images else ""
        if "torch.from_numpy(features)" not in image_text:
            self.issues.add("numpy_to_model")
        if "reshape(-1, 1, 16, 16)" not in image_text:
            self.issues.add("bad_reshape")
        targets = assigned(load, "targets")
        target_text = unparse(value(targets[0])) if targets else ""
        if "np.asarray" in target_text:
            self.issues.add("torch_to_numpy")
        elif ".float()" in target_text:
            self.issues.add("float_targets")
        elif target_text != "torch.from_numpy(labels)":
            self.issues.add("invalid_target_conversion")

    def _check_training_setup(self, run):
        """Check device placement and the training loss expression."""
        model_assignment = assigned(run, "model")
        if (
            not model_assignment
            or unparse(value(model_assignment[0])) != "Model().to(device)"
        ):
            self.issues.add("model_on_cpu")
        train_move = next(
            (
                node
                for node in run.body
                if isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and target_names(node.targets[0]) == {"xtrain", "ytrain"}
                and ".to(" in unparse(value(node))
            ),
            None,
        )
        if train_move is None or "xtrain.to(device)" not in unparse(value(train_move)):
            self.issues.add("inputs_on_cpu")
        if train_move is None or "ytrain.to(device)" not in unparse(value(train_move)):
            self.issues.add("targets_on_cpu")
        losses = assigned(run, "loss")
        loss_text = unparse(value(losses[0])) if losses else ""
        if ".detach()" in loss_text:
            self.issues.add("detached_loss")
        elif "torch.tensor(" in loss_text or ".item()" in loss_text:
            self.issues.add("reboxed_loss")
        elif "F.one_hot" in loss_text:
            self.issues.add("one_hot_targets")
        elif loss_text != "F.cross_entropy(logits, ytrain)":
            if "cross_entropy_typo" not in loss_text:
                self.issues.add("invalid_train_loss")
        logits = assigned(run, "logits")
        if logits and "F.softmax" in unparse(value(logits[0])):
            self.issues.add("softmax_before_loss")

    def _check_gradient_flow(self, run):
        """Check training calls that can disconnect or disable gradients."""
        for node in ast.walk(run):
            if isinstance(node, ast.Call) and name(node.func) == "torch.autograd":
                self.issues.add("invalid_autograd_context")
            if (
                isinstance(node, ast.Call)
                and name(node.func) == "torch.set_grad_enabled"
                and node.args
                and unparse(node.args[0]) == "False"
            ):
                self.issues.add("global_grad_disabled")
            if isinstance(node, ast.With) and has_call(node, "torch.no_grad"):
                if any(
                    isinstance(child, ast.Call)
                    and name(child.func) == "model"
                    and any(unparse(arg) == "xtrain" for arg in child.args)
                    for child in ast.walk(node)
                ):
                    self.issues.add("no_grad_train")
            if (
                isinstance(node, ast.Call)
                and name(node.func) == "requires_grad_"
                and node.args
                and unparse(node.args[0]) == "False"
            ):
                self.issues.add("all_parameters_frozen")
            if (
                isinstance(node, ast.Call)
                and name(node.func) == "parameter.requires_grad_"
                and node.args
                and unparse(node.args[0]) == "False"
            ):
                self.issues.add("all_parameters_frozen")

    def _check_validation(self, run):
        """Check validation split and target usage."""
        val_split = next(
            (
                node
                for node in run.body
                if isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and target_names(node.targets[0]) == {"xval", "yval"}
            ),
            None,
        )
        if (
            val_split is None
            or "images[24:]" not in unparse(value(val_split))
            or "targets[24:]" not in unparse(value(val_split))
        ):
            self.issues.add("validation_on_train")
        val_losses = assigned(run, "val_loss")
        if not val_losses or "F.cross_entropy(val_logits, yval)" != unparse(
            value(val_losses[0])
        ):
            self.issues.add("wrong_validation_target")

    def _check_model_head(self, tree):
        """Check the classifier output width."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Attribute) and target.attr == "head"
                for target in node.targets
            ):
                if (
                    isinstance(node.value, ast.Call)
                    and name(node.value.func) == "nn.Linear"
                    and len(node.value.args) >= 2
                ):
                    if (
                        not isinstance(node.value.args[1], ast.Constant)
                        or node.value.args[1].value != 2
                    ):
                        self.issues.add("wrong_output_classes")

    def _check_optimizer_steps(self, run):
        """Check gradient reset and backward/step ordering."""
        if "optimizer.zero_grad(" not in unparse(run):
            self.issues.add("no_zero_grad")
        backward_line = next(
            (
                node.lineno
                for node in ast.walk(run)
                if isinstance(node, ast.Call) and name(node.func) == "loss.backward"
            ),
            None,
        )
        step_line = next(
            (
                node.lineno
                for node in ast.walk(run)
                if isinstance(node, ast.Call) and name(node.func) == "optimizer.step"
            ),
            None,
        )
        if backward_line is None:
            self.issues.add("missing_backward")
        elif step_line is None or step_line < backward_line:
            self.issues.add("optimizer_before_backward")

    def check(self):
        try:
            source = self.path.read_text()
            tree = ast.parse(source, filename=str(self.path))
        except (OSError, UnicodeError, SyntaxError):
            return ["unreadable_or_syntax_error"]

        if literal_assignment(tree, "TASK") is not None:
            return check_kaggle(self.path, tree, self.cache)
        if literal_assignment(tree, "MODALITY") is not None:
            return check_modality(self.path, tree, self.cache)

        functions = self._check_names(tree)
        self._check_node_data(tree)
        load = next(
            (function for function in functions if function.name == "load_data"), None
        )
        run = next((function for function in functions if function.name == "run"), None)
        if load is None or run is None:
            return sorted(self.issues | {"missing_pipeline"})

        self._check_node_loader(load)
        self._check_training_setup(run)
        self._check_gradient_flow(run)
        self._check_validation(run)
        self._check_model_head(tree)
        self._check_optimizer_steps(run)
        return sorted(self.issues)


def check(path, cache):
    """Compatibility entry point for a declared source node."""
    return NodeValidator(path, cache).check()


class PreflightApp:
    """Parse command-line requests and report bounded validation results."""

    def _run_bounded_competition(self):
        """Run a competition audit in a child process with a hard deadline."""
        args = self.args
        child = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--competition",
            args.competition,
            "--dataset",
            str(args.dataset),
            "--json",
        ]
        if args.asset_root is not None:
            child.extend(("--asset-root", str(args.asset_root)))
        started = time.perf_counter()
        try:
            process = subprocess.run(
                child,
                capture_output=True,
                text=True,
                check=False,
                timeout=args.max_seconds,
            )
        except subprocess.TimeoutExpired:
            report = {
                "competition": args.competition,
                "dataset": str(args.dataset),
                "status": "incomplete",
                "issues": ["audit_incomplete"],
                "assurance": "configured_checks_only",
                "elapsed_seconds": time.perf_counter() - started,
            }
            if args.json:
                print(json.dumps(report, indent=2))
            else:
                print(
                    f"{args.competition}: INCOMPLETE after {report['elapsed_seconds']:.4f}s"
                )
            raise SystemExit(2)
        try:
            report = json.loads(process.stdout)
            if process.returncode not in (0, 1) or not isinstance(
                report.get("issues"), list
            ):
                raise ValueError("invalid child result")
        except (json.JSONDecodeError, ValueError, AttributeError):
            report = {
                "competition": args.competition,
                "dataset": str(args.dataset),
                "status": "incomplete",
                "issues": ["audit_error"],
                "assurance": "configured_checks_only",
            }
            process = None
        report["elapsed_seconds"] = time.perf_counter() - started
        report["status"] = "incomplete" if process is None else "complete"
        report["assurance"] = "configured_checks_only"
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            suffix = (
                "INCOMPLETE"
                if process is None
                else ", ".join(report["issues"])
                if report["issues"]
                else "clean"
            )
            print(f"{args.competition}: {suffix} ({report['elapsed_seconds']:.4f}s)")
        raise SystemExit(2 if process is None else process.returncode)

    def _run_competition(self):
        """Validate one competition index directly."""
        args = self.args
        asset_root = args.asset_root or args.dataset.parent
        start = time.perf_counter()
        issues = sorted(
            kaggle_dataset_issues(args.competition, args.dataset, asset_root, {})
        )
        report = {
            "competition": args.competition,
            "dataset": str(args.dataset),
            "asset_root": str(asset_root),
            "issues": issues,
            "status": "complete",
            "assurance": "configured_checks_only",
            "elapsed_seconds": time.perf_counter() - start,
        }
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(
                f"{args.competition}: {', '.join(issues) if issues else 'clean'} "
                f"({report['elapsed_seconds']:.4f}s)"
            )
        raise SystemExit(1 if issues else 0)

    def _run_dataset_contract(self):
        """Validate a dataset without a model node."""
        args = self.args
        try:
            payload = json.loads(args.contract.read_text())
            for key in ("feature_columns", "image_size"):
                if key in payload:
                    payload[key] = tuple(payload[key])
            contract = DatasetContract(**payload)
            start = time.perf_counter()
            issues = sorted(dataset_issues(args.dataset, contract, {}))
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ) as exc:
            self.parser.error(str(exc))
        report = {
            "dataset": str(args.dataset),
            "kind": contract.kind,
            "issues": issues,
            "elapsed_seconds": time.perf_counter() - start,
        }
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(
                f"{args.dataset}: {', '.join(issues) if issues else 'clean'} "
                f"({report['elapsed_seconds']:.4f}s)"
            )
        raise SystemExit(1 if issues else 0)

    def _run_nodes(self):
        """Check requested nodes or the controlled corpus manifest."""
        args = self.args
        manifest_path = ROOT / "manifest.json"
        if not args.paths and not manifest_path.is_file():
            self.parser.error("manifest.json is missing; supply node paths")
        paths = args.paths or [
            ROOT / entry["path"]
            for entry in json.loads(manifest_path.read_text())
        ]
        start = time.perf_counter()
        cache = {}
        findings = {
            str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path): check(
                path, cache
            )
            for path in paths
        }
        elapsed = time.perf_counter() - start
        report = {
            "nodes": len(paths),
            "flagged": sum(bool(issues) for issues in findings.values()),
            "elapsed_seconds": elapsed,
            "issues": findings,
        }
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(
                f"checked={report['nodes']} flagged={report['flagged']} elapsed={elapsed:.4f}s"
            )
            for path, issues in findings.items():
                if issues:
                    print(f"{path}: {', '.join(issues)}")
        raise SystemExit(1 if report["flagged"] else 0)

    def run(self):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "paths",
            nargs="*",
            type=Path,
            help="node files; defaults to every manifest node",
        )
        parser.add_argument(
            "--json", action="store_true", help="print all findings as JSON"
        )
        parser.add_argument(
            "--dataset",
            type=Path,
            help="validate a dataset directly, without a model node",
        )
        parser.add_argument("--contract", type=Path, help="JSON contract for --dataset")
        parser.add_argument(
            "--competition",
            help="Kaggle competition slug for direct dataset validation",
        )
        parser.add_argument(
            "--asset-root", type=Path, help="competition root containing asset folders"
        )
        parser.add_argument(
            "--max-seconds",
            type=float,
            help="time limit for a full competition scan; returns incomplete if exceeded",
        )
        self.parser = parser
        args = self.args = parser.parse_args()
        if args.max_seconds is not None:
            if (
                args.competition is None
                or args.dataset is None
                or args.max_seconds <= 0
                or args.contract is not None
                or args.paths
            ):
                parser.error(
                    "--max-seconds requires --competition and --dataset, without --contract or node paths"
                )
            self._run_bounded_competition()

        if args.competition is not None:
            if args.dataset is None or args.contract is not None or args.paths:
                parser.error(
                    "--competition requires --dataset, without --contract or node paths"
                )
            if args.competition not in TASKS:
                parser.error(f"unknown competition: {args.competition}")
            self._run_competition()

        if args.asset_root is not None:
            parser.error("--asset-root requires --competition")
        if args.dataset is not None or args.contract is not None:
            if args.dataset is None or args.contract is None or args.paths:
                parser.error(
                    "--dataset and --contract must be used together, without node paths"
                )
            self._run_dataset_contract()
        self._run_nodes()


def main():
    """Compatibility command-line entry point."""
    PreflightApp().run()


if __name__ == "__main__":
    main()

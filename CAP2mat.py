import argparse
import datetime as dt
import math
import os
from fractions import Fraction

import numpy as np
from scipy.io import savemat
from scipy.signal import resample_poly


EPOCH_SEC = 30
UNKNOWN = -1

# Keep the same raw stage ids as SleepEDF so DataAdapter.preprocess()
# can merge N3/N4 and shift REM in one consistent place.
SLEEP_EVENT_TO_LABEL = {
    "SLEEP-S0": 0,
    "SLEEP-S1": 1,
    "SLEEP-S2": 2,
    "SLEEP-S3": 3,
    "SLEEP-S4": 4,
    "SLEEP-REM": 5,
    "SLEEP-MT": UNKNOWN,
    "SLEEP-UNSCORED": UNKNOWN,
}

CHANNEL_ALIASES = {
    "FP2-F4": ["FP2-F4", "F2-F4"],
    "F2-F4": ["F2-F4", "FP2-F4"],
    "FP1-F3": ["FP1-F3", "F1-F3"],
    "F1-F3": ["F1-F3", "FP1-F3"],
}


def _decode_ascii(raw_bytes):
    return raw_bytes.decode("ascii", errors="ignore").strip()


def _parse_edf_datetime(date_str, time_str):
    day, month, year = [int(part) for part in date_str.split(".")]
    year += 2000 if year < 85 else 1900
    hour, minute, second = [int(part) for part in time_str.split(".")]
    return dt.datetime(year, month, day, hour, minute, second)


def _normalize_channel_name(name):
    return name.upper().replace(" ", "").replace("EEG-", "").replace("EEG", "")


def _resolve_channel(labels, requested_channel):
    normalized_to_idx = {_normalize_channel_name(label): idx for idx, label in enumerate(labels)}
    requested = _normalize_channel_name(requested_channel)
    candidates = CHANNEL_ALIASES.get(requested, [requested])
    for candidate in candidates:
        if candidate in normalized_to_idx:
            idx = normalized_to_idx[candidate]
            return idx, labels[idx]
    raise ValueError(
        f"Channel '{requested_channel}' not found. Available channels: {labels}"
    )


def _read_edf_header(edf_path):
    with open(edf_path, "rb") as f:
        header = f.read(256)
        if len(header) != 256:
            raise ValueError(f"Incomplete EDF header: {edf_path}")

        header_bytes = int(_decode_ascii(header[184:192]))
        n_records = int(_decode_ascii(header[236:244]))
        record_length = float(_decode_ascii(header[244:252]))
        n_channels = int(_decode_ascii(header[252:256]))
        extra = f.read(header_bytes - 256)

    offset = 0

    def read_fields(size):
        nonlocal offset
        values = [
            _decode_ascii(extra[offset + i * size : offset + (i + 1) * size])
            for i in range(n_channels)
        ]
        offset += size * n_channels
        return values

    labels = read_fields(16)
    _ = read_fields(80)  # transducer type
    _ = read_fields(8)   # units
    physical_min = np.asarray([float(v) for v in read_fields(8)], dtype=np.float64)
    physical_max = np.asarray([float(v) for v in read_fields(8)], dtype=np.float64)
    digital_min = np.asarray([float(v) for v in read_fields(8)], dtype=np.float64)
    digital_max = np.asarray([float(v) for v in read_fields(8)], dtype=np.float64)
    _ = read_fields(80)  # prefiltering
    n_samples_per_record = np.asarray([int(v) for v in read_fields(8)], dtype=np.int64)

    return {
        "header_bytes": header_bytes,
        "n_records": n_records,
        "record_length": record_length,
        "n_channels": n_channels,
        "labels": labels,
        "physical_min": physical_min,
        "physical_max": physical_max,
        "digital_min": digital_min,
        "digital_max": digital_max,
        "n_samples_per_record": n_samples_per_record,
        "start_time": _parse_edf_datetime(
            _decode_ascii(header[168:176]),
            _decode_ascii(header[176:184]),
        ),
    }


def read_data(name, load_data_path, lead="Fp2-F4"):
    edf_path = os.path.join(load_data_path, name + ".edf")
    header = _read_edf_header(edf_path)
    channel_idx, actual_channel = _resolve_channel(header["labels"], lead)
    bytes_per_record = int(np.sum(header["n_samples_per_record"]) * 2)
    actual_records = (os.path.getsize(edf_path) - header["header_bytes"]) // bytes_per_record
    n_records = min(header["n_records"], actual_records)

    sample_rate = header["n_samples_per_record"][channel_idx] / header["record_length"]
    sample_rate = int(round(sample_rate))
    gain = (
        (header["physical_max"][channel_idx] - header["physical_min"][channel_idx])
        / (header["digital_max"][channel_idx] - header["digital_min"][channel_idx])
    )
    phys_min = header["physical_min"][channel_idx]
    dig_min = header["digital_min"][channel_idx]

    channel_samples = []
    with open(edf_path, "rb") as f:
        f.seek(header["header_bytes"])
        for _ in range(n_records):
            for current_idx, nsamp in enumerate(header["n_samples_per_record"]):
                raw = f.read(int(nsamp) * 2)
                if len(raw) != int(nsamp) * 2:
                    break
                if current_idx != channel_idx:
                    continue
                digital = np.frombuffer(raw, dtype="<i2").astype(np.float32)
                physical = (digital - dig_min) * gain + phys_min
                channel_samples.append(physical)

    if not channel_samples:
        raise ValueError(f"No samples extracted from {edf_path}")

    return np.concatenate(channel_samples), sample_rate, header["start_time"], actual_channel


def _parse_recording_date(lines):
    for line in lines:
        parts = [part.strip() for part in line.split("\t")]
        if len(parts) >= 2 and parts[0] == "Recording Date:":
            return dt.datetime.strptime(parts[-1], "%d/%m/%Y").date()
    raise ValueError("Recording Date not found in CAP annotation file")


def _parse_clock_time(time_str):
    normalized = time_str.strip().replace(".", ":")
    return dt.datetime.strptime(normalized, "%H:%M:%S").time()


def _load_subject_list(subject_list_path):
    with open(subject_list_path, "r", encoding="utf-8") as f:
        return {
            line.strip().replace(".edf", "").replace(".mat", "")
            for line in f
            if line.strip() and not line.strip().startswith("#")
        }


def read_label(name, load_label_path, edf_start_time):
    txt_path = os.path.join(load_label_path, name + ".txt")
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()

    recording_date = _parse_recording_date(lines)
    header = None
    header_idx = None
    for idx, line in enumerate(lines):
        if line.startswith("Sleep Stage"):
            header = [cell.strip().replace("Duration [s]", "Duration[s]") for cell in line.split("\t")]
            header_idx = idx
            break

    if header is None:
        raise ValueError(f"Sleep stage table not found in {txt_path}")

    prev_time = None
    day_offset = 0
    stage_events = []

    for raw_line in lines[header_idx + 1 :]:
        if not raw_line.strip():
            continue

        cols = [cell.strip() for cell in raw_line.split("\t")]
        if len(cols) < len(header):
            continue
        if len(cols) > len(header):
            cols = cols[: len(header)]

        record = dict(zip(header, cols))
        event_name = record.get("Event", "")
        if event_name not in SLEEP_EVENT_TO_LABEL:
            continue

        duration_sec = int(float(record.get("Duration[s]", "0")))
        if duration_sec <= 0:
            continue
        if duration_sec % EPOCH_SEC != 0:
            raise ValueError(f"Non-30-second epoch found in {txt_path}: {duration_sec}s")

        time_str = record.get("Time [hh:mm:ss]")
        if not time_str:
            raise ValueError(f"Time column missing in {txt_path}")

        current_time = _parse_clock_time(time_str)
        event_time = dt.datetime.combine(recording_date, current_time) + dt.timedelta(days=day_offset)
        if prev_time is not None and event_time < prev_time:
            day_offset += 1
            event_time += dt.timedelta(days=1)
        prev_time = event_time

        onset_sec = int(round((event_time - edf_start_time).total_seconds()))
        stage_events.append((onset_sec, duration_sec, SLEEP_EVENT_TO_LABEL[event_name]))

    if not stage_events:
        raise ValueError(f"No sleep stage events found in {txt_path}")

    return stage_events


def _build_epoch_sequence(stage_events):
    ordered_events = sorted(stage_events, key=lambda item: item[0])
    start_sec = max(0, ordered_events[0][0])
    current_sec = start_sec
    labels = []

    for onset_sec, duration_sec, label in ordered_events:
        if onset_sec + duration_sec <= start_sec:
            continue

        onset_sec = max(onset_sec, start_sec)

        if onset_sec > current_sec:
            gap_sec = onset_sec - current_sec
            if gap_sec % EPOCH_SEC != 0:
                raise ValueError(f"Annotation gap is not aligned to {EPOCH_SEC}s epochs: {gap_sec}s")
            labels.extend([UNKNOWN] * (gap_sec // EPOCH_SEC))
            current_sec = onset_sec
        elif onset_sec < current_sec:
            overlap_sec = current_sec - onset_sec
            if overlap_sec >= duration_sec:
                continue
            if overlap_sec % EPOCH_SEC != 0:
                raise ValueError(f"Annotation overlap is not aligned to {EPOCH_SEC}s epochs: {overlap_sec}s")
            duration_sec -= overlap_sec

        labels.extend([label] * (duration_sec // EPOCH_SEC))
        current_sec += duration_sec

    if not labels:
        raise ValueError("No usable sleep epochs found after annotation alignment")

    return start_sec, np.asarray(labels, dtype=np.int64)


def _resample_signal(signal, src_fs, target_fs):
    if src_fs == target_fs:
        return signal.astype(np.float64, copy=False)

    ratio = Fraction(target_fs, src_fs).limit_denominator()
    resampled = resample_poly(signal, ratio.numerator, ratio.denominator)
    return resampled.astype(np.float64, copy=False)


def gen_data(
    load_data_path,
    load_label_path,
    out_path,
    sel_ch,
    target_fs=100,
    resume=False,
    limit=None,
    subject_list_path=None,
):
    os.makedirs(out_path, exist_ok=True)
    edf_files = sorted(file for file in os.listdir(load_data_path) if file.endswith(".edf"))
    if subject_list_path is not None:
        allowed_subjects = _load_subject_list(subject_list_path)
        edf_files = [file for file in edf_files if file[:-4] in allowed_subjects]
    if limit is not None:
        edf_files = edf_files[:limit]

    converted = 0
    failed = 0
    for file_name in edf_files:
        subject = file_name[:-4]
        out_file = os.path.join(out_path, subject + ".mat")
        if resume and os.path.exists(out_file):
            print("skip existing", subject)
            continue

        try:
            signal, src_fs, edf_start_time, actual_channel = read_data(subject, load_data_path, sel_ch)
            stage_events = read_label(subject, load_label_path, edf_start_time)
            start_sec, labels = _build_epoch_sequence(stage_events)
        except Exception as exc:
            failed += 1
            print(subject, "read failed!")
            print(exc)
            continue

        epoch_samples = EPOCH_SEC * src_fs
        start_sample = start_sec * src_fs
        expected_samples = len(labels) * epoch_samples
        if start_sample >= len(signal):
            failed += 1
            print(subject, "start_sample beyond signal length")
            continue

        max_epochs = (len(signal) - start_sample) // epoch_samples
        if max_epochs <= 0:
            failed += 1
            print(subject, "no complete epochs available after alignment")
            continue
        if max_epochs < len(labels):
            labels = labels[:max_epochs]
            expected_samples = len(labels) * epoch_samples

        cropped = signal[start_sample : start_sample + expected_samples]
        resampled = _resample_signal(cropped, src_fs, target_fs)
        expected_resampled = len(labels) * EPOCH_SEC * target_fs
        if len(resampled) != expected_resampled:
            resampled = resampled[:expected_resampled]
        if len(resampled) != expected_resampled:
            failed += 1
            print(subject, "resampled data != label")
            continue

        save_dict = {
            "fpz_cz": resampled,
            "data": resampled,
            "label": labels.astype(np.int64),
            "fs": np.asarray(target_fs, dtype=np.int64),
            "selected_channel": np.asarray(actual_channel, dtype=object),
        }
        savemat(out_file, save_dict)
        converted += 1
        print("save", subject, "success!", f"channel={actual_channel}", f"epochs={len(labels)}")

    print(f"finished: converted={converted}, failed={failed}, output={out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data_path",
        default="../data/physionet.org/files/capslpdb/1.0.0",
        help="Path to the CAP EDF and annotation txt files.",
    )
    parser.add_argument(
        "-l",
        "--label_path",
        default=None,
        help="Optional label path. Defaults to data_path because CAP txt labels live beside the EDFs.",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        default="data/cap-sleep-database-mat",
        help="Directory where converted MAT files will be stored.",
    )
    parser.add_argument(
        "-s",
        "--select_channel",
        default="Fp1-F3",
        help="Preferred CAP EEG channel. Fp1-F3/F1-F3 and Fp2-F4/F2-F4 aliases are supported.",
    )
    parser.add_argument(
        "--target_fs",
        type=int,
        default=100,
        help="Target sampling rate after resampling. Keep 100 Hz to match MRASleepNet's current input windowing.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip subjects that already have converted MAT files in output_path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for smoke tests or partial conversion.",
    )
    parser.add_argument(
        "--subject_list",
        default=None,
        help="Optional text file listing the CAP subject ids to convert, one per line.",
    )
    args = parser.parse_args()

    label_path = args.label_path or args.data_path
    gen_data(
        args.data_path,
        label_path,
        args.output_path,
        args.select_channel,
        target_fs=args.target_fs,
        resume=args.resume,
        limit=args.limit,
        subject_list_path=args.subject_list,
    )

from __future__ import annotations

from collections import Counter

import numpy as np

from sleep_stage.config import INDEX_TO_LABEL, LABELS, LABEL_TO_INDEX


def mode_filter_labels(labels: list[str], window: int = 3) -> list[str]:
    if window <= 1 or len(labels) <= 2:
        return list(labels)
    radius = window // 2
    filtered: list[str] = []
    for index, label in enumerate(labels):
        start = max(0, index - radius)
        end = min(len(labels), index + radius + 1)
        counts = Counter(labels[start:end])
        filtered.append(counts.most_common(1)[0][0] if counts else label)
    return filtered


def build_transition_log_probs(train_sequences: list[list[str]], smoothing: float = 1.0) -> np.ndarray:
    counts = np.full((len(LABELS), len(LABELS)), smoothing, dtype=float)
    for sequence in train_sequences:
        for previous, current in zip(sequence, sequence[1:]):
            counts[LABEL_TO_INDEX[previous], LABEL_TO_INDEX[current]] += 1.0
    probs = counts / counts.sum(axis=1, keepdims=True)
    return np.log(probs)


def viterbi_decode(probabilities: np.ndarray, transition_log_probs: np.ndarray) -> list[str]:
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(LABELS):
        raise ValueError(f"probabilities must have shape (n_epochs, {len(LABELS)})")
    if transition_log_probs.shape != (len(LABELS), len(LABELS)):
        raise ValueError(f"transition_log_probs must have shape {(len(LABELS), len(LABELS))}")

    emission = np.log(np.clip(probabilities, 1e-12, 1.0))
    n_epochs = probabilities.shape[0]
    scores = np.zeros((n_epochs, len(LABELS)), dtype=float)
    backpointers = np.zeros((n_epochs, len(LABELS)), dtype=int)
    scores[0] = emission[0]
    for time_index in range(1, n_epochs):
        candidate = scores[time_index - 1][:, None] + transition_log_probs
        backpointers[time_index] = candidate.argmax(axis=0)
        scores[time_index] = candidate.max(axis=0) + emission[time_index]

    states = [int(scores[-1].argmax())]
    for time_index in range(n_epochs - 1, 0, -1):
        states.append(int(backpointers[time_index, states[-1]]))
    states.reverse()
    return [INDEX_TO_LABEL[state] for state in states]

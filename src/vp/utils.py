import glob
import json
import os
import pickle
import random
import types

from ._words import adjectives, nouns, verbs

CHECKPOINT_SUFFIX = "-model.pkl"


class MetricMeter:
    """
    Computes and stores simple statistics of some metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.min = float("inf")
        self.max = -self.min
        self.last_max = 0
        self.last_min = 0
        self.current = None
        self.mean = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        if val > self.max:
            self.max = val
            self.last_max = 0
        else:
            self.last_max += 1
        if val < self.min:
            self.min = val
            self.last_min = 0
        else:
            self.last_min += 1
        self.current = val
        self.sum += val
        self.count += 1
        self.mean = self.sum / self.count


class MeterCollection:
    """
    A collection of `MetricMeter`s, each with given name.

    An individual `MetericMeter` can be accessed as a member attribute
    by using its name as defined in the constructor.
    """

    def __init__(self, *names):
        for name in names:
            if name.startswith("_") or name in ("meters", "update", "reset"):
                raise ValueError(f"Invalid name `{name}`")
        self.meters = {name: MetricMeter() for name in names}

    def update(self, **kwargs):
        for name, value in kwargs.items():
            self.meters[name].update(value)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def __getattr__(self, name):
        if name in self.meters:
            return self.meters[name]
        else:
            return getattr(super(), name)

    def summary_dict(self, detailed=False):
        if detailed:
            return {
                name: (meter.min, meter.mean, meter.max)
                for name, meter in self.meters.items()
            }
        return {name: meter.mean for name, meter in self.meters.items()}

    def describe(self):
        s = [
            "{name}={value:.4f} (min={min:.4f}, max={max:.4f})".format(
                name=name, value=meter.mean, min=meter.min, max=meter.max
            )
            for name, meter in self.meters.items()
        ]
        s = " ".join(s)
        return s

    def __repr__(self):
        s = [
            "{name}={value:.10f}".format(name=name, value=meter.mean)
            for name, meter in self.meters.items()
        ]
        s = " ".join(s)
        return s


class RunLog:
    def __init__(self, path, config, exist_ok=False):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=exist_ok)
        self.path = path
        self.config = config
        config_path = os.path.join(self.path, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config, f)
        self._log = dict()

    def log(self, d, name):
        if name not in self._log:
            self._log[name] = {}
        for key, value in d.items():
            if key not in self._log[name]:
                self._log[name][key] = [value]
            else:
                self._log[name][key].append(value)
        path = os.path.join(self.path, "log.json")
        with open(path, "w") as f:
            json.dump(self._log, f)

    def checkpoint(self, ckpt, name):
        path = os.path.join(self.path, f"{name}{CHECKPOINT_SUFFIX}")
        with open(path, "wb") as f:
            pickle.dump(ckpt, f)


def load_log(log_path, with_checkpoints=True):
    path = os.path.join(log_path, "config.json")
    with open(path, "r") as f:
        config = json.load(f)
        config = types.SimpleNamespace(**config)
    path = os.path.join(log_path, "log.json")
    with open(path, "r") as f:
        log = json.load(f)
    info = {
        "config": config,
        "log": log,
    }
    if with_checkpoints:
        checkpoints = {}
        for path in glob.glob(os.path.join(log_path, f"*{CHECKPOINT_SUFFIX}")):
            with open(path, "rb") as f:
                checkpoint = pickle.load(f)
            name = os.path.basename(path)[: -len(CHECKPOINT_SUFFIX)]
            checkpoints[name] = checkpoint
        info["checkpoints"] = checkpoints
    return info


def make_random_name():
    verb = random.choice(verbs)
    adjective = random.choice(adjectives)
    noun = random.choice(nouns)
    return f"{verb}-{adjective}-{noun}"

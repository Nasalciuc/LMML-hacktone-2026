import math

SENSORS = ["temperature_c", "humidity_pct", "vibration_g", "acoustic_db", "air_pressure_kpa", "co2_ppm"]

# normal rows of the training file; used until the stream gives its own numbers
PRIOR_MEAN = {"temperature_c": 13.0, "humidity_pct": 72.0, "vibration_g": 0.12,
              "acoustic_db": 32.1, "air_pressure_kpa": 101.5, "co2_ppm": 510.3}
PRIOR_STD = {"temperature_c": 0.37, "humidity_pct": 0.58, "vibration_g": 0.01,
             "acoustic_db": 0.35, "air_pressure_kpa": 0.1, "co2_ppm": 10.5}

WARMUP = 150         # first readings used to re-center on this stream
BASE_ALPHA = 0.005   # how fast the normal level follows the data
SMOOTH_ALPHA = 0.3   # short smoothing for drifts / collective
SPIKE_CAP = 5.0


def _median(v):
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


class Detector:
    def __init__(self):
        self.mean = dict(PRIOR_MEAN)
        self.var = {k: v * v for k, v in PRIOR_STD.items()}
        self.seen = {k: [] for k in SENSORS}
        self.t = 0
        self.smooth = {k: 0.0 for k in SENSORS}
        self.energy = 0.0

    def _read(self, reading, k):
        try:
            x = float(reading[k])
        except (KeyError, TypeError, ValueError):
            return self.mean[k]
        return x if math.isfinite(x) else self.mean[k]

    def _warmup(self, x):
        for k in SENSORS:
            self.seen[k].append(x[k])
            if len(self.seen[k]) >= 5:
                self.mean[k] = _median(self.seen[k])
        if self.t == WARMUP:
            for k in SENSORS:
                mad = _median([abs(v - self.mean[k]) for v in self.seen[k]]) * 1.4826
                # keep the prior spread unless this stream is clearly noisier
                self.var[k] = (mad if mad > 1.5 * PRIOR_STD[k] else PRIOR_STD[k]) ** 2
            self.seen = None

    def update(self, reading: dict) -> float:
        self.t += 1
        x = {k: self._read(reading, k) for k in SENSORS}
        if self.seen is not None:
            self._warmup(x)

        z = {k: (x[k] - self.mean[k]) / math.sqrt(self.var[k]) for k in SENSORS}
        spike = max(abs(v) for v in z.values()) / 6.0

        # one huge spike should not leak into the smoothed signals
        zc = {k: max(-SPIKE_CAP, min(SPIKE_CAP, v)) for k, v in z.items()}
        for k in SENSORS:
            self.smooth[k] += SMOOTH_ALPHA * (zc[k] - self.smooth[k])
        e = sum(v * v for v in zc.values()) / len(SENSORS)
        self.energy += SMOOTH_ALPHA * (e - self.energy)

        # the smoothed level has to be backed by the current reading too,
        # otherwise the score hangs on for a few steps after an event ends
        drift = 0.0
        for k in SENSORS:
            if self.smooth[k] * zc[k] > 0:
                drift = max(drift, min(abs(self.smooth[k]) / 3.0, abs(zc[k]) / 2.0))
        group = min(self.energy / 4.0, e / 3.0)
        m = max(spike, drift, group)

        if self.seen is None and m < 0.7:
            for k in SENSORS:
                d = x[k] - self.mean[k]
                self.mean[k] += BASE_ALPHA * d
                self.var[k] += BASE_ALPHA * (d * d - self.var[k])
                self.var[k] = max(self.var[k], (0.25 * PRIOR_STD[k]) ** 2)  # stuck sensor guard

        return 1.0 / (1.0 + math.exp(-6.0 * (m - 1.0)))

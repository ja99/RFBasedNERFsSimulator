"""Compatibility for RF manifests spanning the opt-in camera CLI addition."""

# The benchmark hashes entire source files. These exact revisions share the
# same RF scene, waveform and label generation. Only CLI/manifest code changed.
# A different hash in any file, or any acquisition parameter change, still
# requires a new cohort. Keep the original manifest when resuming an old run.
_DEPTH_SIDECAR_REVISIONS = {
    "Acquisition.py": (
        "9f7bb1c3137c5abbc00093bea69a41657e12d5551576d46439e1f2af180aa231",
        "174276d625c8274b0e6a5c6a1c39206f9fef1f97805e7eacea038873b002e435",
    ),
    "PhantomBenchmark.py": (
        "fe74db50f1585a87623f63c0b22c43cedc0d3a822093583d56807e292daaee5a",
        "4bd2d6589eea2a82fe426c6de276249251d4b694593230171dbe6ff88fca9ea6",
    ),
    "MeterBenchmark.py": (
        "3934194dc2bf54ff415a0268e9697125405f7161d91e8ad6f390ab068ac65326",
        "ff2ef6c1058e702796798ad7b2036a0ac6cb844868a300fe3c386b3c425b3a0d",
    ),
}


def same_acquisition_manifest(stored, current, source_key):
    sources = dict(current[source_key])
    for name, (previous, updated) in _DEPTH_SIDECAR_REVISIONS.items():
        if sources.get(name) == updated and stored.get(source_key, {}).get(name) == previous:
            sources[name] = previous
    return stored == {**current, source_key: sources}

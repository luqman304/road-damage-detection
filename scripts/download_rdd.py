"""Download RDD2022 country archives (all except Norway) via HTTP range requests."""
import sys, zipfile, pathlib
from remotezip import RemoteZip

URL = "https://ndownloader.figshare.com/files/38030910"
OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw")
COUNTRIES = ["China_Drone", "China_MotorBike", "Czech", "India", "Japan", "United_States"]

OUT.mkdir(parents=True, exist_ok=True)
with RemoteZip(URL) as z:
    for c in COUNTRIES:
        inner = OUT / f"{c}.zip"
        if not (OUT / c).exists():
            print("downloading", c, flush=True)
            z.extract(f"RDD2022/{c}.zip", OUT / "_tmp")
            (OUT / "_tmp" / "RDD2022" / f"{c}.zip").rename(inner)
            with zipfile.ZipFile(inner) as iz:
                iz.extractall(OUT)
            inner.unlink()
        print("done", c, flush=True)

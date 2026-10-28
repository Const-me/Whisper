"""Windows-only native Whisper regression; generated silence, no personal audio.

python reproduce.py --exe PATH/main.exe --model PATH/ggml-small.bin --work-dir PATH/results
Optional: --gpu "AMD Radeon RX 5700 XT" --threads 4 --iterations 10
A nonzero result means at least one crash, timeout, or missing output.
"""
import argparse
import ctypes
import json
from pathlib import Path
import subprocess
import wave


def wav(path, seconds):
    with wave.open(str(path), "wb") as handle:
        handle.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        handle.writeframes(b"\0\0" * int(16000 * seconds))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--gpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    # Children inherit this mode: crashes return a code instead of awaiting a popup.
    ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)
    root = args.work_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    short = [root / f"short-{i}.wav" for i in range(50)]
    for path in short:
        wav(path, 0.5)
    normal = root / "normal.wav"
    wav(normal, 3)
    long = root / "long.wav"
    wav(long, 70)
    cases = [("short", short, []), ("mixed", short + [normal], []),
             ("early-return", [long], ["-d", "1100"])]
    failures = 0
    for name, files, extra in cases:
        for iteration in range(args.iterations):
            for path in files:
                path.with_suffix(".txt").unlink(missing_ok=True)
            cmd = [str(args.exe.resolve()), "-m", str(args.model.resolve()), "-l", "en",
                   "-nt", "-otxt", "-t", str(args.threads)]
            if args.gpu:
                cmd += ["-gpu", args.gpu]
            cmd += extra + [str(path) for path in files]
            with (root / f"{name}-{iteration}.log").open("w", encoding="utf-8") as log:
                try:
                    result = subprocess.run(cmd, cwd=args.exe.resolve().parent,
                                            stdout=log, stderr=subprocess.STDOUT, timeout=120)
                    status = f"0x{result.returncode & 0xffffffff:08X}"
                    missing = sum(not path.with_suffix(".txt").is_file() for path in files)
                    failed = result.returncode != 0 or missing != 0
                except subprocess.TimeoutExpired:
                    status, missing, failed = "timeout", len(files), True
            failures += int(failed)
            print(json.dumps({"case": name, "iteration": iteration, "exit": status,
                              "files": len(files), "missing": missing}), flush=True)
    print(json.dumps({"runs": len(cases) * args.iterations, "failures": failures}), flush=True)
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())

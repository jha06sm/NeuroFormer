import argparse
import os
import re
import shutil


def parse_subjects(spec: str):
    """
    Parse subject spec like:
      "0-19" or "0,1,2,5-7"
    Returns a set of ints.
    """
    spec = spec.replace(" ", "")
    parts = [p for p in spec.split(",") if p]
    subjects = set()
    for p in parts:
        if "-" in p:
            a, b = p.split("-", 1)
            if a == "" or b == "":
                raise ValueError(f"Bad range: {p}")
            start = int(a)
            end = int(b)
            if end < start:
                raise ValueError(f"Bad range: {p}")
            subjects.update(range(start, end + 1))
        else:
            subjects.add(int(p))
    return subjects


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat_dir", default="data/sleep-edf-cassette-mat",
                        help="Directory containing SC .mat files.")
    parser.add_argument("--out_dir", default="data/sleep-edf-20-mat",
                        help="Output directory for subset (symlinks by default).")
    parser.add_argument("--subjects", default="0-19",
                        help="Subject IDs to include, e.g. '0-19' or '0,1,2,5-7'.")
    parser.add_argument("--copy", action="store_true",
                        help="Copy files instead of creating symlinks.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip files that already exist in out_dir.")
    args = parser.parse_args()

    subjects = parse_subjects(args.subjects)
    os.makedirs(args.out_dir, exist_ok=True)

    # SC file pattern: SC4<subject><night>... e.g., SC4001E0.mat
    pat = re.compile(r"^SC4(\d{2})(\d).*\.mat$")

    selected = 0
    for fname in sorted(os.listdir(args.mat_dir)):
        if not fname.endswith(".mat"):
            continue
        m = pat.match(fname)
        if not m:
            continue
        subj = int(m.group(1))
        if subj not in subjects:
            continue

        src = os.path.join(args.mat_dir, fname)
        dst = os.path.join(args.out_dir, fname)
        if args.resume and os.path.exists(dst):
            continue

        if args.copy:
            shutil.copy2(src, dst)
        else:
            # Replace existing broken link if needed
            if os.path.islink(dst) or os.path.exists(dst):
                if not args.resume:
                    os.remove(dst)
            os.symlink(os.path.abspath(src), dst)

        selected += 1

    print(f"Subset created: {selected} files in {args.out_dir}")


if __name__ == "__main__":
    main()

import argparse
import ast
import os
import numpy as np


def newest_log_file(folder):
    if not os.path.isdir(folder):
        return None
    files = [f for f in os.listdir(folder) if not f.startswith(".")]
    if not files:
        return None
    files = [os.path.join(folder, f) for f in files]
    return max(files, key=os.path.getmtime)


def parse_log(path):
    acc = None
    kappa = None
    mf1 = None
    conf = None
    next_is_conf = False

    with open(path, "r", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("Test Accuracy:"):
                acc = float(s.split(":", 1)[1])
            elif s.startswith("Test Kappa:"):
                kappa = float(s.split(":", 1)[1])
            elif s.startswith("Test MF1:"):
                mf1 = float(s.split(":", 1)[1])
            elif s.startswith("Test confusion matrix"):
                next_is_conf = True
            elif next_is_conf:
                try:
                    conf = np.array(ast.literal_eval(s), dtype=float)
                except Exception:
                    conf = None
                next_is_conf = False

    return acc, kappa, mf1, conf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_root", default="logs", help="Root folder containing fold_* subdirs.")
    parser.add_argument("--folds", type=int, default=78, help="Number of folds.")
    args = parser.parse_args()

    accs, kappas, mf1s = [], [], []
    conf_sum = None
    found = 0

    for i in range(args.folds):
        fold_dir = os.path.join(args.log_root, f"fold_{i}")
        log_file = newest_log_file(fold_dir)
        if not log_file:
            continue
        acc, kappa, mf1, conf = parse_log(log_file)
        if acc is None or kappa is None or mf1 is None:
            continue
        accs.append(acc)
        kappas.append(kappa)
        mf1s.append(mf1)
        if conf is not None:
            conf_sum = conf if conf_sum is None else conf_sum + conf
        found += 1

    if found == 0:
        print("No fold results found.")
        return

    print(f"Folds found: {found}/{args.folds}")
    print(f"Acc mean/std: {np.mean(accs):.4f} / {np.std(accs):.4f}")
    print(f"Kappa mean/std: {np.mean(kappas):.4f} / {np.std(kappas):.4f}")
    print(f"MF1 mean/std: {np.mean(mf1s):.4f} / {np.std(mf1s):.4f}")

    if conf_sum is not None:
        print("Confusion matrix sum:")
        print(conf_sum.astype(int))

        row_sum = conf_sum.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            row_norm = np.divide(conf_sum, row_sum, out=np.zeros_like(conf_sum), where=row_sum != 0)
        print("Confusion matrix row-normalized:")
        print(np.round(row_norm, 4))


if __name__ == "__main__":
    main()

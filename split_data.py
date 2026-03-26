import argparse
import os
import random
import re


CAP_SPLIT_RATIOS = (0.7, 0.1, 0.2)


def list_mat_files(path):
    return sorted(
        files for files in os.listdir(path)
        if files.endswith('.mat') and os.path.isfile(os.path.join(path, files))
    )


def load_subject_list(subject_list_path):
    with open(subject_list_path, 'r', encoding='utf-8') as f:
        return {
            line.strip().replace('.edf', '').replace('.mat', '')
            for line in f
            if line.strip() and not line.strip().startswith('#')
        }

def combine_sample(path,types = 'ALL'):
    sample = []
    dirs = list_mat_files(path)
    for files in dirs:
        if types == 'ALL':
            if files[5] == '1':
                tmp_name = files[:5] + '2' + files[6:]
                if tmp_name in dirs:
                    sample.append([files,tmp_name])
                else:
                    sample.append([files])
        elif types == 'SC':
            if files[5] == '1' and files[:2] == 'SC':
                tmp_name = files[:5] + '2' + files[6:]
                if tmp_name in dirs:
                    sample.append([files,tmp_name])
                else:
                    sample.append([files])
        elif types == 'ST':
            if files[5] == '1' and files[:2] == 'ST':
                tmp_name = files[:5] + '2' + files[6:]
                if tmp_name in dirs:
                    sample.append([files,tmp_name])
                else:
                    sample.append([files])
        elif types == 'CAP':
            sample.append([files])
    return sample


def write_split_file(data_path, output_path, file_name, samples):
    with open(os.path.join(output_path, file_name), 'w') as f:
        for sample in sorted(samples):
            f.write(os.path.join(data_path, sample) + '\n')


def cap_category(file_name):
    stem = os.path.splitext(file_name)[0].lower()
    match = re.match(r'[a-z]+', stem)
    if match is None:
        return 'unknown'
    return match.group(0)


def stratified_cap_split(data_path, output_path, seed, subject_list_path=None):
    os.makedirs(output_path, exist_ok=True)
    allowed_subjects = None
    if subject_list_path is not None:
        allowed_subjects = load_subject_list(subject_list_path)
    grouped = {}
    for sample in list_mat_files(data_path):
        if allowed_subjects is not None and os.path.splitext(sample)[0] not in allowed_subjects:
            continue
        grouped.setdefault(cap_category(sample), []).append(sample)

    train_samples = []
    val_samples = []
    test_samples = []

    rng = random.Random(seed)

    for category, samples in sorted(grouped.items()):
        samples = samples[:]
        rng.shuffle(samples)
        total = len(samples)
        train_count = int(total * CAP_SPLIT_RATIOS[0])
        val_count = int(total * CAP_SPLIT_RATIOS[1])
        test_count = total - train_count - val_count

        if total >= 3 and val_count == 0:
            val_count = 1
            if train_count > test_count and train_count > 1:
                train_count -= 1
            else:
                test_count -= 1
        if total >= 2 and test_count == 0:
            test_count = 1
            if train_count > 1:
                train_count -= 1
            elif val_count > 1:
                val_count -= 1

        train_end = train_count
        val_end = train_end + val_count
        train_samples.extend(samples[:train_end])
        val_samples.extend(samples[train_end:val_end])
        test_samples.extend(samples[val_end:])

    write_split_file(data_path, output_path, 'train.txt', train_samples)
    write_split_file(data_path, output_path, 'val.txt', val_samples)
    write_split_file(data_path, output_path, 'test.txt', test_samples)

    print('CAP split summary:')
    print('train:', len(train_samples))
    print('val:', len(val_samples))
    print('test:', len(test_samples))
    for category, samples in sorted(grouped.items()):
        category_train = sum(cap_category(sample) == category for sample in train_samples)
        category_val = sum(cap_category(sample) == category for sample in val_samples)
        category_test = sum(cap_category(sample) == category for sample in test_samples)
        print(category, category_train, category_val, category_test)

def split_dataset(data_path,output_path,sample,folds,fold_count = 0):
    os.makedirs(output_path, exist_ok=True)
    step = max(1,int(len(sample)/folds))
    test_sample = sample[fold_count*step:(fold_count+1)*step]
    all_sample = []
    for samp in sample:
        all_sample.extend(samp)
    res_test_sample = []
    for samp in test_sample:
        res_test_sample.extend(samp)
    res_train_sample = list(set(all_sample) - set(res_test_sample))

    write_split_file(data_path, output_path, 'train.txt', res_train_sample)
    write_split_file(data_path, output_path, 'test.txt', res_test_sample)

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--seed', type=int, default=10, help='random seed')
parser.add_argument('-f', '--file_type', choices=['SC','ST','ALL','CAP'], default='SC', help = 'Select the dataset split mode')
parser.add_argument('-d', '--data_path', help = 'The path to store data')
parser.add_argument('-o', '--output_path', help = 'The path to output train data file and test data file')
parser.add_argument('-fd', '--folds', type=int, default=10, help = 'K folds cross validation')
parser.add_argument('-fi', '--fold_idx', type=int, default=0, help = 'The fold_idx fold in the K-fold cross validation')
parser.add_argument('--subject_list', default=None, help='Optional text file listing CAP subject ids to include, one per line')
args = parser.parse_args()

file_type = args.file_type
data_path = args.data_path
seed = args.seed
folds = args.folds
fold_idx = args.fold_idx
output_path = args.output_path

if file_type == 'CAP':
    stratified_cap_split(data_path, output_path, seed, subject_list_path=args.subject_list)
else:
    sample = combine_sample(data_path,types=file_type)
    random.seed(seed)
    random.shuffle(sample)
    split_dataset(data_path,output_path,sample,folds,fold_idx)

print('split data finished!')

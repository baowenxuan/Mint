import torch
from torch.utils.data import DataLoader
import clip
from tqdm import tqdm
import argparse
import os
import numpy as np
import random
import yaml
import csv

from datasets import get_dataset_class
from datasets.utils import TaggedMultipleDataset
from methods import get_method_class


def single_domain_test(datasets, tta_model, args):
    """
    Adaptation on each domain is independent, since the tta model is reset after each domain.
    """
    dataset_accs = []

    for i, dataset in enumerate(datasets):

        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                pin_memory=True)

        num_correct, num_total = 0, 0
        for image, label in tqdm(dataloader):
            image, label = image.to(args.device), label.to(args.device)
            logits = tta_model(image)

            with torch.no_grad():
                pred = torch.argmax(logits, dim=1)
                num_correct += pred.eq(label).sum().item()
                num_total += label.size(0)

        dataset_accs.append(num_correct / num_total)
        tqdm.write(f"{datasets.environments[i]}: {dataset_accs[-1]}")

        tta_model.reset()

    domain_names = datasets.environments

    return domain_names, dataset_accs


def mixed_domain_test(datasets, tta_model, args):
    mix_dataset = TaggedMultipleDataset(datasets)

    dataset_num_corrects = torch.zeros(len(datasets), dtype=torch.int)
    dataset_num_samples = np.array([len(dataset) for dataset in datasets])

    dataloader = DataLoader(mix_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                            pin_memory=True)

    for image, label, domain_idx, sample_idx in tqdm(dataloader):
        image, label = image.to(args.device), label.to(args.device)

        logits = tta_model(image)

        with torch.no_grad():
            pred = torch.argmax(logits, dim=1)
            is_correct = pred.eq(label).cpu().int()
            dataset_num_corrects.index_add_(0, domain_idx, is_correct)

    dataset_num_corrects = dataset_num_corrects.numpy()
    dataset_accs = dataset_num_corrects / dataset_num_samples

    domain_names = datasets.environments

    tta_model.reset()

    return domain_names, dataset_accs


def main(args):
    print('-' * 80)
    print(args)
    print('-' * 80)
    print(args.config)
    print('-' * 80)

    print("Loading model...")
    clip_model, preprocess = clip.load(args.model, device=args.device)

    print("Loading datasets...")
    datasets = get_dataset_class(args.dataset)(root=args.data_root, transform=preprocess)
    print(f"dataset includes environments: \n{datasets.environments}")

    print("Initializing model")
    tta_method = get_method_class(args.tta_algo)(clip_model, datasets.classes, args.config)

    if args.tta_mode == 'single':
        domain_names, dataset_accs = single_domain_test(datasets, tta_method, args)

    elif args.tta_mode == 'mixed':
        domain_names, dataset_accs = mixed_domain_test(datasets, tta_method, args)

    else:
        raise ValueError(f'Unknown tta mode: {args.tta_mode}')

    # Print results
    print('-' * 80)
    for env, acc in zip(domain_names, dataset_accs):
        print(f"{env}: {acc * 100:.2f}%")

    print('-' * 80)
    print(f"total: {np.mean(dataset_accs) * 100:.2f}%")

    # SAVE csv here if needed
    csv_path = args.save_to
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, "a", newline='') as f:
        writer = csv.writer(f)

        args_dict = {k: str(v) for k, v in vars(args).items() if isinstance(v, (int, str, float, bool))}
        writer.writerow([f"{k}={v}" for k, v in args_dict.items()])

        writer.writerow([f"{k}={v}" for k, v in args.config.items()])

        writer.writerow(datasets.environments + ["Average"])

        avg_acc = np.mean(dataset_accs)
        accs_formatted = [f"{acc * 100:.2f}" for acc in dataset_accs]
        writer.writerow(accs_formatted + [f"{avg_acc * 100:.2f}"])

        writer.writerow([])


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='CIFAR10C')

    parser.add_argument('--data_root', type=str, default='~/data')

    parser.add_argument('--tta_mode', type=str, default='single')

    parser.add_argument('--model', type=str, default='ViT-B/32',
                        help='model name')

    parser.add_argument('--use_tbn', action='store_true', default=False,
                        help='whether to use target batch norm statistics')

    parser.add_argument('--tta_algo', type=str, default='CLIP',
                        help='tta algorithm name')

    parser.add_argument('--batch_size', type=int, default=20,
                        help='number of images in each mini-batch')

    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers for dataloader')

    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--cuda', action='store_true', default=False,
                        help='whether use cuda to train')

    parser.add_argument('--num_threads', type=int, default=4,
                        help='Number of threads')

    parser.add_argument('--config', type=str, default='../cfg')

    parser.add_argument('--save_to', type=str, default='../log/default.csv')

    args = parser.parse_args()

    args.data_root = os.path.expanduser(args.data_root)

    args.device = torch.device('cuda') if torch.cuda.is_available() and args.cuda else torch.device('cpu')

    filepaths = [
        os.path.expanduser(os.path.join(args.config, args.dataset, args.tta_algo + '.yaml')),
        os.path.expanduser(os.path.join(args.config, 'default', args.tta_algo + '.yaml')),
    ]

    found_yaml = False

    for filepath in filepaths:
        if os.path.exists(filepath):
            print('Loading config from {}'.format(filepath))
            with open(filepath, 'r') as f:
                args.config = yaml.safe_load(f)

            found_yaml = True
            break

    if not found_yaml:
        args.config = {}

    args.max_batch_size = args.batch_size

    return args


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    args = args_parser()
    torch.set_num_threads(args.num_threads)
    setup_seed(args.seed)
    main(args)

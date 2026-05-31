import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.utils import get_model
from data.utils import get_dataset
from demix.utils import set_random_seed
from demix.pipelines import DeMixPipeline


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--data_name', type=str, default='adult',
                        choices=['adult', 'bank', 'credit', 'covertype', 'bike_sharing', 'air_quality'])
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--model_name', type=str, default='mlp1')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--alpha', type=float, default=0.5)
    args = parser.parse_args()
    set_random_seed(args.seed)
    print(f"----- data name: {args.data_name}, noise ratio: {1 - args.alpha} -----")
    dataset = get_dataset(args.save_dir, args.data_name, args.seed)
    d_train = dataset.controlled_error_injection(mode='train', clean_ratio=args.alpha)
    dataset.load_erroneous_data(d_train, mode='train')
    d_valid = dataset.controlled_error_injection(mode='valid', clean_ratio=args.alpha)
    dataset.load_erroneous_data(d_valid, mode='valid')
    model_path = f'{args.save_dir}/{args.data_name}/error_{args.model_name}.pth'
    model = get_model(args.model_name, model_path, args.seed, args.device)
    model.fit(dataset)
    print('model performance before repair:')
    model.evaluate(dataset)
    dec_path = f'{args.save_dir}/dec_ckpts/dec_{args.data_name}.pth'
    # use a more unified dec across all datasets:
    # dec_path = f'{args.save_dir}/dec_ckpts/dec_unif.pth'
    pipeline = DeMixPipeline(args.model_name, model, dataset, args.seed, args.device, dec_path)
    pipeline.run()

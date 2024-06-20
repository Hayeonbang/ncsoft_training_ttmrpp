from torch.utils.data import DataLoader
from ttmrpp_nc.dataset_wavs.annotation import Annotation_Dataset


def get_dataloader(args, split):
    dataset = get_dataset(
        eval_dataset= args.eval_dataset,
        data_path= args.msu_dir,
        split= split,
        sr= args.sr,
        duration= args.duration,
        num_chunks= args.num_chunks
    )
    if split == "TRAIN":
        data_loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=False
        )
    elif split == "VALID":
        data_loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False
        )
    elif split == "TEST":
        data_loader = DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False
        )
    elif split == "ALL":
        data_loader = DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False
        )
    return data_loader


def get_dataset(
        eval_dataset,
        data_path,
        split,
        sr,
        duration,
        num_chunks
    ):
    if eval_dataset == "annotation":
        dataset = Annotation_Dataset(data_path, split, sr, duration, num_chunks)
    else:
        print("error")
    return dataset
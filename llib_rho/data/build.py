from llib_rho.data.single import SingleDataset

def build_datasets(
    datasets_cfg,
    body_model_type,
    build_train=True,
    build_val=True,
):
    """
    Load all datasets specified in config file.
    Parameters
    ----------
    datasets_cfg: cfg
        config file of datasets
    body_model_type: str
        type of body model
    build_train: bool, optional
        whether to build training dataset
    build_val: bool, optional
        whether to build validation dataset
    """

    train_ds, val_ds = None, None

    assert len(datasets_cfg.train_names) <= 1, "Max. one training dataset in optimization"
    assert len(datasets_cfg.val_names) <= 1, "Max. one validation dataset in optimization"

    if len(datasets_cfg.train_names) == 1 and build_train:
        dataset_name = datasets_cfg.train_names[0]
        dataset_cfg = eval(f'datasets_cfg.{dataset_name}')
        # create dataset
        train_ds = SingleDataset(
            dataset_cfg=dataset_cfg, 
            dataset_name=dataset_name, 
            augmentation=datasets_cfg.augmentation,
            image_processing=datasets_cfg.processing,
            split='train',
            body_model_type=body_model_type,
        )

    if len(datasets_cfg.val_names) == 1 and build_val:
        dataset_name = datasets_cfg.val_names[0]
        dataset_cfg = eval(f'datasets_cfg.{dataset_name}')
        # create dataset
        val_ds = SingleDataset(
            dataset_cfg=dataset_cfg, 
            dataset_name=dataset_name, 
            augmentation=datasets_cfg.augmentation,
            image_processing=datasets_cfg.processing,
            split='val',
            body_model_type=body_model_type,
        )

    return train_ds, val_ds

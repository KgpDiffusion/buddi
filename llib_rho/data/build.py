from llib_rho.data.single import SingleDataset
from .single_optimization import SingleOptiDataset

def build_optimization_datasets(datasets_cfg, body_model_cfg):
    """
    Build datasets for optimization routine. 
    This function only returns a single dataset for each split.
    Parameters
    ----------
    datasets_cfg: cfg
        config file of datasets
    body_model_type: str
        type of body model
    """

    train_ds, val_ds, test_ds = None, None, None

    assert len(datasets_cfg.train_names) <= 1, "Max. one training dataset in optimization"
    assert len(datasets_cfg.val_names) <= 1, "Max. one validation dataset in optimization"
    assert len(datasets_cfg.test_names) <= 1, "Max. one test dataset in optimization"

    body_model_type = body_model_cfg.type
    joint_mapper = eval(f'body_model_cfg.{body_model_type}.init.joint_mapper')
    joint_mapper_type = joint_mapper.type
    use_hands = eval(f'joint_mapper.{joint_mapper_type}.use_hands')
    use_face = eval(f'joint_mapper.{joint_mapper_type}.use_face')
    use_face_contour = eval(f'joint_mapper.{joint_mapper_type}.use_face_contour')

    if len(datasets_cfg.train_names) == 1:
        dataset_name = datasets_cfg.train_names[0]
        dataset_cfg = eval(f'datasets_cfg.{dataset_name}')
        # create dataset
        # train_ds = SingleOptiDataset(
        #     dataset_cfg=dataset_cfg, 
        #     dataset_name=dataset_name, 
        #     image_processing=datasets_cfg.processing,
        #     split='train',
        #     body_model_type=body_model_type,
        #     use_hands=use_hands,
        #     use_face=use_face,
        #     use_face_contour=use_face_contour,
        # )
        train_ds = None

    if len(datasets_cfg.val_names) == 1:
        dataset_name = datasets_cfg.val_names[0]
        dataset_cfg = eval(f'datasets_cfg.{dataset_name}')
        # create dataset
        # val_ds = SingleOptiDataset(
        #     dataset_cfg=dataset_cfg, 
        #     dataset_name=dataset_name, 
        #     image_processing=datasets_cfg.processing,
        #     split='val',
        #     body_model_type=body_model_type,
        # )
        val_ds = None

    if len(datasets_cfg.test_names) == 1:
        dataset_name = datasets_cfg.test_names[0]
        dataset_cfg = eval(f'datasets_cfg.{dataset_name}')
        # create dataset
        test_ds = SingleOptiDataset(
            dataset_cfg=dataset_cfg, 
            dataset_name=dataset_name, 
            image_processing=datasets_cfg.processing,
            split='test',
            body_model_type=body_model_type,
        )

    return train_ds, val_ds, test_ds

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

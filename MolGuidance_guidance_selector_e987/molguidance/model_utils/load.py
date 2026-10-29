from molguidance.models.flowmol import FlowMol
from pathlib import Path
import copy
import math
import yaml
from molguidance.data_processing.data_module import MoleculeDataModule

def read_config_file(config_file: Path, _seen=None) -> dict:
    """Resolve explicit relative _base_ configs for reproducible ablations.

    The trainer saves the fully expanded result, so checkpoints do not depend
    on future edits to parent YAML files. Cycles and non-mapping files fail.
    """
    path = Path(config_file).resolve()
    seen = set() if _seen is None else set(_seen)
    if path in seen:
        raise ValueError(f"cyclic config inheritance: {path}")
    seen.add(path)
    with path.open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"config must be a mapping: {path}")
    parent = config.pop("_base_", None)
    if parent is None:
        return config
    base = read_config_file(path.parent / parent, seen)
    def merge(dst, src):
        for key, value in src.items():
            if isinstance(value, dict) and isinstance(dst.get(key), dict):
                merge(dst[key], value)
            else:
                dst[key] = value
        return dst
    return merge(base, config)


def resolve_training_resume(resume: Path) -> tuple[Path, Path]:
    """Resolve ordinary AND nested generation-quality checkpoints."""
    resume = Path(resume).resolve()
    if resume.is_dir():
        run_dir = resume
        checkpoint = run_dir / "checkpoints" / "last.ckpt"
    elif resume.is_file():
        checkpoint = resume
        run_dir = next(
            (parent for parent in checkpoint.parents if (parent / "config.yaml").is_file()),
            None,
        )
        if run_dir is None:
            raise ValueError("resume checkpoint has no parent run/config.yaml")
    else:
        raise ValueError("resume must be an existing run directory or checkpoint")
    if not (run_dir / "config.yaml").is_file() or not checkpoint.is_file():
        raise ValueError("resume requires both a saved config.yaml and checkpoint")
    return run_dir, checkpoint


def validate_chemistry_schema(config: dict) -> None:
    """Fail before training when dataset/model categorical widths disagree."""
    dataset = config['dataset']
    model = config['mol_fm']
    dataset_joint = bool(dataset.get('use_joint_atom_state', False))
    model_state_count = int(model.get('n_atom_states', 0))
    model_joint = model_state_count > 0
    if dataset_joint != model_joint:
        raise ValueError(
            'dataset.use_joint_atom_state and mol_fm.n_atom_states disagree'
        )
    if dataset_joint:
        dataset_state_count = int(dataset.get('n_atom_states', 0))
        if dataset_state_count != model_state_count:
            raise ValueError(
                'dataset.n_atom_states and mol_fm.n_atom_states disagree: '
                f'{dataset_state_count} != {model_state_count}'
            )
        if bool(dataset.get('use_hydrogen_counts', False)) or int(
            model.get('n_hydrogen_counts', 0)
        ) != 0:
            raise ValueError(
                'joint atom state cannot be combined with a separate H state'
            )

    dataset_bond_count = int(dataset.get('n_bond_types', 5))
    model_bond_count = int(model.get('n_bond_types', 5))
    if dataset_bond_count != model_bond_count:
        raise ValueError(
            'dataset.n_bond_types and mol_fm.n_bond_types disagree: '
            f'{dataset_bond_count} != {model_bond_count}'
        )

    self_conditioning = bool(
        config.get("vector_field", {}).get("self_conditioning", False)
    )
    self_conditioning_rate = float(model.get("self_conditioning_rate", 0.0))
    if (
        not math.isfinite(self_conditioning_rate)
        or not 0.0 <= self_conditioning_rate <= 1.0
    ):
        raise ValueError("mol_fm.self_conditioning_rate must be finite and in [0, 1]")
    if self_conditioning != (self_conditioning_rate > 0.0):
        raise ValueError(
            "vector_field.self_conditioning and a positive "
            "mol_fm.self_conditioning_rate must be enabled together"
        )
    if self_conditioning and model.get("parameterization", "ctmc") != "ctmc":
        raise ValueError("self-conditioning is currently supported only for CTMC")

    modality_time = model.get('modality_time', {}) or {}
    unknown_modality_keys = set(modality_time) - {
        'enabled', 'graph_clean_probability'
    }
    if unknown_modality_keys:
        raise ValueError(
            'unknown mol_fm.modality_time options: '
            + ', '.join(sorted(unknown_modality_keys))
        )
    independent_time = bool(modality_time.get('enabled', False))
    graph_clean_probability = float(
        modality_time.get('graph_clean_probability', 0.5)
    )
    if (
        not math.isfinite(graph_clean_probability)
        or not 0.0 <= graph_clean_probability <= 1.0
    ):
        raise ValueError(
            'mol_fm.modality_time.graph_clean_probability must be in [0, 1]'
        )
    vector_independent_time = config.get('vector_field', {}).get(
        'independent_modality_time'
    )
    if (
        vector_independent_time is not None
        and bool(vector_independent_time) != independent_time
    ):
        raise ValueError(
            'mol_fm.modality_time.enabled and '
            'vector_field.independent_modality_time disagree'
        )
    if independent_time and model.get('parameterization', 'ctmc') != 'ctmc':
        raise ValueError('independent modality time is supported only for CTMC')

    fake_atom_fraction = float(model.get('fake_atom_fraction', 0.0))
    fake_atom_std = float(model.get('fake_atom_std', 1.0))
    fake_atom_seed = int(model.get('fake_atom_validation_seed', 17_041))
    fake_capacity_mode = str(
        model.get('fake_atom_capacity_mode', 'stochastic')
    )
    if not math.isfinite(fake_atom_fraction) or not 0.0 <= fake_atom_fraction <= 1.0:
        raise ValueError('mol_fm.fake_atom_fraction must be finite and in [0, 1]')
    if not math.isfinite(fake_atom_std) or fake_atom_std <= 0.0:
        raise ValueError('mol_fm.fake_atom_std must be finite and positive')
    if fake_atom_seed < 0:
        raise ValueError('mol_fm.fake_atom_validation_seed must be non-negative')
    if fake_capacity_mode not in {'stochastic', 'maximum', 'none'}:
        raise ValueError(
            'mol_fm.fake_atom_capacity_mode must be stochastic, maximum, or none'
        )
    if fake_atom_fraction > 0.0:
        if not dataset_joint or model_state_count < 2:
            raise ValueError('fake atoms require a joint atom-state vocabulary')
        if model.get('parameterization', 'ctmc') != 'ctmc':
            raise ValueError('fake atoms are currently supported only for CTMC')
    for key, expected in (
        ('fake_atom_fraction', fake_atom_fraction),
        ('fake_atom_std', fake_atom_std),
        ('fake_atom_validation_seed', fake_atom_seed),
    ):
        if key in dataset and dataset[key] != expected:
            raise ValueError(f'dataset.{key} and mol_fm.{key} disagree')

    chemistry_config = model.get('chemistry_auxiliary', {})
    chemistry_enabled = any(
        float(value) > 0.0 for key, value in chemistry_config.items() if key.endswith("_weight")
    )
    if chemistry_enabled:
        if not dataset.get('require_chemistry_auxiliary_labels', False):
            raise ValueError(
                'enabled chemistry auxiliary losses require dataset '
                'require_chemistry_auxiliary_labels=true'
            )
        if not dataset_joint or model_bond_count != 4:
            raise ValueError(
                'chemistry auxiliary losses require joint atom states and '
                'four Kekule edge classes'
            )


def model_from_config(config: dict, seed_ckpt: Path = None) -> FlowMol:
    validate_chemistry_schema(config)
    # for qm9 dataset: find min and max values for Gaussian expansion
    Gaussian_expansion_start_dict = {
        'mu': 0.0, 'alpha': 6.31, 'homo': -0.4286, 'lumo': -0.175, 'gap': 0.0246, 'cv': 6.002,
    }

    Gaussian_expansion_stop_dict = {
        'mu': 29.5564, 'alpha': 196.62, 'homo': -0.1017,'lumo': 0.1935, 'gap': 0.6221, 'cv': 46.969,
    }

    atom_type_map = config['dataset']['atom_map']
    dataset_name = config['dataset'].get("dataset_name", 'qm9')
    conditional_generation = config['dataset']['conditioning']['enabled']
    property_embedding_dim = config['model_setting']['property_embedding_dim']
    gaussian_expansion = config['model_setting']['gaussian_expansion']['enabled']
    n_gaussians = config['model_setting']['gaussian_expansion']['n_gaussians']
    gaussian_start, gaussian_stop = None, None
    enable_tune_cv_min = config['model_setting']['gaussian_expansion'].get('enable_tune_cv_min', False)

    if gaussian_expansion and conditional_generation:
        properties = config['dataset']['conditioning']['property']
        gaussian_start = Gaussian_expansion_start_dict.get(properties)
        gaussian_stop = Gaussian_expansion_stop_dict.get(properties)

    # if config['dataset']['conditioning']['property'] == 'cv' and enable_tune_cv_min:
    #     gaussian_start = 20.0

    properties_handle_method = config['model_setting']['properties_handle_method']

    # get the sample interval (how many epochs between drawing/evaluating)
    sample_interval = config['training']['evaluation']['sample_interval']
    mols_to_sample = config['training']['evaluation']['mols_to_sample']


    # get the filepath of the n_atoms histogram
    processed_data_dir = Path(config['dataset']['processed_data_dir'])
    n_atoms_hist_filepath = processed_data_dir / 'train_data_n_atoms_histogram.pt'
    marginal_dists_file = processed_data_dir / 'train_data_marginal_dists.pt'

    if seed_ckpt is not None:
        model = FlowMol.load_from_checkpoint(seed_ckpt, 
                                            atom_type_map=atom_type_map,
                                            n_atoms_hist_file=n_atoms_hist_filepath,
                                            marginal_dists_file=marginal_dists_file,
                                            sample_interval=sample_interval,
                                            n_mols_to_sample=mols_to_sample,
                                            vector_field_config=config['vector_field'],
                                            interpolant_scheduler_config=config['interpolant_scheduler'], 
                                            lr_scheduler_config=config['lr_scheduler'],
                                            property_embedding_dim=property_embedding_dim,
                                            gaussian_expansion=gaussian_expansion,
                                            gaussian_start=gaussian_start,
                                            gaussian_stop=gaussian_stop,
                                            n_gaussians=n_gaussians,
                                            conditional_generation=conditional_generation,
                                            properties_handle_method=properties_handle_method,
                                            dataset_name=dataset_name,
                                            **config['mol_fm'])
    else:
        model = FlowMol(atom_type_map=atom_type_map, 
                        n_atoms_hist_file=n_atoms_hist_filepath,
                        marginal_dists_file=marginal_dists_file,
                        sample_interval=sample_interval,
                        n_mols_to_sample=mols_to_sample,
                        vector_field_config=config['vector_field'],
                        interpolant_scheduler_config=config['interpolant_scheduler'], 
                        lr_scheduler_config=config['lr_scheduler'],
                        property_embedding_dim=property_embedding_dim,
                        gaussian_expansion=gaussian_expansion,
                        gaussian_start=gaussian_start,
                        gaussian_stop=gaussian_stop,
                        n_gaussians=n_gaussians,
                        conditional_generation=conditional_generation,
                        properties_handle_method=properties_handle_method,
                        dataset_name=dataset_name,
                        **config['mol_fm'])
    
    return model

def data_module_from_config(config: dict) -> MoleculeDataModule:
    batch_size = config['training']['batch_size']
    num_workers = config['training']['num_workers']

    # determine if we are doing distributed training
    if config['training']['trainer_args']['devices'] > 1:
        distributed = True
    else:
        distributed = False

    dataset_config = copy.deepcopy(config['dataset'])
    for key, default in (
        ('fake_atom_fraction', 0.0),
        ('fake_atom_std', 1.0),
        ('fake_atom_validation_seed', 17_041),
    ):
        dataset_config[key] = config['mol_fm'].get(key, default)

    data_module = MoleculeDataModule(dataset_config=dataset_config,
                                     dm_prior_config=config['mol_fm']['prior_config'],
                                     batch_size=batch_size, 
                                     num_workers=num_workers, 
                                     distributed=distributed,
                                     max_num_edges=config['training'].get('max_num_edges'))
    
    return data_module

import logging
import os
from datasets import load_dataset, concatenate_datasets, Dataset
from omegaconf import DictConfig
import hydra

log = logging.getLogger(__name__)

def _load_full_dataset(task_cfg: DictConfig) -> Dataset:
    """Función auxiliar para cargar el dataset completo desde la ruta especificada."""
    data_path = os.path.join(hydra.utils.get_original_cwd(), task_cfg.data_path)
    log.info(f"Cargando dataset desde: {data_path}")
    try:
        return load_dataset('json', data_files=data_path, split='train')
    except Exception as e:
        log.error(f"No se pudo cargar el archivo del dataset en {data_path}: {e}")
        raise

def load_dataset_for_indexing(task_cfg: DictConfig) -> Dataset:
    """
    Carga y combina los splits de 'train' y 'validation' para la indexación.
    """
    dataset = _load_full_dataset(task_cfg)
    
    train_ds = dataset.filter(lambda x: x['split'] == 'train')
    val_ds = dataset.filter(lambda x: x['split'] == 'validation')

    combined_ds = concatenate_datasets([train_ds, val_ds])

    log.info(f"Cargados {len(train_ds)} elementos del train set y {len(val_ds)} del validation set.")
    log.info(f"Dataset combinado para indexación contiene {len(combined_ds)} elementos.")

    return combined_ds

def load_test_set(task_cfg: DictConfig) -> Dataset:
    """
    Carga únicamente el split de 'test' para la evaluación.
    """
    dataset = _load_full_dataset(task_cfg)
    
    test_ds = dataset.filter(lambda x: x['split'] == 'test')
    log.info(f"Cargados {len(test_ds)} elementos del test set para evaluación.")
    
    return test_ds

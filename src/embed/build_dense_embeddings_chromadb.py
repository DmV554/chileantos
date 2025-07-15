import hydra
from omegaconf import DictConfig
import os
import logging
from datasets import load_dataset, concatenate_datasets
import hashlib

# Imports de Haystack
from haystack.dataclasses import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack import Pipeline
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

log = logging.getLogger(__name__)

def load_train_val_data(task_cfg: DictConfig):
    """Carga y combina los splits 'train' y 'validation' en un solo dataset."""
    data_path = os.path.join(hydra.utils.get_original_cwd(), task_cfg.data_path)
    dataset = load_dataset('json', data_files=data_path, split='train')

    train_ds = dataset.filter(lambda x: x['split'] == 'train')
    val_ds = dataset.filter(lambda x: x['split'] == 'validation')

    combined_ds = concatenate_datasets([train_ds, val_ds])

    log.info(f"Cargados {len(train_ds)} elementos del train set.")
    log.info(f"Cargados {len(val_ds)} elementos del validation set.")
    log.info(f"Dataset combinado contiene {len(combined_ds)} elementos.")

    return combined_ds

def create_haystack_documents(dataset):
    """Convierte el dataset en una lista de Documentos Haystack con metadatos."""
    log.info("Creando Documentos Haystack con doc_id y metadatos compatibles...")
    haystack_documents = []
    
    for row in dataset:
        clause_text = str(row['text'])
        doc_id = hashlib.sha256(clause_text.encode()).hexdigest()
        
        label_keys_list = row['human_readable_labels']
        labels_as_string = ", ".join(label_keys_list) if label_keys_list else "N/A"
        
        metadata = {
            "doc_id": doc_id,
            "source_split": str(row.get('split')),
            "label_keys": labels_as_string,
        }
        
        doc = Document(content=clause_text, meta=metadata)
        haystack_documents.append(doc)
        
    log.info(f"Se crearon {len(haystack_documents)} Documentos Haystack.")
    return haystack_documents

@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # La configuración de la tarea se carga directamente en cfg.task
    task_cfg = cfg.task
    task_name_key = task_cfg.name.split(' - ')[-1].lower()
    log.info(f"--- Iniciando creación de KB con Haystack/ChromaDB para: '{task_cfg.name}' ---")

    # Cargar datos y crear documentos
    full_dataset = load_train_val_data(task_cfg)
    haystack_documents = create_haystack_documents(full_dataset)
    
    # Configurar ChromaDocumentStore con rutas de la nueva config
    storage_dir = cfg.db.naive.storage_dir.format(
        task_key=task_name_key, 
        embed_short=cfg.db.naive.embed_model_name_short
    )
    full_storage_path = os.path.join(hydra.utils.get_original_cwd(), storage_dir)

    document_store = ChromaDocumentStore(
        collection_name=task_name_key,
        persist_path=full_storage_path
    )
    log.info(f"Usando ChromaDB en la ruta: {full_storage_path}")
    
    # Configurar el pipeline de indexación
    doc_embedder = SentenceTransformersDocumentEmbedder(
        model=cfg.db.naive.embedding_model
    )
    writer = DocumentWriter(document_store=document_store)
    
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component("embedder", doc_embedder)
    indexing_pipeline.add_component("writer", writer)
    indexing_pipeline.connect("embedder.documents", "writer.documents")

    log.info(f"Ejecutando pipeline de indexación para {len(haystack_documents)} documentos...")
    indexing_pipeline.run({"embedder": {"documents": haystack_documents}})
    log.info("Indexación en ChromaDB completada.")
    
    log.info(f"--- ✅ KB de ChromaDB para '{task_cfg.name}' creada en: {full_storage_path} ---")

if __name__ == "__main__":
    main()
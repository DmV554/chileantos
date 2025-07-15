# worker_fewshot.py

import hydra
from omegaconf import DictConfig
import os
import ast
import mlflow
import pandas as pd
import logging
import re
from datasets import load_dataset
from openai import OpenAI
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
import docx2txt  # Para leer los nuevos prompts

# --- DEFINICIÓN DE FUNCIONES AUXILIARES ---
log = logging.getLogger(__name__)

def extract_shots_from_path(path: str) -> int:
    """
    Extrae el número de 'shots' del nombre de archivo del prompt usando regex.
    Busca un patrón como '_1-shot_', '_5-shot_', etc.
    """
    match = re.search(r'_(\d+)-shot', path)
    if match:
        return int(match.group(1))
    log.warning(f"No se pudo extraer el número de shots de la ruta: {path}. Se usará 0 como default.")
    return 0

def load_static_prompt(path: str):
    """
    Carga el contenido de un prompt estático desde un archivo .docx.
    """
    log.info(f"Cargando plantilla de prompt estático desde: {path}")
    full_path = os.path.join(hydra.utils.get_original_cwd(), path)
    try:
        return docx2txt.process(full_path)
    except Exception as e:
        log.error(f"No se pudo leer el archivo .docx en {full_path}: {e}")
        raise

def load_data_for_evaluation(task_cfg: DictConfig):
    """
    Carga el test set para evaluación y las etiquetas extendidas del config.
    """
    data_path = os.path.join(hydra.utils.get_original_cwd(), task_cfg.data_path)
    log.info(f"Cargando dataset desde: {data_path}")
    dataset = load_dataset('json', data_files=data_path, split='train')
    
    test_ds = dataset.filter(lambda x: x['split'] == 'test')
    log.info(f"Cargados {len(test_ds)} elementos del test set para evaluación.")
    
    if not hasattr(task_cfg, 'options_extended'):
        raise ValueError("La configuración de la tarea debe tener una sección 'options_extended' para este worker.")
    
    # Usamos las nuevas 'options_extended' para la evaluación
    possible_options_extended = list(task_cfg.options_extended.values())
    return test_ds, possible_options_extended

def parse_model_output(output: str):
    """
    Parsea la salida del modelo. Se espera una sola etiqueta en formato de texto.
    """
    # Limpiamos espacios y comillas que pueda añadir el modelo
    # Se devuelve como lista para ser compatible con el MultiLabelBinarizer
    return [output.strip().replace("'", "").replace('"', '')]

def openai_query(client, model: str, prompt: str):
    """Función de consulta a la API de OpenAI."""
    # Como todo el contexto está en el prompt, usamos solo el rol 'user'
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(model=model, messages=messages, temperature=0.0)
    return response.choices[0].message.content

def run_inference_loop(cfg: DictConfig, task_cfg: DictConfig, client: OpenAI, prompt_template: str, test_ds):
    """
    Ejecuta el bucle de inferencia con el prompt estático.
    """
    true_labels, pred_labels, model_outputs = [], [], []
    
    # Creamos un mapeo de las claves cortas del dataset (ej: 'NA')
    # a las etiquetas extendidas del prompt (ej: 'Cláusula no aplicable')
    label_mapping = dict(task_cfg.options_extended)

    log.info(f"Iniciando inferencia en {len(test_ds)} cláusulas de prueba...")
    for item in tqdm(test_ds, desc="Procesando cláusulas"):
        # La lógica ahora es una simple sustitución del placeholder
        prompt = prompt_template.replace("{{ }}", item['text'])
        
        # Usa la clave unificada para el modelo LLM
        output = openai_query(client, cfg.models.llm, prompt)
        detected = parse_model_output(output)
        
        print(f"OUTPUT -> {output} \n\n")
        print(f"DETECTED -> {detected} \n\n")
        
        
        # Convertimos las etiquetas verdaderas del dataset a su formato extendido para poder evaluarlas
        true_labels_extended = [label_mapping.get(lbl, lbl) for lbl in item['human_readable_labels']]
        
        true_labels.append(true_labels_extended)
        pred_labels.append(detected)
        model_outputs.append(output)
        
    log.info("--- Inferencia Completada ---")
    return true_labels, pred_labels, model_outputs

def evaluate_and_log(true_labels, pred_labels, possible_options_extended, model_outputs, test_ds):
    """Calcula métricas y las registra en MLflow."""
    mlb = MultiLabelBinarizer(classes=possible_options_extended)
    true_bin = mlb.fit_transform(true_labels)
    pred_bin = mlb.transform(pred_labels)

    metrics = {
        "micro_f1": f1_score(true_bin, pred_bin, average='micro', zero_division=0),
        "macro_f1": f1_score(true_bin, pred_bin, average='macro', zero_division=0)
    }
    mlflow.log_metrics(metrics)
    log.info(f"Resultados -> Micro-F1: {metrics['micro_f1']:.4f}, Macro-F1: {metrics['macro_f1']:.4f}")
    
    report = classification_report(true_bin, pred_bin, target_names=possible_options_extended, zero_division=0, output_dict=True)
    mlflow.log_dict(report, "classification_report.json")
    
    pd.DataFrame({
        'text': [item['text'] for item in test_ds],
        'true_labels': true_labels,
        'predicted_labels': pred_labels,
        'model_output': model_outputs
    }).to_csv("detailed_results.csv", index=False)
    mlflow.log_artifact("detailed_results.csv")

@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Orquesta el flujo del experimento."""
    # La configuración de la tarea activa se carga directamente en cfg.task gracias a Hydra
    task_cfg = cfg.task

    # --- Acceder a la configuración desde las rutas correctas ---
    mlflow.set_tracking_uri(cfg.run_config.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.run_config.mlflow_experiment_name)
    
    # Usa la clave unificada para el modelo LLM
    run_name = f"fewshot_{task_cfg.name.replace(' ', '_')}_{cfg.models.llm}"

    with mlflow.start_run(run_name=run_name) as run:
        # Verifica y setea el parent run ID desde la sub-configuración
        if "parent_run_id" in cfg.run_config and cfg.run_config.parent_run_id != "placeholder":
            mlflow.set_tag("mlflow.parentRunId", cfg.run_config.parent_run_id)

        # Extraer el número de shots del path del prompt, que ahora está en la estrategia
        prompt_path = cfg.strategy.fewshot.prompt_shot_path
        few_shot_n = extract_shots_from_path(prompt_path)

        params_to_log = {
            "task_name": task_cfg.name,
            "model_name": cfg.models.llm,  # Usa la clave unificada
            "prompt_path": prompt_path, # Usa la variable local
            "few_shot_n": few_shot_n
        }
        mlflow.log_params(params_to_log)
        
        client = OpenAI(api_key=cfg.run_config.api_key, base_url=cfg.run_config.base_url_model)
        
        # 1. Cargar el prompt estático desde el .docx, usando la ruta de la estrategia
        prompt_template = load_static_prompt(prompt_path)
        
        # 2. Cargar datos para evaluación y las opciones extendidas
        test_ds, possible_options_extended = load_data_for_evaluation(task_cfg)
        
        # 3. Ejecutar la inferencia (lógica simplificada)
        true_labels, pred_labels, model_outputs = run_inference_loop(
            cfg, task_cfg, client, prompt_template, test_ds
        )
        
        # 4. Evaluar y registrar resultados
        evaluate_and_log(true_labels, pred_labels, possible_options_extended, model_outputs, test_ds)
        
        log.info(f"✅ Experimento '{task_cfg.name}' finalizado y registrado en MLflow.")

if __name__ == "__main__":
    main()
# run_experiments.py

import yaml
import subprocess
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import argparse
from datetime import datetime
from itertools import product

# --- Configuración General ---
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constantes de Configuración ---
RESULTS_DIR = "results"
# Asegúrate de que la ruta al archivo de configuración sea la correcta para tu sistema
CONFIG_FILE = "config/experiments.yaml"
#CONFIG_FILE = "/srv/dmiranda/graphrag_chileantos/graph-rag/experiments.yaml"

# URI de tu servidor de MLflow
MLFLOW_TRACKING_URI = "http://localhost:5002"

def generate_and_log_plot(task_df: pd.DataFrame, task_name: str, metric: str, results_dir: str):
    """
    Genera un gráfico para una métrica específica, lo guarda localmente
    y como artefacto en la corrida activa de MLflow.
    """
    log.info(f"Generando gráfico para '{task_name}' - Métrica: {metric}")
    
    metric_mean_col = f'{metric}_mean'
    
    # Determinar dinámicamente el parámetro para el 'hue' del gráfico
    hue_param = None
    if 'few_shot_n' in task_df.columns:
        hue_param = 'few_shot_n'
    elif 'similarity_top_k' in task_df.columns:
        hue_param = 'similarity_top_k'

    plt.figure(figsize=(15, 8))
    sns.barplot(
        data=task_df,
        x='model_name',
        y=metric_mean_col,
        hue=hue_param
    )
    
    plt.title(f'Rendimiento {metric.upper()} para la Tarea: {task_name}', fontsize=16)
    plt.xlabel('Modelo LLM', fontsize=12)
    plt.ylabel(f'{metric.upper()}-Score (Media)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    if hue_param:
        plt.legend(title=hue_param.replace('_', ' ').capitalize())
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plot_path = os.path.join(results_dir, f"plot_{metric}_{task_name.replace(' ', '_')}.png")
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path)
    log.info(f"Gráfico guardado y logueado: {plot_path}")
    plt.close()

def main():
    """
    Función principal que orquesta todo, leyendo la suite activa desde el archivo YAML.
    """
    # 1. Cargar el plan de batalla
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)
    
    # --- CAMBIO: Leemos la suite activa desde el archivo de configuración ---
    suite_to_run = config.get('active_suite')
    if not suite_to_run:
        raise ValueError("La clave 'active_suite' no está definida en experiments.yaml. Debes especificar qué suite ejecutar.")

    if suite_to_run not in config:
        raise ValueError(f"La suite '{suite_to_run}' especificada en 'active_suite' no existe en el archivo de configuración.")
    
    log.info(f"--- Ejecutando la suite de experimentos: '{suite_to_run}' ---")
    
    suite_config = config[suite_to_run]
    experiment_name = suite_config['mlflow_experiment_name']
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    # El resto del script sigue la misma lógica de "Run Padre"
    with mlflow.start_run(run_name=f"Orchestrator_{suite_to_run}_{datetime.now().strftime('%Y%m%d-%H%M')}") as parent_run:
        parent_run_id = parent_run.info.run_id
        log.info(f"Iniciado Run Padre con ID: {parent_run_id} en el experimento '{experiment_name}'")
        mlflow.log_dict(suite_config, f"suite_{suite_to_run}_config.yaml")

        worker_path = suite_config['worker_script_path']
        params_matrix = suite_config['parameters']
        num_runs = suite_config.get('num_runs_per_experiment', 1)

        param_keys = params_matrix.keys()
        param_values = params_matrix.values()
        
        # Bucle genérico para ejecutar todas las combinaciones de la suite
        for combo in product(*param_values):
            run_params = dict(zip(param_keys, combo))
            for i in range(num_runs):
                log.info(f"-> Ejecutando Run {i+1}/{num_runs} con params: {run_params}")
                
                # Pasa los overrides con sus rutas completas (fully qualified keys) según config.yaml
                command = ["python3", worker_path, f"run_config.parent_run_id={parent_run_id}"]
                command.append(f"run_config.mlflow_experiment_name={experiment_name}")
                command.append(f"run_config.mlflow_tracking_uri={MLFLOW_TRACKING_URI}")

                # Construcción dinámica de overrides para el worker, alineada con config.yaml
                for key, value in run_params.items():
                    if key == 'tasks':
                        # Esto selecciona el grupo de configuración de tarea correcto (ej. task=illegal carga illegal.yaml)
                        command.append(f"task={value}")
                    
                    elif key in ['few_shot_models', 'ollama_rag_models', 'llm_models']:
                        command.append(f"models.llm={value}")
                    
                    # --- Lógica de prompts diferenciada por suite ---
                    elif key == 'prompts':
                        if suite_to_run == 'few_shot_suite':
                            # Para few-shot, el prompt va a la configuración de la estrategia
                            command.append(f"strategy.fewshot.prompt_shot_path={value}")
                        else:
                            # Para RAG y otras suites, va al prompt general en config.yaml
                            command.append(f"prompts.prompt={value}")
                    
                    elif key == 'embedding_models':
                        command.append(f"db.naive.embedding_model={value}")
                        short_name = str(value).split('/')[-1]
                        command.append(f"db.naive.embed_model_name_short={short_name}")
                        
                    elif key == 'similarity_top_k':
                        command.append(f"strategy.naive_rag.top_k={value}")
                    
                    # Parámetros específicos para la estrategia Hybrid RAG
                    elif key == 'retriever_top_k':
                        command.append(f"strategy.hybrid_rag.top_k_retriever={value}")
                    elif key == 'ranker_top_k':
                        command.append(f"strategy.hybrid_rag.top_k_ranker={value}")

                    # Parámetros específicos para la estrategia GraphRAG
                    elif key == 'graphrag_projects':
                        command.append(f"strategy.graphrag.project_path={value}")
                
                try:
                    subprocess.run(command, check=True, text=True)
                except subprocess.CalledProcessError as e:
                    log.error(f"La corrida con params {run_params} falló. Error: {e.stderr}")
        
        log.info("--- Todas las corridas hijas han finalizado ---")
        
        # --- Análisis de Resultados ---
        log.info("--- Analizando resultados y generando reporte ---")
        filter_string = f"tags.`mlflow.parentRunId` = '{parent_run_id}'"
        runs_df = mlflow.search_runs(experiment_ids=[parent_run.info.experiment_id], filter_string=filter_string)

        if runs_df.empty:
            log.error("No se encontraron corridas hijas para analizar.")
            return

        runs_df.columns = [col.replace("metrics.", "").replace("params.", "") for col in runs_df.columns]
        
        # Parámetros para agrupar los resultados
        grouping_params = [p for p in ['task_name', 'model_name', 'few_shot_n', 'similarity_top_k', 'embedding_model'] if p in runs_df.columns]
        metrics_to_agg = ['micro_f1', 'macro_f1']
        
        for col in metrics_to_agg: runs_df[col] = pd.to_numeric(runs_df[col], errors='coerce')

        agg_df = runs_df.groupby(grouping_params)[metrics_to_agg].agg(['mean', 'std']).reset_index()
        agg_df.columns = ['_'.join(col).strip('_') for col in agg_df.columns.values]

        if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)
        report_path = os.path.join(RESULTS_DIR, "summary_report.csv")
        agg_df.to_csv(report_path, index=False)
        mlflow.log_artifact(report_path)
        log.info(f"Reporte de resumen guardado y logueado: {report_path}")

        # Generación de gráficos
        for task_name in agg_df['task_name'].unique():
            task_df = agg_df[agg_df['task_name'] == task_name]
            generate_and_log_plot(task_df, task_name, "macro_f1", RESULTS_DIR)
            generate_and_log_plot(task_df, task_name, "micro_f1", RESULTS_DIR)

        log.info("--- Análisis completado. Artefactos logueados en el Run Padre. ---")

# --- CAMBIO: El punto de entrada ahora simplemente llama a main() ---
if __name__ == "__main__":
    main()
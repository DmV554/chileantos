import hydra
from omegaconf import DictConfig
import os
import ast
import mlflow
import pandas as pd
import logging
import re
import sys

# --- Inicio: Modificación del Path para Imports Absolutos ---
# Añade la raíz del proyecto al sys.path
# Esto permite que el script se ejecute tanto directamente como a través de un orquestador
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- Fin: Modificación del Path ---

from openai import OpenAI
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
import docx2txt

from src.utils.data_loader import load_test_set
from src.pipelines.common.base_worker import BaseExperimentWorker

log = logging.getLogger(__name__)

# --- Funciones de utilidad específicas para este worker ---

def extract_shots_from_path(path: str) -> int:
    match = re.search(r'_(\d+)-shot', path)
    if match:
        return int(match.group(1))
    log.warning(f"No se pudo extraer el número de shots de la ruta: {path}. Se usará 0 como default.")
    return 0

def load_static_prompt(path: str):
    full_path = os.path.join(hydra.utils.get_original_cwd(), path)
    try:
        return docx2txt.process(full_path)
    except Exception as e:
        log.error(f"No se pudo leer el archivo .docx en {full_path}: {e}")
        raise

def parse_model_output(output: str):
    return [output.strip().replace("'", "").replace('"', '')]

def openai_query(client, model: str, prompt: str):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(model=model, messages=messages, temperature=0.0)
    return response.choices[0].message.content

class FewShotWorker(BaseExperimentWorker):
    """
    Worker para experimentos de Few-Shot. Hereda de la clase base
    y solo implementa la lógica específica de la tarea.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.strategy_cfg = self.cfg.strategy.fewshot

    def _get_run_name(self) -> str:
        return f"FewShot_{self.task_name_key}_{self.cfg.models.llm}"

    def _get_parameters_to_log(self) -> dict:
        prompt_path = self.strategy_cfg.prompt_shot_path
        return {
            "task_name": self.task_cfg.name,
            "model_name": self.cfg.models.llm,
            "prompt_path": prompt_path,
            "few_shot_n": extract_shots_from_path(prompt_path)
        }

    def _execute_task(self) -> None:
        client = OpenAI(api_key=self.run_config.api_key, base_url=self.run_config.base_url_model)
        prompt_template = load_static_prompt(self.strategy_cfg.prompt_shot_path)
        
        test_ds = load_test_set(self.task_cfg)
        possible_options_extended = list(self.task_cfg.options_extended.values())
        
        true_labels, pred_labels, model_outputs = self._run_inference_loop(
            client, prompt_template, test_ds
        )
        
        self._evaluate_and_log(true_labels, pred_labels, possible_options_extended, model_outputs, test_ds)

    def _run_inference_loop(self, client: OpenAI, prompt_template: str, test_ds):
        true_labels, pred_labels, model_outputs = [], [], []
        label_mapping = dict(self.task_cfg.options_extended)

        log.info(f"Iniciando inferencia en {len(test_ds)} cláusulas de prueba...")
        for item in tqdm(test_ds, desc="Procesando cláusulas"):
            prompt = prompt_template.replace("{{ }}", item['text'])
            output = openai_query(client, self.cfg.models.llm, prompt)
            detected = parse_model_output(output)
            
            true_labels_extended = [label_mapping.get(lbl, lbl) for lbl in item['human_readable_labels']]
            
            true_labels.append(true_labels_extended)
            pred_labels.append(detected)
            model_outputs.append(output)
            
        return true_labels, pred_labels, model_outputs

    def _evaluate_and_log(self, true_labels, pred_labels, possible_options, model_outputs, test_ds):
        mlb = MultiLabelBinarizer(classes=possible_options)
        true_bin = mlb.fit_transform(true_labels)
        pred_bin = mlb.transform(pred_labels)

        metrics = {
            "micro_f1": f1_score(true_bin, pred_bin, average='micro', zero_division=0),
            "macro_f1": f1_score(true_bin, pred_bin, average='macro', zero_division=0)
        }
        mlflow.log_metrics(metrics)
        log.info(f"Resultados -> Micro-F1: {metrics['micro_f1']:.4f}, Macro-F1: {metrics['macro_f1']:.4f}")
        
        report = classification_report(true_bin, pred_bin, target_names=possible_options, zero_division=0, output_dict=True)
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
    """Punto de entrada principal. Instancia y ejecuta el worker."""
    worker = FewShotWorker(cfg)
    worker.run()

if __name__ == "__main__":
    main()
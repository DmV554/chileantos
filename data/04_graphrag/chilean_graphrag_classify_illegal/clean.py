import json
import os

# --- CONFIGURACIÓN ---
# Nombre de tu archivo JSON de entrada (que contiene la lista completa)
input_filename = 'input/Illegal.json'
# Nombre del archivo JSON de salida que se creará
output_filename = 'Illegal_train_val.json' # <-- CAMBIADO a .json

def filtrar_y_exportar_a_json(input_path, output_path):
    """
    Lee un archivo JSON, filtra los registros para mantener solo los splits
    'train' y 'validation', y guarda el resultado en un nuevo archivo JSON estándar.
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            full_data = json.load(f)
            print(f"Se cargaron {len(full_data)} registros en total desde '{input_path}'.")

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de entrada '{input_path}'.")
        return
    except json.JSONDecodeError:
        print(f"Error: El archivo '{input_path}' no contiene un JSON válido.")
        return

    # La lógica de filtrado no cambia
    filtered_data = [
        record for record in full_data 
        if record.get("split") in ["train", "validation"]
    ]

    print(f"Se conservaron {len(filtered_data)} registros ('train' y 'validation').")

    # --- INICIO DE LA MODIFICACIÓN ---
    # Guarda la lista filtrada completa como un único objeto JSON en el archivo de salida
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            # json.dump escribe la estructura de Python (nuestra lista) directamente al archivo.
            # indent=4 formatea el archivo para que sea legible por humanos.
            # ensure_ascii=False preserva caracteres en español.
            json.dump(filtered_data, f, ensure_ascii=False, indent=4)
        
        print(f"Los datos filtrados se han guardado exitosamente en '{output_path}'.")

    except IOError as e:
        print(f"Error al escribir en el archivo de salida '{output_path}': {e}")
    # --- FIN DE LA MODIFICACIÓN ---


# --- Ejecutar el script ---
if __name__ == "__main__":
    # Construimos rutas relativas al directorio actual para mayor robustez
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file_path = os.path.join(current_dir, input_filename)
    output_file_path = os.path.join(current_dir, output_filename)
    
    filtrar_y_exportar_a_json(input_file_path, output_file_path)
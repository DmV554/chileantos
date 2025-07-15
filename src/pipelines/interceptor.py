# interceptor.py - Versión simple y robusta

import requests
import json
from functools import wraps

class SimpleOllamaInterceptor:
    """Interceptor simple que funciona en la mayoría de casos"""
    
    def __init__(self):
        self.original_post = requests.post
        self.installed = False
    
    def intercept_post(self, *args, **kwargs):
        print("=== INTERCEPTANDO LLAMADA A OLLAMA ===")
        print("URL:", args[0] if args else "No URL")
        print("Headers:", kwargs.get('headers', {}))
        
        if 'json' in kwargs:
            print("Payload JSON:")
            try:
                print(json.dumps(kwargs['json'], indent=2, ensure_ascii=False))
            except:
                print(kwargs['json'])
            
            # Verificar el prompt
            if 'messages' in kwargs['json']:
                for i, msg in enumerate(kwargs['json']['messages']):
                    print(f"\nMensaje {i} ({msg.get('role', 'unknown')}):")
                    content = msg.get('content', '')
                    if len(content) > 500:
                        print(f"[TRUNCADO] {content[:500]}...")
                    else:
                        print(content)
        
        print("\n" + "="*50)
        
        # Hacer la llamada original
        response = self.original_post(*args, **kwargs)
        
        print("=== RESPUESTA DE OLLAMA ===")
        print("Status:", response.status_code)
        try:
            resp_json = response.json()
            print("Response JSON:")
            print(json.dumps(resp_json, indent=2, ensure_ascii=False))
        except:
            print("Response text:", response.text)
        print("="*50)
        
        return response
    
    def install(self):
        """Instala el interceptor"""
        if not self.installed:
            requests.post = self.intercept_post
            self.installed = True
            print("Interceptor instalado correctamente")
    
    def uninstall(self):
        """Desinstala el interceptor"""
        if self.installed:
            requests.post = self.original_post
            self.installed = False
            print("Interceptor desinstalado")

# Función de conveniencia
def install_ollama_interceptor():
    """Función simple para instalar el interceptor"""
    interceptor = SimpleOllamaInterceptor()
    interceptor.install()
    return interceptor

# Uso directo si se ejecuta como script
if __name__ == "__main__":
    install_ollama_interceptor()
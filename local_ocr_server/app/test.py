import os
import requests
import time

INPUT_DIR = "./input_pdfs"
OUTPUT_DIR = "./output_texts"
API_URL = "http://localhost:8506/ocr"


# Создаем директории, если их нет
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_directory():
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith('.pdf')]
    
    if not files:
        print(f"No PDFs found in {INPUT_DIR}")
        return

    print(f"Found {len(files)} PDFs. Processing...")

    for filename in files:
        file_path = os.path.join(INPUT_DIR, filename)
        file_base_name = os.path.splitext(filename)[0]
        
        # Создаем подпапку для результата (как в ТЗ)
        file_output_dir = os.path.join(OUTPUT_DIR, file_base_name)
        os.makedirs(file_output_dir, exist_ok=True)
        output_txt_path = os.path.join(file_output_dir, "result.txt")

        print(f"Sending {filename}...", end=" ", flush=True)
        start_time = time.time()

        try:
            with open(file_path, 'rb') as f:
                response = requests.post(API_URL, files={'file': f})
            
            if response.status_code == 200:
                result = response.json()
                with open(output_txt_path, 'w', encoding='utf-8') as out_f:
                    out_f.write(result['text'])
                duration = time.time() - start_time
                print(f"Done in {duration:.2f}s. Saved to {output_txt_path}")
            else:
                print(f"Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"Failed: {e}")

if __name__ == "__main__":
    process_directory()
# LightOnOCR-docker

A containerized FastAPI service for LightOnOCR-1B-1025 model with Docker Compose orchestration. This project provides a high-performance OCR (Optical Character Recognition) service with parallel processing capabilities for PDF documents.

## Overview

This project deploys the LightOnOCR-1B-1025 model using a two-service architecture with Docker Compose:

- **vLLM Server**: Backend inference server running the LightOnOCR model with GPU acceleration
- **FastAPI Service**: Frontend API for OCR processing with async parallel processing
- **GPU Support**: CUDA-enabled Docker containers with NVIDIA GPU acceleration
- **Parallel Processing**: Concurrent page processing with configurable concurrency (default: 8 parallel requests)
- **PDF Support**: Automatic multi-page PDF to image conversion (200 DPI)
- **Client Script**: Python client for batch processing PDFs

## Project Structure

```
.
├── client.py              # Python client for batch PDF processing
├── docker-compose.yml     # Docker Compose configuration (2 services)
├── README.md              # This file
└── service/
    ├── app.py             # FastAPI application with async parallel processing
    ├── Dockerfile         # Docker image definition
    └── start.sh           # Legacy startup script (not used)
```

## Architecture

The system consists of two Docker services:

1. **vllm-server**: Runs the LightOnOCR-1B-1025 model using vLLM OpenAI-compatible server
   - Internal port: 8000
   - External port: 8507
   - Handles model inference with GPU acceleration

2. **ocr-app**: FastAPI frontend that orchestrates OCR processing
   - Port: 8506
   - Converts PDFs to images
   - Sends parallel requests to vLLM server
   - Aggregates results and manages output

## Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit
- Hugging Face authentication token (HF_TOKEN)

## Setup & Installation

### 1. Set Hugging Face Token

```bash
export HF_TOKEN=your_hugging_face_token
```

### 2. Build and Run with Docker Compose

```bash
docker-compose up --build
```

This will:
- Pull the vLLM OpenAI image
- Build the FastAPI service image from `service/Dockerfile`
- Start the vLLM inference server on port 8507 (internal: 8000)
- Start the FastAPI OCR service on port 8506
- Mount Hugging Face cache for model caching
- Allocate GPU resources (device 0, 20% memory utilization)

### 3. Verify Services are Running

```bash
# Check both services are up
docker-compose ps

# Check vLLM server health
curl http://localhost:8507/health

# View logs
docker-compose logs -f
```

## Configuration

### Service Configuration (service/app.py)

Key parameters:

- **PDF_DPI**: Resolution for PDF conversion (default: 200)
- **SAVE_QUALITY**: JPEG quality for saved images (default: 90)
- **MAX_CONCURRENT_REQUESTS**: Number of parallel OCR requests (default: 8)
- **VLLM_URL**: URL of the vLLM inference server (default: `http://vllm-server:8000/v1/chat/completions`)
- **OUTPUT_DIR**: Directory for saving processed outputs (default: `./output_texts`)

### Docker Configuration (docker-compose.yml)

**vllm-server**:
- **Port**: 8507 (external) → 8000 (internal)
- **GPU Device**: CUDA device 0
- **GPU Memory Utilization**: 20% (0.2)
- **Shared Memory**: 8GB
- **Volume**: Hugging Face cache mounted at `/root/.cache/huggingface`

**ocr-app**:
- **Port**: 8506
- **Volumes**:
  - `./service` mounted at `/app` (with --reload)
  - `./output_texts` mounted at `/app/output_texts`

## Usage

### API Endpoint

**POST** `/ocr` - Process a PDF file (multi-page PDFs supported)

Request:
```bash
curl -X POST "http://localhost:8506/ocr" \
  -F "file=@path/to/document.pdf"
```

Response:
```json
{
  "filename": "document.pdf",
  "text": "Extracted OCR text from all pages...\n\n--- Page Break ---\n\nPage 2 content...",
  "processing_time": 45.3
}
```

**Note**: Currently only PDF files are accepted (validated by file extension)

### Batch Processing with Client

The `client.py` script processes all PDFs in the `input_pdfs/` directory sequentially:

```bash
# Create input directory and add PDFs
mkdir -p input_pdfs
cp your_files.pdf input_pdfs/

# Run the client
python client.py
```

Output structure:
```
output_texts/
└── document_name/
    ├── images/              # Converted page images
    │   ├── image_0000.jpeg
    │   ├── image_0001.jpeg
    │   └── ...
    └── result.txt           # Complete OCR text
```

## API Details

### Supported Formats
- **PDFs**: Multi-page PDFs (converted to images at 200 DPI)

### Response Fields
- `filename`: Original filename of the processed PDF
- `text`: Extracted OCR text with page breaks (`--- Page Break ---` separators)
- `processing_time`: Total processing time in seconds

### Processing Flow
1. PDF uploaded to `/ocr` endpoint
2. PDF converted to individual JPEG images (page by page)
3. Images processed in parallel (up to 8 concurrent requests to vLLM)
4. Results aggregated in correct page order
5. Complete text saved to `output_texts/{filename}/result.txt`
6. Images saved to `output_texts/{filename}/images/`

## Environment Variables

**vllm-server**:
- `CUDA_VISIBLE_DEVICES`: GPU device to use (default: 0)
- `HF_TOKEN`: Hugging Face authentication token (required for model access)

**ocr-app**:
- `VLLM_URL`: URL of the vLLM inference server (default: `http://vllm-server:8000/v1/chat/completions`)

## Troubleshooting

### GPU Not Detected
- Verify NVIDIA Container Toolkit is installed: `docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu20.04 nvidia-smi`
- Check `CUDA_VISIBLE_DEVICES` in docker-compose.yml

### Out of Memory (OOM)
- Reduce `MAX_CONCURRENT_REQUESTS` in service/app.py (try 4 or 2)
- Reduce `--gpu-memory-utilization` in docker-compose.yml
- Increase `shm_size` in docker-compose.yml (currently 8gb)
- Lower `PDF_DPI` in service/app.py

### Service Communication Issues
- Ensure both services are running: `docker-compose ps`
- Check vLLM logs: `docker-compose logs vllm-server`
- Check FastAPI logs: `docker-compose logs ocr-app`
- Verify vLLM is healthy: `curl http://localhost:8507/health`

### Model Download Issues
- Ensure `HF_TOKEN` is set correctly
- Verify internet connectivity in container
- Check Hugging Face cache directory permissions

## Service Endpoints

- **OCR API**: http://localhost:8506/ocr (POST endpoint for PDF processing)
- **vLLM Server**: http://localhost:8507 (vLLM OpenAI-compatible API)
- **vLLM Health**: http://localhost:8507/health (Health check endpoint)

## Development

The ocr-app service runs with `--reload` flag, which means:
- Code changes in `service/app.py` are automatically detected
- The service restarts automatically on code changes
- No need to rebuild the container during development
- The `service/` directory is mounted as a volume for live updates

## Performance Tips

- **Parallel Processing**: Adjust `MAX_CONCURRENT_REQUESTS` in app.py (default: 8)
  - Higher values = faster processing but more GPU memory usage
  - Lower values = slower but more stable on limited GPU memory
- **GPU Memory**: Adjust `--gpu-memory-utilization` in docker-compose.yml (default: 0.2)
  - Increase if you have more VRAM available
  - Decrease if experiencing OOM errors
- **PDF DPI**: Adjust `PDF_DPI` in app.py (default: 200)
  - Higher DPI = better quality but slower processing and more memory
  - Lower DPI = faster but may reduce OCR accuracy
- **Batch Processing**: Use client.py for processing multiple PDFs sequentially
- **Hot Reload**: The ocr-app service runs with `--reload` flag for development

## License

Refer to LightOnOCR documentation for model licensing terms.

## References

- [LightOnOCR Model](https://huggingface.co/lightonai/LightOnOCR-1B-1025)
- [vLLM Documentation](https://docs.vllm.ai/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Documentation](https://docs.docker.com/)

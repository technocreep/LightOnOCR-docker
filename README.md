# LightOnOCR-docker

A containerized FastAPI service for LightOnOCR-1B-1025 model with Docker Compose orchestration. This project provides an OCR (Optical Character Recognition) service that supports PDF and image processing with advanced preprocessing capabilities.

## Overview

This project wraps the LightOnOCR-1B-1025 model in a FastAPI application and deploys it using Docker Compose. It includes:

- **FastAPI Service**: RESTful API for OCR processing
- **GPU Support**: CUDA-enabled Docker container with NVIDIA GPU acceleration
- **Image Preprocessing**: CLAHE (Contrast Limited Adaptive Histogram Equalization) and sharpening for improved OCR accuracy
- **PDF Support**: Automatic PDF to image conversion (400 DPI)
- **Client Script**: Python client for batch processing PDFs

## Project Structure

```
.
├── client.py              # Python client for batch PDF processing
├── docker-compose.yml     # Docker Compose configuration
├── README.md              # This file
└── service/
    ├── app.py             # FastAPI application
    ├── Dockerfile         # Docker image definition
    └── start.sh           # Service startup script
```

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
- Build the Docker image from `service/Dockerfile`
- Start the OCR service on port 8506
- Mount Hugging Face cache for model caching
- Allocate GPU resources as specified

## Configuration

### Service Configuration (service/app.py)

Key parameters:

- **PREPROCESS**: Enable/disable image preprocessing (default: True)
- **CLAHE_CLIP_LIMIT**: Contrast limiting threshold (default: 0.2)
- **CLAHE_GRID_SIZE**: Grid size for adaptive histogram equalization (default: (4, 4))
- **SHARP_STRENGTH**: Image sharpening intensity (default: 1.8)
- **PDF_DPI**: Resolution for PDF conversion (default: 400)

### Docker Configuration (docker-compose.yml)

- **Container Port**: 8506
- **GPU Device**: CUDA device 0
- **Shared Memory**: 8GB (for model processing)
- **Volume**: Hugging Face cache mounted at `/root/.cache/huggingface`

## Usage

### API Endpoint

**POST** `/ocr` - Process an image or PDF file

Request:
```bash
curl -X POST "http://localhost:8506/ocr" \
  -F "file=@path/to/image.jpg"
```

Response:
```json
{
  "text": "Extracted OCR text...",
  "status": "success"
}
```

### Batch Processing with Client

The `client.py` script processes all PDFs in the `input_pdfs/` directory:

```bash
# Create input directory and add PDFs
mkdir -p input_pdfs
cp your_files.pdf input_pdfs/

# Run the client
python client.py
```

Output text files will be saved to `output_texts/`

## API Details

### Supported Formats
- **Images**: JPG, PNG, WebP, BMP, GIF
- **PDFs**: Multi-page PDFs (converted to images at 400 DPI)

### Response Fields
- `text`: Extracted OCR text
- `status`: Processing status (success/error)
- Additional metadata as needed

## Environment Variables

- `CUDA_VISIBLE_DEVICES`: GPU device to use (default: 0)
- `HUGGING_FACE_HUB_TOKEN`: Required for model access
- `HF_TOKEN`: Alternative name for Hugging Face token

## Troubleshooting

### GPU Not Detected
- Verify NVIDIA Container Toolkit is installed: `docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu20.04 nvidia-smi`
- Check `CUDA_VISIBLE_DEVICES` in docker-compose.yml

### Out of Memory (OOM)
- Increase `shm_size` in docker-compose.yml
- Reduce batch size or image resolution

### Model Download Issues
- Ensure `HF_TOKEN` is set correctly
- Verify internet connectivity in container
- Check Hugging Face cache directory permissions

## Performance Tips

- **Preprocessing Enabled**: Better OCR accuracy, slower processing (~2-5s per image)
- **Preprocessing Disabled**: Faster processing (~1-2s per image), lower accuracy
- **Batch Processing**: Use client.py for efficient multi-file processing

## License

Refer to LightOnOCR documentation for model licensing terms.

## References

- [LightOnOCR Model](https://huggingface.co/lightonai/LightOnOCR-1B-1025)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Documentation](https://docs.docker.com/)

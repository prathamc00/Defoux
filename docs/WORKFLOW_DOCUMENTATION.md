# DeepGuard RAG-Enhanced Deepfake Detection System

## Complete Workflow Documentation

---

## 1. System Overview

DeepGuard is an AI-powered deepfake detection system that combines **deep learning vision models** with **Retrieval-Augmented Generation (RAG)** for explainable, evidence-based predictions.

### Key Features
- **EfficientNet-based Vision Model** - Trained deepfake classifier
- **RAG Evidence System** - ChromaDB vector store for similarity matching
- **Forensic Analysis** - FFT, color, noise, and compression artifact detection
- **Explainable Predictions** - Human-readable reports with evidence

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DEEPGUARD SYSTEM                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐     ┌─────────────────────────────────────────────────┐  │
│   │   FRONTEND  │     │                    BACKEND                      │  │
│   │   (React)   │────▶│                   (FastAPI)                     │  │
│   │             │     │                                                 │  │
│   │ • Upload    │     │  ┌─────────────────────────────────────────┐   │  │
│   │ • Results   │     │  │          DETECTION PIPELINE             │   │  │
│   │ • History   │     │  │                                         │   │  │
│   └─────────────┘     │  │  Image → Face Detection → Vision Model  │   │  │
│                       │  │              ↓                          │   │  │
│                       │  │    Forensic Analysis ← → RAG Search     │   │  │
│                       │  │              ↓                          │   │  │
│                       │  │      Evidence Fusion Engine             │   │  │
│                       │  │              ↓                          │   │  │
│                       │  │   Explanation Generator → Response      │   │  │
│                       │  └─────────────────────────────────────────┘   │  │
│                       │                                                 │  │
│                       │  ┌──────────────┐    ┌──────────────────────┐  │  │
│                       │  │ ML MODELS    │    │   VECTOR DATABASE    │  │  │
│                       │  │              │    │     (ChromaDB)       │  │  │
│                       │  │ • EfficientNet│   │                      │  │  │
│                       │  │ • MTCNN      │    │ • Evidence Records   │  │  │
│                       │  └──────────────┘    │ • Similarity Search  │  │  │
│                       │                      └──────────────────────┘  │  │
│                       └─────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Detection Workflow (Step-by-Step)

### Step 1: Image Upload
```
User uploads image via Frontend (React)
         │
         ▼
    POST /api/v2/detect
         │
         ▼
    Backend receives file
```
- Supported formats: JPEG, PNG, WebP
- Frontend creates form data and sends to backend

### Step 2: Face Detection & Preprocessing
```
    Raw Image
         │
         ▼
    ┌─────────────┐
    │   MTCNN     │ ← Face detection neural network
    │  Detector   │
    └─────────────┘
         │
         ▼
    Face Cropped (224x224)
    + Normalized
```
- MTCNN (Multi-task Cascaded CNN) extracts face region
- Image resized to 224x224 pixels
- Pixel values normalized using ImageNet statistics

### Step 3: Vision Model Inference
```
    Preprocessed Face
         │
         ▼
    ┌──────────────────┐
    │   EfficientNet   │
    │  (TorchScript)   │
    └──────────────────┘
         │
         ├── Label: "real" or "fake"
         ├── Confidence: 0.0 - 1.0
         └── Embedding: feature vector
```
- EfficientNet-B0 backbone trained on deepfake datasets
- Outputs classification logits and feature embeddings
- Loaded as TorchScript model for production

### Step 4: Forensic Feature Extraction
```
    Original Image
         │
         ▼
    ┌─────────────────────────────────────────────────────────┐
    │                   FORENSIC EXTRACTOR                    │
    ├─────────────────────────────────────────────────────────┤
    │                                                         │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
    │  │ FFT Spectrum│  │Color Channel│  │ Noise Analysis  │ │
    │  │  Analysis   │  │ Deviation   │  │  (Laplacian)    │ │
    │  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘ │
    │         │                │                   │          │
    │         ▼                ▼                   ▼          │
    │   High-freq ratio   RGB deviation      Variance score  │
    │                                                         │
    │  ┌──────────────────────────────────────────────────┐  │
    │  │           Compression Artifact Detection         │  │
    │  │         (8x8 JPEG block boundary analysis)       │  │
    │  └──────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────┘
         │
         ▼
    ForensicFeatures {
      fft_high_freq_ratio: 0.72
      color_deviation_r: 0.35
      color_deviation_g: 0.32
      color_deviation_b: 0.28
      noise_variance: 0.45
      compression_artifact_score: 0.55
    }
```

### Step 5: RAG Evidence Retrieval
```
    Feature Embedding
         │
         ▼
    ┌─────────────────────────────────────────────────────────┐
    │                   VECTOR STORE (ChromaDB)               │
    ├─────────────────────────────────────────────────────────┤
    │                                                         │
    │   Query: embedding vector from current image            │
    │                    │                                    │
    │                    ▼                                    │
    │   ┌─────────────────────────────────────────────────┐  │
    │   │            Similarity Search (k=5)              │  │
    │   │                                                 │  │
    │   │   Indexed Evidence Records:                     │  │
    │   │   • FaceForensics++ dataset                    │  │
    │   │   • DFDC dataset                               │  │
    │   │   • Celeb-DF dataset                           │  │
    │   │   • Custom indexed samples                     │  │
    │   └─────────────────────────────────────────────────┘  │
    │                    │                                    │
    │                    ▼                                    │
    │   Similar Cases:                                        │
    │   [                                                     │
    │     { dataset: "FF++", method: "Deepfakes", sim: 0.89 }│
    │     { dataset: "DFDC", method: "FaceSwap", sim: 0.76 } │
    │     ...                                                │
    │   ]                                                     │
    └─────────────────────────────────────────────────────────┘
```

### Step 6: Evidence Fusion
```
    ┌────────────────┐   ┌────────────────┐   ┌────────────────┐
    │ Vision Score   │   │ RAG Evidence   │   │ Forensic Score │
    │   (60% weight) │   │   (25% weight) │   │   (15% weight) │
    └───────┬────────┘   └───────┬────────┘   └───────┬────────┘
            │                    │                    │
            └────────────────────┼────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │    FUSION ENGINE       │
                    │                        │
                    │  weighted_score =      │
                    │    0.6 × vision +      │
                    │    0.25 × rag +        │
                    │    0.15 × forensic     │
                    └────────────────────────┘
                                 │
                                 ▼
                    FusedPrediction {
                      result: "fake"
                      confidence: 0.87
                      vision_score: 0.92
                      rag_adjustment: +0.05
                    }
```

### Step 7: Explanation Generation
```
    FusedPrediction + SimilarCases + ForensicFeatures
                          │
                          ▼
              ┌───────────────────────────────┐
              │    EXPLANATION GENERATOR      │
              │                               │
              │  • Generate summary text      │
              │  • Explain confidence level   │
              │  • List forensic findings     │
              │  • Summarize similar cases    │
              └───────────────────────────────┘
                          │
                          ▼
              ExplanationReport {
                summary: "Image shows strong indicators 
                         of manipulation"
                confidence_reasoning: "Vision model is 
                         highly confident (92%)..."
                forensic_findings: [
                  "Abnormal high-frequency patterns",
                  "Color channel inconsistencies"
                ]
                similar_cases_summary: "Found 5 similar 
                         cases (4 fake, 1 real)..."
              }
```

### Step 8: Response to Frontend
```
    Final Response JSON:
    {
      "result": "fake",
      "confidence": 0.87,
      "explanation": {
        "summary": "Image shows strong indicators...",
        "confidence_reasoning": "Vision model...",
        "forensic_findings": [...],
        "similar_cases_summary": "...",
        "similar_cases": [...],
        "forensic_evidence": {
          "fft_anomaly": true,
          "fft_score": 0.72,
          "color_anomaly": true,
          ...
        }
      }
    }
```

---

## 4. Component Architecture

```
backend/
├── app/
│   ├── api/
│   │   ├── upload.py          # v1 endpoint
│   │   └── rag_detection.py   # v2 RAG endpoint
│   │
│   ├── models/
│   │   └── image_model.py     # EfficientNet wrapper
│   │
│   ├── services/
│   │   ├── preprocessing/
│   │   │   └── image.py       # MTCNN face detection
│   │   │
│   │   ├── forensics/
│   │   │   └── forensic_extractor.py  # FFT, color, noise
│   │   │
│   │   ├── rag/
│   │   │   ├── vector_store.py     # ChromaDB wrapper
│   │   │   ├── evidence_schema.py  # Data models
│   │   │   ├── fusion_engine.py    # Evidence fusion
│   │   │   └── dataset_indexer.py  # Batch indexing
│   │   │
│   │   └── explainability/
│   │       └── explanation_generator.py
│   │
│   └── main.py                # FastAPI app

frontend/
├── components/
│   ├── UploadZone.tsx         # File upload
│   ├── AnalysisView.tsx       # Loading state
│   └── ResultView.tsx         # Results with tabs
│
├── services/
│   └── backendService.ts      # API calls
│
└── types.ts                   # TypeScript interfaces
```

---

## 5. API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v2/detect` | POST | RAG-enhanced detection with explainability |
| `/api/v2/index/stats` | GET | Get evidence database statistics |
| `/api/v2/index/clear` | DELETE | Clear all indexed evidence |
| `/upload` | POST | Legacy v1 detection (fallback) |

---

## 6. Data Flow Summary

```
                    USER
                      │
                      ▼
              ┌───────────────┐
              │   FRONTEND    │
              │   (React)     │
              └───────┬───────┘
                      │ HTTP POST /api/v2/detect
                      ▼
              ┌───────────────┐
              │   BACKEND     │
              │   (FastAPI)   │
              └───────┬───────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
   ┌─────────┐  ┌─────────┐  ┌─────────────┐
   │ MTCNN   │  │ Vision  │  │  Forensic   │
   │ Face    │  │ Model   │  │  Extractor  │
   │ Detect  │  └────┬────┘  └──────┬──────┘
   └────┬────┘       │              │
        │            ▼              ▼
        │      ┌─────────┐    ┌─────────┐
        │      │ ChromaDB│    │ Fusion  │
        │      │ Search  │───▶│ Engine  │
        │      └─────────┘    └────┬────┘
        │                          │
        └──────────────────────────┤
                                   ▼
                          ┌────────────────┐
                          │  Explanation   │
                          │   Generator    │
                          └───────┬────────┘
                                  │
                                  ▼
                          ┌────────────────┐
                          │    JSON        │
                          │   Response     │
                          └───────┬────────┘
                                  │
                                  ▼
                              FRONTEND
                          (Display Results)
```

---

## 7. Technology Stack

| Layer | Technology |
|-------|------------|
| Frontend | React, TypeScript, Vite, TailwindCSS |
| Backend | FastAPI, Python, Uvicorn |
| ML Framework | PyTorch, TorchVision |
| Face Detection | MTCNN (facenet-pytorch) |
| Vision Model | EfficientNet-B0 (TorchScript) |
| Vector Database | ChromaDB |
| Forensic Analysis | NumPy, SciPy, OpenCV |

---

## 8. Performance Characteristics

| Metric | Value |
|--------|-------|
| Detection Latency | < 2 seconds (GPU) |
| Face Detection | ~50ms |
| Vision Inference | ~100ms |
| Forensic Analysis | ~200ms |
| RAG Search | ~50ms |
| Model Size | ~19MB (TorchScript) |

---

*Document generated for DeepGuard v2.0 with RAG Enhancement*

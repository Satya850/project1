# 🛡️ College Security Surveillance System

Real-time CCTV anomaly detection system for college security with violence classification, face recognition, and incident management.

## Features

| # | Feature | Description |
|---|---------|-------------|
| 1 | **Live Detection** | Real-time webcam/RTSP/video analysis |
| 2 | **Violence Levels** | Low / Medium / High severity classification |
| 3 | **Violence Timestamps** | Automatic start–end time tracking |
| 4 | **Face Recognition** | Identify college persons vs outsiders |
| 5 | **Incident Records** | Searchable history with evidence images/clips |
| 6 | **Analytics Dashboard** | Daily/weekly trends, severity charts, peak hours |
| 7 | **Person Management** | Add students/staff with photos for recognition |
| 8 | **Exam Cheating** | Detect copying during exams |

### Incident Classification (3 Cases)
- **Case 1: College-Only** — All faces matched to database
- **Case 2: College + Outsider** ⚠️ — Mix of known + unknown persons (HIGH PRIORITY)
- **Case 3: Outsiders Only** — No faces matched, flagged as external threat

## Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/Satya850/project1.git
cd project1
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python main.py --mode synthetic --epochs 50 --batch_size 32 --model_type standard --output_dir outputs
```

### 3. Run the Dashboard
```bash
streamlit run dashboard.py
```

Then open **http://localhost:8501** in your browser.

## Project Structure

```
├── dashboard.py              # Multi-page Streamlit dashboard (main UI)
├── main.py                   # Model training & evaluation
├── app.py                    # FastAPI backend
├── config.py                 # Configuration
├── requirements.txt          # Dependencies
│
├── models/                   # Neural network models
│   ├── autoencoder.py        # Convolutional autoencoder
│   ├── detector.py           # Anomaly detector
│   └── violence_classifier.py # Severity + timestamp tracking
│
├── face_recognition_module/  # Face detection & matching
│   ├── face_detector.py      # OpenCV DNN / Haar cascade
│   └── face_matcher.py       # Embedding comparison + 3-case classification
│
├── live/                     # Real-time processing
│   ├── camera.py             # Camera manager (webcam/RTSP/file)
│   └── processor.py          # Full detection pipeline
│
├── incidents/                # Incident management
│   └── incident_manager.py   # Use-case tagging + analytics
│
├── database/                 # SQLite database
│   ├── schema.py             # Tables: persons, cameras, incidents, evidence
│   └── db.py                 # CRUD operations
│
├── data/                     # Dataset loading
├── evaluation/               # Metrics & visualization
└── utils/                    # Logging, GPU, file utilities
```

## Tech Stack

- **Deep Learning**: PyTorch (Convolutional Autoencoder)
- **Computer Vision**: OpenCV (face detection, video processing)
- **Dashboard**: Streamlit + Plotly
- **Database**: SQLite
- **API**: FastAPI + Uvicorn

## Training Results

| Metric | Score |
|--------|-------|
| AUC | 1.0000 |
| Precision | 1.0000 |
| Recall | 1.0000 |
| F1-Score | 1.0000 |
| Inference Speed | 339 FPS |

## License

This project is for educational/research purposes.
Step 1: Install Dependencies
Open a terminal in the project folder and run:

bash
pip install -r requirements.txt
Step 2: Train the Model
bash
python main.py --mode synthetic --epochs 50 --batch_size 32 --model_type standard --output_dir outputs
This trains the anomaly detection model using synthetic data.

Step 3: Run the Dashboard
bash
streamlit run dashboard.py
Then open http://localhost:8501 in your browser.

Summary
Step	Command	What it does
1	pip install -r requirements.txt	Installs all required Python packages
2	python main.py --mode synthetic --epochs 50 --batch_size 32 --model_type standard --output_dir outputs	Trains the anomaly detection model
3	streamlit run dashboard.py	Launches the web dashboard
Note: Make sure you're in the cctv-video-anomaly-detection-main folder when running these commands. You need Python 3.8+ installed on your system.




# 🛡️ Deep Shield Mail

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.x-black?style=for-the-badge&logo=flask&logoColor=white)
![Naive Bayes](https://img.shields.io/badge/Naive%20Bayes-Final%20Model-green?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-EC2%20%7C%20ECR%20%7C%20S3-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-945DD6?style=for-the-badge&logo=dvc&logoColor=white)
![Nginx](https://img.shields.io/badge/Nginx-SSL%20Proxy-009639?style=for-the-badge&logo=nginx&logoColor=white)

<br/>

**A production-grade, end-to-end MLOps project that detects spam & phishing emails in real-time using Naive Bayes + NLP, deployed on AWS with full CI/CD pipeline.**

<br/>

[![Live Demo](https://img.shields.io/badge/🌐_Live_Application-deepshieldmail.duckdns.org-00B4CC?style=for-the-badge&labelColor=1a1a2e)](https://deepshieldmail.duckdns.org)

</div>

---

## 📌 Table of Contents

- [💼 Business Problem](#-business-problem)
- [✨ Features](#-features)
- [📸 Screenshots](#-screenshots)
- [🔬 ML Experiments](#-ml-experiments--algorithm-selection)
- [🧠 Feature Engineering](#-feature-engineering)
- [🔄 MLOps Pipeline](#-mlops-pipeline)
- [🏗️ Architecture](#️-architecture)
- [⚙️ CI/CD & Deployment](#️-cicd--deployment)
- [📁 Project Structure](#-project-structure)
- [🚀 Local Setup](#-local-setup)
- [🐳 Docker](#-docker)
- [🔌 API Endpoints](#-api-endpoints)
- [👨‍💻 Author](#-author)

---

## 💼 Business Problem

Email-based threats — spam, phishing, and malicious content — are among the most common cyberattacks globally. Users unknowingly interact with dangerous emails that lead to:

- **Financial fraud** — phishing emails stealing banking credentials
- **Identity theft** — fake login pages harvesting personal data
- **Malware infection** — malicious attachments and links
- **Privacy breaches** — sensitive data exposed via social engineering

Traditional spam filters fail against modern sophisticated attacks. Users need **intelligent, real-time email threat detection** that connects directly to their inbox and works instantly.

---

## ✨ Features

- 🔐 **Gmail OAuth2 Integration** — Securely connect your Gmail inbox (read-only)
- 🤖 **AI-Powered Detection** — Naive Bayes classifier with TF-IDF + hand-crafted NLP features
- 📊 **Real-time Dashboard** — Safe vs Spam stats, line chart, donut visualization
- 📥 **Inbox Analysis** — Full inbox scan with per-email predictions & confidence scores
- 🕵️ **Threat History** — Complete scan history with SAFE/SPAM badges
- ✍️ **Manual Email Analysis** — Paste any raw email text for instant analysis
- 📈 **MLflow Experiment Tracking** — All 6 algorithms tracked and compared
- 🐳 **Dockerized & Cloud Deployed** — AWS EC2 + ECR + Nginx + Let's Encrypt SSL
- 🔒 **Zero Data Storage** — Emails never stored permanently, privacy first

---

## 📸 Screenshots

### 🏠 Home Page — Stop Email Threats Instantly

![Home Page](./screenshots/home.png)

---

### ⚡ Why Deep Shield? — AI Detection · Real-time Protection · Secure & Private

![Features Section](./screenshots/features.png)

---

### ✍️ Manual Email Analysis — Paste & Analyze Instantly

![Manual Demo](./screenshots/manual_demo.png)

---

### 📊 Full UI Overview — Dashboard · Gmail Login · Spam/Safe Results · Threat History

![Full UI](./deeplshaild.png)

---

## 🔬 ML Experiments — Algorithm Selection

All experiments tracked with **MLflow**. 6 algorithms evaluated on the same dataset:

| Algorithm | Accuracy | Precision | Recall | F1 Score |
|-----------|----------|-----------|--------|----------|
| **Naive Bayes** ✅ | **97.8%** | **98.1%** | **97.4%** | **97.7%** |
| Logistic Regression | 96.2% | 96.8% | 95.7% | 96.2% |
| SVM (SVC) | 96.5% | 97.0% | 95.9% | 96.4% |
| Random Forest | 95.9% | 96.1% | 95.3% | 95.7% |
| XGBoost | 95.4% | 95.8% | 94.9% | 95.3% |
| Decision Tree | 93.1% | 93.4% | 92.6% | 93.0% |

> ✅ **Naive Bayes selected as final algorithm** — highest F1 score, fastest inference, and best suited for high-dimensional TF-IDF sparse text features. Probabilistic nature provides natural confidence scores.

### Experiment Notebooks

| Notebook | Algorithm |
|----------|-----------|
| `exp-Naive_Bayes.ipynb` | ✅ Final Model |
| `exp-Logistic_Regression.ipynb` | Baseline |
| `exp-RandomForest.ipynb` | Ensemble |
| `exp-SVC.ipynb` | Kernel-based |
| `exp-XGBoost.ipynb` | Gradient Boosting |
| `exp-DecisionTree.ipynb` | Tree-based |

---

## 🧠 Feature Engineering

```
Raw Email Text
      │
      ├── EmailParser           → From, To, Subject, Date, Body
      │
      ├── MetaFeatureExtractor  → 12 hand-crafted features
      │     ├── has_suspicious_links
      │     ├── sender_domain_match
      │     ├── exclamation_count
      │     ├── uppercase_ratio
      │     ├── has_html_tags
      │     ├── url_count
      │     └── ... (6 more)
      │
      └── BodyFeatureExtractor  → TF-IDF (30,000 features)
                                           │
                           Final Matrix: (1, 30,012)
                                           │
                               Naive Bayes Prediction
                                           │
                               SAFE (0) / SPAM (1) + Probability Score
```

---

## 🔄 MLOps Pipeline

```
Data Ingestion → Data Validation → Data Transformation → Model Training → Model Evaluation → Model Pusher
      │                 │                  │                    │                 │                │
  CSV Load         Schema Check        TF-IDF Fit          Naive Bayes        Metrics          S3/ECR
  Train/Test       report.yaml        preprocessing.pkl     model.pkl          Compare           Push
```

### Pipeline Components

| Component | File | Description |
|-----------|------|-------------|
| Data Ingestion | `data_ingestion.py` | Load raw CSV, train/test split |
| Data Validation | `data_validation.py` | Schema checks, drift detection |
| Data Transformation | `data_transformation.py` | TF-IDF + hand-crafted features |
| Model Trainer | `model_trainer.py` | Train Naive Bayes, save model.pkl |
| Model Evaluation | `model_evaluation.py` | Compare new model vs production |
| Model Pusher | `model_pusher.py` | Push best model to S3/ECR |

### DVC Pipeline

```bash
dvc repro   # Run full pipeline
dvc dag     # View pipeline DAG
```

---

## 🏗️ Architecture

```
User Browser
     │
     ▼
Nginx (SSL/TLS — Let's Encrypt)
deepshieldmail.duckdns.org
     │
     ▼
Gunicorn (Flask App) ── Port 8000
     │
     ├── /auth/gmail  ── Gmail OAuth2 ── Google API
     │
     ├── /api/emails  ── Fetch Inbox (Gmail API, limit=50)
     │
     └── /api/scan    ── Prediction Pipeline
                              │
                              ├── EmailParser
                              ├── MetaFeatureExtractor  (12 features)
                              ├── BodyFeatureExtractor  (TF-IDF 30,000)
                              └── Naive Bayes → SAFE / SPAM + Score
```

---

## ⚙️ CI/CD & Deployment

```
Local Code
    │
    ▼
git push → GitHub
    │
    ▼
Docker Build (Local)
    │
    ▼
AWS ECR (Image Push)
343980058839.dkr.ecr.us-east-1.amazonaws.com/deep-shield-mail:latest
    │
    ▼
AWS EC2 (Ubuntu) — docker pull + run
    │
    ▼
Nginx Reverse Proxy (Port 443 SSL)
    │
    ▼
deepshieldmail.duckdns.org ✅ Live
```

### Infrastructure

| Component | Technology |
|-----------|-----------|
| Cloud | AWS EC2 (Ubuntu 24) |
| Container Registry | AWS ECR |
| Container | Docker |
| Web Server | Nginx + Let's Encrypt SSL |
| App Server | Gunicorn (2 workers, 2 threads, 300s timeout) |
| DNS | DuckDNS |
| Experiment Tracking | MLflow |
| Data Versioning | DVC |

---

## 📁 Project Structure

```
Deep-Shield-Mail/
│
├── serving/
│   └── api/
│       └── app.py                   ← Flask app (OAuth, scan, predict APIs)
│
├── src/
│   ├── pipeline/
│   │   ├── prediction_pipeline.py   ← EmailParser + Features + Naive Bayes
│   │   └── training_pipeline.py     ← End-to-end MLOps training
│   ├── components/                  ← Pipeline stage implementations
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   ├── model_evaluation.py
│   │   └── model_pusher.py
│   ├── entity/                      ← Config & artifact dataclasses
│   └── utils/                       ← Logger, exception handler
│
├── templates/                       ← Jinja2 HTML templates
├── static/
│   ├── css/style.css
│   └── js/main.js
│
├── notebooks/                       ← EDA + all 6 experiment notebooks
│   ├── EDA.ipynb
│   ├── exp-Naive_Bayes.ipynb        ← ✅ Final model
│   ├── exp-Logistic_Regression.ipynb
│   ├── exp-RandomForest.ipynb
│   ├── exp-SVC.ipynb
│   ├── exp-XGBoost.ipynb
│   └── exp-DecisionTree.ipynb
│
├── config/schema.yaml
├── params.yaml
├── dvc.yaml                         ← DVC pipeline definition
├── dvc.lock
├── Dockerfile
└── requirements.txt
```

---

## 🚀 Local Setup

```bash
# 1. Clone repo
git clone https://github.com/kaushik-chariya/Deep-Shield-Mail.git
cd Deep-Shield-Mail

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variables
export GOOGLE_CLIENT_ID=your_client_id
export GOOGLE_CLIENT_SECRET=your_client_secret
export REDIRECT_URI=http://localhost:8000/auth/gmail/callback
export FLASK_SECRET_KEY=your_secret_key

# 5. Pull data & run pipeline
dvc pull
dvc repro

# 6. Run the app
gunicorn --workers 2 --threads 2 --timeout 300 \
  --bind 0.0.0.0:8000 serving.api.app:app
```

---

## 🐳 Docker

```bash
# Build
docker build -t deep-shield-mail .

# Run
docker run -d \
  --name deep-shield-mail \
  --restart unless-stopped \
  -p 8000:8000 \
  -e GOOGLE_CLIENT_ID=your_client_id \
  -e GOOGLE_CLIENT_SECRET=your_client_secret \
  -e REDIRECT_URI=https://yourdomain.com/auth/gmail/callback \
  -e FLASK_SECRET_KEY=your_secret_key \
  -e GUNICORN_TIMEOUT=300 \
  deep-shield-mail \
  gunicorn --workers 2 --threads 2 --timeout 300 \
    --bind 0.0.0.0:8000 serving.api.app:app
```

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Landing page |
| `/auth/gmail` | GET | Gmail OAuth2 login |
| `/auth/gmail/callback` | GET | OAuth callback |
| `/inbox` | GET | Dashboard |
| `/inbox/emails` | GET | Inbox Analysis page |
| `/inbox/history` | GET | Threat History page |
| `/email/<id>` | GET | Email detail view |
| `/api/emails?limit=50` | GET | Fetch Gmail inbox |
| `/api/scan` | POST | Scan & classify emails |
| `/api/predict` | POST | Predict single email |
| `/api/predict/manual` | POST | Manual email analysis |
| `/health` | GET | Health check |
| `/logout` | GET | Logout & clear session |

---

## 🔐 Security & Privacy

- Gmail OAuth2 — **read-only** scope — no send/delete permissions
- **Zero email storage** — scan results cleared on logout
- SSL/TLS via **Let's Encrypt**
- Session-based auth, no passwords stored
- File-based session storage — **no cookie overflow**

---

## 👨‍💻 Author

<div align="center">

**Kaushik Chariya**

[![GitHub](https://img.shields.io/badge/GitHub-kaushik--chariya-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/kaushik-chariya)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Kaushik_Chariya-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kaushik-chariya)
[![Live App](https://img.shields.io/badge/🌐_Live_App-deepshieldmail.duckdns.org-00B4CC?style=for-the-badge)](https://deepshieldmail.duckdns.org)

</div>

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with ❤️ | **MLOps · NLP · Naive Bayes · Flask · Docker · AWS · MLflow · DVC**

</div>
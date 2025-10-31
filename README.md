# Predictive Modeling for User Engagement

A comprehensive machine learning platform that predicts user engagement patterns and provides actionable insights through an interactive dashboard. This project combines Python-based ML pipelines with a modern Next.js web interface to deliver real-time analytics and predictions.

![Dashboard Preview](https://img.shields.io/badge/Status-Active-success)
![License](https://img.shields.io/badge/License-MIT-blue)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Next.js](https://img.shields.io/badge/Next.js-16.0-black)
![React](https://img.shields.io/badge/React-19.2-blue)

## 🎯 Project Overview

This platform leverages machine learning to forecast user engagement behavior, helping product teams make data-driven decisions. The system processes user interaction data, engineers relevant features, trains multiple ML models, and presents results through an intuitive web dashboard.

### Key Objectives
- **Predict User Engagement**: Achieve ≥85% accuracy in predicting user engagement
- **Optimize Data Processing**: Reduce data preparation time by 40%
- **Drive Feature Adoption**: Enable 20% increase in feature adoption through insights
- **Real-time Analytics**: Provide interactive visualizations and model performance metrics

## ✨ Features

### Machine Learning Pipeline
- **Synthetic Data Generation**: Realistic user behavior simulation with 10,000+ samples
- **Feature Engineering**: 20+ engineered features including interaction terms and normalized metrics
- **Multiple Models**: Logistic Regression, Random Forest, and Gradient Boosting classifiers
- **Comprehensive Evaluation**: ROC curves, confusion matrices, feature importance analysis
- **Model Persistence**: Trained models saved for production deployment

### Interactive Dashboard
- **Real-time Metrics**: Live display of model accuracy, F1-scores, and ROC-AUC
- **Visual Analytics**: Interactive charts powered by Recharts
- **Performance Tracking**: Model comparison and evaluation metrics
- **Feature Analysis**: Top feature importance visualization
- **Responsive Design**: Mobile-first UI with dark mode support

### Analytics & Insights
- Device type distribution analysis
- User segment classification
- Traffic source attribution
- Session duration patterns
- Click-through rate optimization

## 🛠️ Tech Stack

### Backend & ML Pipeline
- **Python 3.8+**: Core language for ML pipeline
- **pandas & NumPy**: Data manipulation and numerical computing
- **scikit-learn**: Machine learning models and evaluation
- **Matplotlib & Seaborn**: Statistical visualizations
- **PyYAML**: Configuration management

### Frontend & Visualization
- **Next.js 16.0**: React framework with App Router
- **React 19.2**: UI library
- **TypeScript**: Type-safe development
- **Tailwind CSS 4.x**: Utility-first styling
- **Radix UI**: Accessible component primitives
- **Recharts**: Data visualization library
- **Lucide React**: Icon library

### UI Components
- **shadcn/ui**: High-quality React components
- **React Hook Form**: Form state management
- **Zod**: Schema validation
- **next-themes**: Dark mode support

## 📋 Prerequisites

- **Python**: 3.8 or higher
- **Node.js**: 18.x or higher
- **pnpm**: Package manager (or npm/yarn)
- **Git**: Version control

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/johaankjis/Predictive-Modeling-for-User-Engagement.git
cd Predictive-Modeling-for-User-Engagement
```

### 2. Python Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install Python dependencies
pip install pandas numpy scikit-learn matplotlib seaborn pyyaml
```

### 3. Run ML Pipeline

Execute the ML pipeline scripts in sequence:

```bash
# Step 1: Generate and clean data
python scripts/data_ingestion.py

# Step 2: Engineer features
python scripts/feature_engineering.py

# Step 3: Train models
python scripts/model_training.py

# Step 4: Evaluate models
python scripts/model_evaluation.py
```

This will:
- Generate synthetic user engagement data
- Create engineered features
- Train three ML models (LR, RF, GB)
- Generate evaluation metrics and visualizations
- Save models and results to `models/` and `public/` directories

### 4. Frontend Setup

```bash
# Install Node.js dependencies
pnpm install

# Run development server
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser to view the dashboard.

### 5. Build for Production

```bash
# Build the Next.js application
pnpm build

# Start production server
pnpm start
```

## 📁 Project Structure

```
Predictive-Modeling-for-User-Engagement/
├── app/                          # Next.js App Router
│   ├── page.tsx                  # Main dashboard page
│   ├── layout.tsx                # Root layout
│   └── globals.css               # Global styles
├── components/                   # React components
│   ├── dashboard-client.tsx      # Main dashboard component
│   ├── theme-provider.tsx        # Theme management
│   └── ui/                       # shadcn/ui components
├── scripts/                      # Python ML pipeline
│   ├── data_ingestion.py         # Data generation & cleaning
│   ├── feature_engineering.py    # Feature creation
│   ├── model_training.py         # Model training
│   └── model_evaluation.py       # Model evaluation
├── src/                          # Configuration
│   └── config.yaml               # ML pipeline config
├── public/                       # Static assets
│   ├── plots/                    # Generated visualizations
│   ├── evaluation_report.json    # Model metrics
│   ├── feature_importance.json   # Feature analysis
│   └── data_summary.json         # Dataset statistics
├── models/                       # Trained models (generated after running pipeline)
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── gradient_boosting.pkl
│   ├── scaler.pkl
│   └── training_metadata.json
├── data/                         # Generated datasets (created by ML pipeline)
│   ├── user_engagement_data.csv
│   └── engineered_features.csv
├── hooks/                        # React custom hooks
├── lib/                          # Utility functions
├── styles/                       # Additional styles
├── package.json                  # Node.js dependencies
├── tsconfig.json                 # TypeScript config
├── next.config.mjs               # Next.js config
├── postcss.config.mjs            # PostCSS config
└── components.json               # shadcn/ui config
```

## 🔬 ML Pipeline Details

### Data Ingestion (`data_ingestion.py`)
- Generates 10,000 synthetic user records
- Simulates realistic user behavior patterns
- Features: device type, user segment, traffic source, session metrics
- Engagement rate: ~60% (balanced dataset)
- Outputs: `data/user_engagement_data.csv`

### Feature Engineering (`feature_engineering.py`)
- **Interaction Features**: engagement_intensity, click_efficiency, user_maturity, session_quality
- **Categorical Encoding**: One-hot encoding for device, segment, and traffic source
- **Binned Features**: Session duration bins, user age bins
- **Normalized Features**: Min-max normalization for key metrics
- Outputs: `data/engineered_features.csv`

### Model Training (`model_training.py`)
- **Train/Test Split**: 80/20 stratified split
- **Feature Scaling**: StandardScaler for normalization
- **Models Trained**:
  - Logistic Regression: Baseline linear model
  - Random Forest: 100 estimators, max_depth=10
  - Gradient Boosting: 100 estimators, learning_rate=0.1
- Outputs: Trained models in `models/` directory

### Model Evaluation (`model_evaluation.py`)
- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Visualizations**:
  - Confusion matrices for all models
  - ROC curves comparison
  - Feature importance analysis
  - Metrics comparison bar chart
- Target: ≥85% accuracy achieved
- Outputs: Evaluation report and plots in `public/`

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | ~85% | ~84% | ~87% | ~85% | ~92% |
| Random Forest | ~90% | ~89% | ~91% | ~90% | ~95% |
| Gradient Boosting | ~91% | ~90% | ~92% | ~91% | ~96% |

*Note: Exact values depend on random seed and data generation*

## 🎨 Dashboard Features

### MVP Success Metrics
- Real-time model accuracy tracking
- Data preparation time reduction metrics
- Feature adoption increase indicators
- Models trained counter

### Interactive Components
- **Model Performance Cards**: Live metrics for each model
- **ROC Curves**: Interactive comparison of all models
- **Confusion Matrices**: Visual representation of predictions
- **Feature Importance**: Top contributing features
- **Data Distribution**: Device, segment, and traffic source breakdown

## 🔧 Configuration

Edit `src/config.yaml` to customize:

```yaml
data:
  sample_size: 10000        # Number of synthetic records
  train_test_split: 0.8     # Train/test ratio
  random_state: 42          # Reproducibility seed

models:
  logistic_regression:
    max_iter: 1000
    random_state: 42
  random_forest:
    n_estimators: 100       # Number of trees
    max_depth: 10           # Tree depth
  gradient_boosting:
    n_estimators: 100
    learning_rate: 0.1
    max_depth: 5

evaluation:
  target_accuracy: 0.85     # Success threshold
```

## 🧪 Development

### Running Linter

```bash
# Lint Next.js code
pnpm lint
```

### Building for Production

```bash
# Build Next.js application
pnpm build
```

### Development Mode

```bash
# Run with hot reload
pnpm dev
```

## 📈 Future Enhancements

- [ ] Real-time data ingestion from production sources
- [ ] A/B testing integration
- [ ] Automated model retraining pipeline
- [ ] Advanced feature engineering (time-series, NLP)
- [ ] Model explainability (SHAP, LIME)
- [ ] Multi-model ensemble predictions
- [ ] API endpoints for model inference
- [ ] User segmentation clustering
- [ ] Churn prediction model
- [ ] Recommendation system integration

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- **Johan** - [johaankjis](https://github.com/johaankjis)

## 🙏 Acknowledgments

- Built with [v0](https://v0.dev) by Vercel
- UI components from [shadcn/ui](https://ui.shadcn.com)
- ML framework: [scikit-learn](https://scikit-learn.org)
- Icons by [Lucide](https://lucide.dev)

## 📞 Support

For questions or issues, please open an issue on GitHub or contact the maintainers.

---

**Made with ❤️ for data-driven product decisions**

# 🎮 Minecraft Education Analytics Dashboard

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://4n4umjzbkvgwt66pshlnj6.streamlit.app/)

**A comprehensive educational data science platform for analyzing game-based learning patterns in Minecraft Education Edition**

[🚀 Live Demo](https://4n4umjzbkvgwt66pshlnj6.streamlit.app/) • [Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Documentation](#-documentation) • [Contributing](#-contributing)

<br>

### 🎯 **[Launch the Dashboard →](https://4n4umjzbkvgwt66pshlnj6.streamlit.app/)**
*Deployed on Streamlit Cloud - No installation required!*

</div>

---

## 📋 Overview

The Minecraft Education Analytics Dashboard is an advanced data science portfolio project that demonstrates the application of statistical analysis, machine learning, and data visualization techniques to educational gaming environments. This platform provides educators and researchers with actionable insights into student learning patterns, engagement metrics, and collaborative behaviors within Minecraft Education Edition.

### 🌟 Try It Live!

<div align="center">
  
**[🚀 Launch Live Dashboard](https://4n4umjzbkvgwt66pshlnj6.streamlit.app/)**

No installation required - explore all features with sample data instantly!

</div>

### 🎯 Key Objectives

- **Analyze** student engagement and learning progression in game-based environments
- **Predict** educational outcomes using machine learning models
- **Identify** at-risk students for early intervention
- **Visualize** complex educational data through interactive 3D visualizations
- **Support** data-driven decision making in educational settings

## ✨ Features

### 📊 Data Management
- **Synthetic Data Generation**: Create realistic educational datasets for testing
- **Real Data Upload**: Support for CSV, Excel, and JSON formats
- **Smart Data Validation**: Automatic validation and error handling
- **Template Downloads**: Pre-formatted templates for easy data preparation

### 📈 Statistical Analysis
- **T-Tests & ANOVA**: Compare learning outcomes between student groups
- **Effect Size Calculations**: Measure practical significance of findings
- **Correlation Analysis**: Identify relationships between variables
- **Assumption Testing**: Validate statistical test requirements

### 🤖 Machine Learning
- **Predictive Models**: Linear, Ridge, Random Forest, and Gradient Boosting
- **Feature Importance**: Identify key factors affecting student outcomes
- **At-Risk Detection**: Early warning system for struggling students
- **Model Comparison**: Automated evaluation of multiple algorithms

### 🎨 Visualizations
- **3D World Maps**: Interactive visualization of student movements
- **Collaboration Networks**: Social learning pattern analysis
- **Learning Progression Charts**: Track skill development over time
- **Engagement Heatmaps**: Identify activity patterns and zones

### 📚 Educational Insights
- **Research-Based Metrics**: Grounded in educational data mining research
- **Actionable Dashboards**: Following SpringerOpen design guidelines
- **Intervention Recommendations**: Evidence-based suggestions for educators
- **Export Capabilities**: Generate reports and download processed data

## 🚀 Live Demo

### 🌐 Try it Now!
**[Launch the Live Dashboard →](https://4n4umjzbkvgwt66pshlnj6.streamlit.app/)**

Experience the full functionality of the Minecraft Education Analytics Dashboard with sample data. No installation required!

### Screenshots

#### Main Dashboard
![Dashboard Overview](https://via.placeholder.com/800x400?text=Dashboard+Overview)
*Interactive overview showing key performance indicators and student metrics*

#### Statistical Analysis
![Statistical Analysis](https://via.placeholder.com/800x400?text=Statistical+Analysis)
*Comprehensive statistical testing with effect sizes and interpretations*

#### 3D Visualizations
![3D World View](https://via.placeholder.com/800x400?text=3D+World+Visualization)
*Interactive 3D visualization of student movements and building activities*

## 🛠️ Installation

### Quick Access - No Installation Required!

**[🌐 Use the Live Dashboard](https://4n4umjzbkvgwt66pshlnj6.streamlit.app/)** - Start analyzing data immediately in your browser!

### Local Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Git (for cloning the repository)

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/minecraft-education-dashboard.git
cd minecraft-education-dashboard
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run app.py
```

5. **Open your browser**
Navigate to `http://localhost:8501`

### Docker Installation

```dockerfile
# Build the Docker image
docker build -t minecraft-edu-dashboard .

# Run the container
docker run -p 8501:8501 minecraft-edu-dashboard
```

### Deploy Your Own Instance on Streamlit Cloud

1. **Fork this repository** to your GitHub account
2. **Sign in** to [share.streamlit.io](https://share.streamlit.io)
3. **Deploy** your forked repository:
   - Click "New app"
   - Select your repository
   - Set branch to `main`
   - Set main file path to `app.py`
   - Click "Deploy"
4. **Your app** will be live in minutes!

**Current Live Instance: [https://4n4umjzbkvgwt66pshlnj6.streamlit.app/](https://4n4umjzbkvgwt66pshlnj6.streamlit.app/)**

## 📖 Usage

### Quick Start with Live Demo

The easiest way to explore the dashboard is through the **[live deployment](https://4n4umjzbkvgwt66pshlnj6.streamlit.app/)**:

1. **Visit** [https://4n4umjzbkvgwt66pshlnj6.streamlit.app/](https://4n4umjzbkvgwt66pshlnj6.streamlit.app/)
2. **Generate** synthetic data or upload your own
3. **Explore** all features instantly in your browser

### Getting Started with Local Installation

1. **Choose Data Source**
   - Generate synthetic data for testing
   - Upload your own educational data
   - Use sample data to explore features

2. **Configure Parameters**
   - Set number of students (20-200)
   - Choose simulation duration (7-90 days)
   - Select random seed for reproducibility

3. **Explore Analytics**
   - View overview metrics and KPIs
   - Run statistical analyses
   - Train predictive models
   - Visualize data in 3D

### Data Upload Guide

#### Required Data Format

The dashboard accepts data in the following formats:

**Students Dataset**
| Column | Type | Description |
|--------|------|-------------|
| student_id | string | Unique student identifier |
| grade_level | integer | Grade level (6-8) |
| learning_style | string | visual/kinesthetic/auditory |
| collaboration_preference | string | solo/pairs/groups |

**Learning Analytics Dataset**
| Column | Type | Description |
|--------|------|-------------|
| student_id | string | Student identifier |
| engagement_score | float | Engagement level (0-1) |
| quest_completion_rate | float | Quest completion (0-1) |
| days_active | integer | Number of active days |
| skill_progression | float | Skill development (0-1) |

**Movements Dataset**
| Column | Type | Description |
|--------|------|-------------|
| student_id | string | Student identifier |
| timestamp | datetime | Movement timestamp |
| x, y, z | float | 3D coordinates |
| zone | string | Current zone name |

[View complete data specifications →](docs/data_format.md)

## 🏗️ Project Structure

```
minecraft-education-dashboard/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── README.md                   # Project documentation
│
├── src/                        # Source code modules
│   ├── __init__.py
│   ├── data_generation/        # Data simulation
│   │   ├── __init__.py
│   │   └── simulator.py
│   ├── analysis/               # Statistical analysis
│   │   ├── __init__.py
│   │   ├── statistical.py
│   │   └── time_series.py
│   ├── visualization/          # Plotting utilities
│   │   ├── __init__.py
│   │   └── plots.py
│   └── utils/                  # Helper functions
│       ├── __init__.py
│       └── helpers.py
│
├── pages/                      # Streamlit pages
│   ├── 1_📊_Overview.py
│   ├── 2_📈_Statistical_Analysis.py
│   ├── 3_🤖_Predictive_Models.py
│   ├── 4_🎮_3D_World_View.py
│   └── 5_📚_Documentation.py
│
├── data/                       # Data directory
│   ├── simulated/             # Generated data
│   └── templates/             # Template files
│
├── tests/                      # Unit tests
│   └── test_app.py
│
└── docs/                       # Documentation
    ├── data_format.md
    ├── api_reference.md
    └── research_papers.md
```

## 🔬 Technical Architecture

### Core Technologies

- **Frontend**: Streamlit 1.28+
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, NetworkX
- **Statistical Analysis**: SciPy, Statsmodels
- **Machine Learning**: Scikit-learn
- **File Handling**: OpenPyXL, XlsxWriter

### Key Components

1. **Data Generation Module**
   - Realistic student profile generation
   - Movement pattern simulation
   - Quest completion modeling
   - Collaboration network creation

2. **Statistical Analysis Engine**
   - Parametric and non-parametric tests
   - Effect size calculations
   - Power analysis
   - Assumption validation

3. **Machine Learning Pipeline**
   - Feature engineering
   - Model training and evaluation
   - Cross-validation
   - Hyperparameter tuning

4. **Visualization System**
   - Interactive 3D plots
   - Real-time dashboard updates
   - Network graph rendering
   - Export-ready charts

## 📊 Research Foundation

This project is based on cutting-edge research in educational data mining:

- **Changepoint Detection**: Based on EDM 2024 conference paper on behavioral pattern shifts
- **Dashboard Design**: Follows SpringerOpen 2021 guidelines for actionable learning analytics
- **Statistical Methods**: Implements best practices from educational psychology research
- **Minecraft in Education**: Systematic review of game-based learning effectiveness (2025)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### How to Contribute

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run linting
flake8 src/

# Format code
black src/
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Minecraft Education Team** - For inspiring educational gaming
- **Streamlit Community** - For excellent framework and support
- **Educational Data Mining Society** - For research foundations
- **Open Source Contributors** - For amazing libraries and tools

## 📧 Contact

**Project Maintainer**: [Your Name]  
**Email**: your.email@example.com  
**LinkedIn**: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)  
**Portfolio**: [yourportfolio.com](https://yourportfolio.com)

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/minecraft-education-dashboard&type=Date)](https://star-history.com/#yourusername/minecraft-education-dashboard&Date)

## 📈 Project Status

### 🟢 Live Deployment
**The app is live and running at: [https://4n4umjzbkvgwt66pshlnj6.streamlit.app/](https://4n4umjzbkvgwt66pshlnj6.streamlit.app/)**

### ✅ Completed Features
- [x] Core dashboard implementation
- [x] Statistical analysis module
- [x] Machine learning predictions
- [x] 3D visualizations
- [x] Data upload functionality
- [x] Streamlit Cloud deployment

### 🚧 Upcoming Features
- [ ] Real-time data streaming
- [ ] Advanced deep learning models
- [ ] Multi-language support
- [ ] Mobile responsive design
- [ ] API endpoints

---

<div align="center">

**Made with ❤️ for Education**

**[🚀 Try the Live Demo](https://4n4umjzbkvgwt66pshlnj6.streamlit.app/)**

If you find this project helpful, please consider giving it a ⭐!

[Report Bug](https://github.com/yourusername/minecraft-education-dashboard/issues) • [Request Feature](https://github.com/yourusername/minecraft-education-dashboard/issues) • [Live Dashboard](https://4n4umjzbkvgwt66pshlnj6.streamlit.app/)

</div>

# 🎓 GradeGraph - Student Performance Analyzer

A comprehensive Streamlit-based web application for analyzing student performance data from Excel files. Built for educational institutions to track, analyze, and provide actionable insights on student academic performance.

## ✨ Features

### 📊 Core Functionality
- **Excel File Processing**: Upload and process student performance data from Excel files
- **Student Classification**: Automatically categorize students as Bright, Average, or Weak based on performance metrics
- **Performance Analytics**: Comprehensive analysis of academic and practical performance
- **Subject-wise Analysis**: Detailed breakdown of performance by individual subjects
- **Threshold-based Recommendations**: Customizable threshold analysis for targeted interventions

### 📈 Dashboard & Visualization
- **Interactive Dashboard**: Real-time performance metrics and statistics
- **Visual Charts**: Pie charts, histograms, and bar graphs for data visualization
- **Student Search**: Quick lookup of individual student performance
- **Export Options**: Download processed data and reports in CSV format

### 🎯 Advanced Analytics
- **Dynamic Recommendations**: AI-powered suggestions based on performance patterns
- **Subject Difficulty Analysis**: Identify challenging subjects requiring attention
- **Performance Insights**: Actionable recommendations for academic improvement
- **Comprehensive Reporting**: Detailed analysis reports for administrative use

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/gradegraphs.git
   cd gradegraphs
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   
   **Windows:**
   ```bash
   venv\Scripts\activate
   ```
   
   **macOS/Linux:**
   ```bash
   source venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r gg/requirements.txt
   ```

5. **Run the application**
   ```bash
   streamlit run gg/app.py
   ```

6. **Access the application**
   Open your browser and navigate to `http://localhost:8501`

## 📁 Project Structure

```
gradegraphs/
├── gg/                          # Main application directory
│   ├── app.py                   # Main Streamlit application
│   ├── requirements.txt         # Python dependencies
│   └── services/                # Service modules
│       ├── __init__.py
│       ├── processing.py        # Excel file processing logic
│       ├── recommendations.py   # Recommendation algorithms
│       ├── students.py          # Student data management
│       ├── subjects.py          # Subject analysis functions
│       └── utils.py             # Utility functions
├── .gitignore                   # Git ignore rules
└── README.md                    # Project documentation
```

## 📋 Usage Guide

### 1. Upload & Process Data
- Navigate to the "📤 Upload & Process" page
- Upload an Excel file containing student performance data
- Ensure the file starts with 'SR.No.' row for proper processing
- View processed statistics and data preview

### 2. Dashboard Analysis
- Access the "📈 Dashboard" page for overview metrics
- View student category distribution and performance charts
- Download comprehensive reports

### 3. Individual Student Analysis
- Use "👥 Student Search" to find specific students
- View detailed performance breakdown by subject
- Analyze academic and practical performance metrics

### 4. Subject-wise Analysis
- Select "📊 Subject Analysis" for detailed subject breakdown
- View performance distributions and top/bottom performers
- Analyze theory vs practical performance

### 5. Performance Insights
- Access "🎯 Performance Insights" for actionable recommendations
- View subject difficulty analysis
- Generate comprehensive reports

### 6. Threshold-based Recommendations
- Use "🎯 Threshold-Based Recommendations" for custom analysis
- Set specific thresholds for targeted interventions
- Generate detailed performance reports

## 🔧 Configuration

### Excel File Format Requirements
- File must start with 'SR.No.' row
- Include student information columns (Name, Roll No, etc.)
- Subject columns should follow naming conventions:
  - Theory: `Subject ISE (25)`, `Subject MSE (25)`, `Subject ESE (60)`
  - Practical: `Subject PRACTICAL (25)` or `Subject PR (25)`
  - Max marks should be specified in parentheses

### Performance Calculation Logic
- **Academic Performance %**: `(Total Obtained Marks / Total Maximum Marks) × 100`
- **Practical Performance %**: Calculated from practical assessment columns only
- **Student Classification**:
  - **Bright**: Academic Performance ≥ 80% OR Coding Expertise = 'Advanced'
  - **Average**: Academic Performance 60-79% OR Coding Expertise = 'Intermediate'
  - **Weak**: Academic Performance < 60% AND Coding Expertise = 'Beginner'

## 🛠️ Technical Details

### Dependencies
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive data visualization
- **NumPy**: Numerical computing
- **OpenPyXL**: Excel file processing

### Architecture
- **Clean Architecture**: Modular service-based design
- **MVVM Pattern**: Separation of concerns between data, logic, and presentation
- **Service Layer**: Dedicated modules for processing, recommendations, and utilities

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Kshitij** - *Initial work*
- **Shweta** - *Development and enhancement*

Built for the **COMPUTER DEPARTMENT** educational institution.

## 📞 Support

For support, email your-email@example.com or create an issue in the repository.

## 🔮 Future Enhancements

- [ ] Database integration for persistent data storage
- [ ] User authentication and role-based access
- [ ] Advanced machine learning recommendations
- [ ] Mobile-responsive design improvements
- [ ] Real-time collaboration features
- [ ] Integration with Learning Management Systems (LMS)

---

**GradeGraph** - Empowering education through data-driven insights! 🎓📊

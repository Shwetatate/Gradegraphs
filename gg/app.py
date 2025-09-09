import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from services import (
    process_excel_file,
    get_subject_list,
    get_student_performance,
    get_dynamic_subject_recommendations,
    get_threshold_based_recommendations,
    get_subject_exam_types,
    extract_max_marks_from_header,
)
import re
import numpy as np

def _is_practical_col(col_name):
    if not col_name or pd.isna(col_name):
        return False
    col_upper = str(col_name).upper()
    if 'TW' in col_upper and 'PR' not in col_upper:
        return False
    return ('PRACTICAL' in col_upper) or (re.search(r'\bPR\b', col_upper) is not None)

def _row_percentage(row, cols):
    obtained_total = 0.0
    max_total = 0.0
    for c in cols:
        val = row.get(c, np.nan)
        try:
            val = float(val)
        except Exception:
            val = np.nan
        if pd.notna(val):
            obtained_total += val
            max_total += extract_max_marks_from_header(c)
    return (obtained_total / max_total) * 100.0 if max_total > 0 else np.nan

def _get_max_marks_for_subject(subject_name, exam_type=None):
    """
    Get maximum marks for a subject and exam type
    """
    if exam_type:
        exam_upper = exam_type.upper()
        if 'ESE' in exam_upper:
            return 60
        elif 'ISE' in exam_upper or 'MSE' in exam_upper:
            return 25
        elif 'PRACTICAL' in exam_upper or 'PR' in exam_upper:
            return 25
        elif 'TW' in exam_upper:
            return 50
        else:
            return 100
    else:
        # Default max marks for subject (sum of all exam types)
        return 100

# Configure Streamlit page
st.set_page_config(
    page_title="GradeGraph - Student Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling with improved recommendation colors
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-msg {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    .recommendation-card {
        background: #ffffff;
        color: #000000;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        border: 1px solid #000000;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
        transition: transform 0.3s ease;
    }
    .recommendation-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
    }
    .recommendation-title {
        font-size: 1.2em;
        font-weight: bold;
        color: #000000;
        margin-bottom: 10px;
    }
    .recommendation-description {
        font-size: 1em;
        color: #222222;
        margin-bottom: 12px;
        line-height: 1.4;
    }
    .recommendation-action {
        font-size: 0.95em;
        color: #333333;
        font-style: normal;
        border-top: 1px solid rgba(0, 0, 0, 0.1);
        padding-top: 10px;
    }
    .critical-recommendation {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        border-left-color: #c0392b;
    }
    .high-recommendation {
        background: linear-gradient(135deg, #ffa726 0%, #fb8c00 100%);
        border-left-color: #e65100;
    }
    .medium-recommendation {
        background: linear-gradient(135deg, #42a5f5 0%, #1e88e5 100%);
        border-left-color: #0d47a1;
    }
    .low-recommendation {
        background: linear-gradient(135deg, #66bb6a 0%, #43a047 100%);
        border-left-color: #1b5e20;
    }
    .weak-student-card {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #dc3545;
    }
    .subject-input-container {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 20px;
        border-radius: 12px;
        margin: 20px 0;
        color: white;
    }
    .dynamic-rec-header {
        font-size: 1.5em;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin: 20px 0;
        padding: 15px;
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<div class="main-header">🎓 GradeGraph - Student Performance Analyzer</div>', unsafe_allow_html=True)

# Initialize session state
if 'df_full' not in st.session_state:
    st.session_state.df_full = None
if 'df_suggestions' not in st.session_state:
    st.session_state.df_suggestions = None

# Sidebar for navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Select Page", [
    "📤 Upload & Process",
    "📈 Dashboard",
    "👥 Student Search",
    "📊 Subject Analysis",
    "🎯 Performance Insights",
    "🎯 Threshold-Based Recommendations"
])

# Persistent dashboard snapshot in sidebar
if st.session_state.get('df_full') is not None and st.session_state.get('df_suggestions') is not None:
    df_full_sidebar = st.session_state.df_full
    df_sugg_sidebar = st.session_state.df_suggestions
    with st.sidebar.expander("Dashboard Snapshot", expanded=True):
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Students", len(df_full_sidebar))
        with col_b:
            st.metric("Subjects", len(get_subject_list(df_full_sidebar)))
        col_c, col_d = st.columns(2)
        with col_c:
            bright = (df_sugg_sidebar['Category'] == 'Bright').sum()
            st.metric("Bright", int(bright))
        with col_d:
            weak = (df_sugg_sidebar['Category'] == 'Weak').sum()
            st.metric("Weak", int(weak))

# File upload section
if page == "📤 Upload & Process":
    st.header("📤 Upload Excel Sheet")

    uploaded_file = st.file_uploader(
        "Choose an Excel file",
        type=["xlsx"],
        help="Upload an Excel file containing student performance data starting with 'SR.No.' row"
    )

    if uploaded_file:
        try:
            with st.spinner("🔄 Processing Excel file..."):
                df_full, df_suggestions = process_excel_file(uploaded_file)

            # Store in session state
            st.session_state.df_full = df_full
            st.session_state.df_suggestions = df_suggestions

            st.markdown('<div class="success-msg">✅ File processed successfully!</div>', unsafe_allow_html=True)

            # Display basic statistics (REMOVED avg and excellence scores)
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("📚 Total Students", len(df_full))

            with col2:
                subjects = get_subject_list(df_full)
                st.metric("📖 Subjects", len(subjects))

            with col3:
                bright_count = len(df_suggestions[df_suggestions['Category'] == 'Bright'])
                st.metric("🌟 Bright Learners", bright_count)

            with col4:
                weak_count = len(df_suggestions[df_suggestions['Category'] == 'Weak'])
                st.metric("⚠️ Weak Learners", weak_count)

            # Calculation Logic Explanation
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("### 📊 **Calculation Logic Explanation**")
            st.markdown("""
            **Academic Performance %:**
            - Ratio of total marks obtained vs. total maximum marks possible
            - Formula: `(Total Obtained Marks / Total Maximum Marks) × 100`
            - Maximum marks are extracted from column headers (e.g., "Subject ISE (25)" → 25 marks)

            **Practical Performance %:**
            - Calculated only from columns identified as practical assessments (containing "PRACTICAL" or "PR")
            - Formula: `(Practical Marks Obtained / Total Practical Marks Possible) × 100`

            **Student Classification:**
            - **Bright**: Academic Performance ≥ 80% OR Coding Expertise = 'Advanced'
            - **Average**: Academic Performance 60-79% OR Coding Expertise = 'Intermediate'
            - **Weak**: Academic Performance < 60% AND Coding Expertise = 'Beginner'
            """)
            st.markdown('</div>', unsafe_allow_html=True)

            # Show preview of processed data
            with st.expander("📋 Preview Processed Data", expanded=False):
                st.subheader("🔍 Full Processed Data (Sample)")
                st.dataframe(df_full.head(10), use_container_width=True)

                st.subheader("📊 Learner Classification")
                st.dataframe(df_suggestions, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error while processing file: {e}")
            st.error("Please ensure your Excel file follows the expected format with 'SR.No.' as the starting row.")

    else:
        st.info("📌 Please upload a student Excel file to begin analysis.")

# Dashboard page (REMOVED avg and excellence metrics)
elif page == "📈 Dashboard":
    if st.session_state.df_full is not None:
        df_full = st.session_state.df_full
        df_suggestions = st.session_state.df_suggestions

        st.header("📈 Performance Dashboard")

        # Performance distribution
        col1, col2 = st.columns(2)

        with col1:
            # Category distribution pie chart
            category_counts = df_suggestions['Category'].value_counts()
            fig_pie = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="📊 Student Category Distribution",
                color_discrete_map={
                    'Bright': '#28a745',
                    'Average': '#ffc107',
                    'Weak': '#dc3545',
                    'Unknown': '#6c757d'
                }
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Academic Performance distribution
            if 'Academic_Performance_%' in df_full.columns:
                fig_hist = px.histogram(
                    df_full,
                    x='Academic_Performance_%',
                    title="📈 Academic Performance Distribution",
                    nbins=20,
                    color_discrete_sequence=['#1f77b4']
                )
                fig_hist.update_layout(xaxis_title="Academic Performance %", yaxis_title="Number of Students")
                st.plotly_chart(fig_hist, use_container_width=True)

        # Top and bottom performers
        col3, col4 = st.columns(2)

        with col3:
            st.subheader("🏆 Top 10 Performers")
            if 'Academic_Performance_%' in df_full.columns:
                top_performers = df_full.nlargest(10, 'Academic_Performance_%')[['SR.No', 'Name', 'Academic_Performance_%', 'Category']]
                st.dataframe(top_performers, use_container_width=True)

        with col4:
            st.subheader("⚠️ Students Needing Attention")
            if 'Academic_Performance_%' in df_full.columns:
                bottom_performers = df_full.nsmallest(10, 'Academic_Performance_%')[['SR.No', 'Name', 'Academic_Performance_%', 'Category']]
                st.dataframe(bottom_performers, use_container_width=True)

        # Download options
        st.subheader("⬇️ Download Reports")
        col5, col6 = st.columns(2)

        with col5:
            csv_full = df_full.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📊 Download Full Data CSV",
                csv_full,
                "full_student_data.csv",
                "text/csv"
            )

        with col6:
            csv_suggestions = df_suggestions.to_csv(index=False).encode('utf-8')
            st.download_button(
                "🎯 Download Classifications CSV",
                csv_suggestions,
                "student_classifications.csv",
                "text/csv"
            )

    else:
        st.warning("📤 Please upload and process an Excel file first.")

# Student search page
elif page == "👥 Student Search":
    if st.session_state.df_full is not None:
        df_full = st.session_state.df_full

        st.header("👥 Individual Student Analysis")

        # Search options
        search_col1, search_col2 = st.columns([2, 1])

        with search_col1:
            search_query = st.text_input(
                "🔍 Search Student",
                placeholder="Enter SR.No, Roll No, or Name",
                help="You can search by student number or name"
            )

        with search_col2:
            search_button = st.button("🔍 Search", type="primary")

        if search_query and search_button:
            student_data = get_student_performance(df_full, search_query)

            if student_data:
                st.success(f"✅ Found student: {student_data['Name']}")

                # Student basic info
                info_col1, info_col2, info_col3 = st.columns(3)

                with info_col1:
                    st.metric("📝 SR.No", str(student_data['SR.No']))

                with info_col2:
                    roll_no = student_data.get('Roll No', 'N/A')
                    st.metric("🎓 Roll No", str(roll_no))

                with info_col3:
                    category = student_data.get('Category', 'Unknown')
                    st.metric("📊 Category", str(category))

                # Performance metrics (REMOVED Overall Average)
                perf_col1, perf_col2 = st.columns(2)

                with perf_col1:
                    academic_perf = student_data.get('Academic_Performance_%', 0)
                    if pd.notna(academic_perf) and academic_perf > 0:
                        st.metric("🎯 Academic %", f"{float(academic_perf):.1f}%")
                    else:
                        st.metric("🎯 Academic %", "N/A")

                with perf_col2:
                    coding_expertise = student_data.get('Coding_Expertise', 'N/A')
                    if pd.notna(coding_expertise):
                        coding_full = {'A': 'Advanced', 'I': 'Intermediate', 'B': 'Beginner'}.get(str(coding_expertise), str(coding_expertise))
                        st.metric("💻 Coding Level", coding_full)
                    else:
                        st.metric("💻 Coding Level", "N/A")

                # Subject-wise performance
                st.subheader("📚 Subject-wise Performance")

                # Get subject columns
                exclude_cols = ['SR.No', 'Roll No', 'Name', 'Academic_Performance_%', 'Previous_Performance_Analysis',
                               'Practical_%', 'Coding_Expertise', 'Performance_Analysis', 'Category']

                subject_cols = []
                for col in student_data.keys():
                    if col not in exclude_cols and pd.notna(student_data[col]):
                        try:
                            # Check if it's a numeric value and greater than 0
                            val = float(student_data[col])
                            if val > 0:
                                subject_cols.append(col)
                        except (ValueError, TypeError):
                            continue

                if subject_cols:
                    # Subject performance table only (graph removed as requested)
                    subject_entries = []
                    for col in subject_cols:
                        clean_name = col.replace('_', ' ').title()
                        subject_entries.append({
                            'Subject': clean_name,
                            'Score': float(student_data[col])
                        })

                    subject_df = pd.DataFrame(subject_entries).sort_values('Score', ascending=False)
                    st.dataframe(subject_df, use_container_width=True)
                else:
                    st.info("No subject-wise performance data available for this student.")

            else:
                st.error("❌ Student not found. Please check the search query.")

        elif search_query and not search_button:
            st.info("👆 Click the Search button to find the student.")

    else:
        st.warning("📤 Please upload and process an Excel file first.")

# Subject analysis page
elif page == "📊 Subject Analysis":
    if st.session_state.df_full is not None:
        df_full = st.session_state.df_full
        subjects = get_subject_list(df_full)

        st.header("📊 Subject-wise Analysis")

        # Subject Analysis Logic Explanation (updated)
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### 📚 **Subject Analysis Logic**")
        st.markdown("""
        **Percentages:**
        - Theory % = (ISE + MSE + ESE obtained) / (sum of their max marks) × 100
        - Subject % = (ISE + MSE + ESE + Practical obtained) / (sum of their max marks) × 100
        - Max marks are parsed from headers like "(25)", "(60)"
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        if subjects:
            selected_subject = st.selectbox("📚 Select Subject", subjects)

            if selected_subject:
                # Get all columns for the selected subject
                subject_cols = []
                for col in df_full.columns:
                    # More flexible subject matching
                    col_upper = col.upper()
                    subject_upper = selected_subject.upper()

                    # Check if column starts with subject name or contains it
                    if (col_upper.startswith(subject_upper) or
                        subject_upper in col_upper):
                        # Make sure it's actually a subject column, not metadata
                        exclude_words = ['ACADEMIC', 'PERFORMANCE', 'CODING', 'EXPERTISE', 'CATEGORY']
                        if not any(word in col_upper for word in exclude_words):
                            subject_cols.append(col)

                if subject_cols:
                    st.subheader(f"📈 Analysis for {selected_subject}")

                    # Calculate subject statistics
                    subject_data = df_full[subject_cols + ['Name', 'SR.No', 'Category']].copy()

                    # Split into theory and practical columns
                    theory_cols = [c for c in subject_cols if any(x in c.upper() for x in ['ISE', 'MSE', 'ESE'])]
                    prac_cols = [c for c in subject_cols if _is_practical_col(c)]

                    # Compute percentages per student
                    subject_data['Theory_%'] = subject_data.apply(lambda r: _row_percentage(r, theory_cols), axis=1) if theory_cols else np.nan
                    subject_data['Subject_%'] = subject_data.apply(lambda r: _row_percentage(r, theory_cols + prac_cols), axis=1) if (theory_cols or prac_cols) else np.nan

                    # Performance distribution
                    col1, col2 = st.columns(2)

                    with col1:
                        vals = subject_data['Theory_%'].dropna()
                        if len(vals) > 0:
                            fig_t = px.histogram(vals, nbins=15, title=f"📊 {selected_subject} - Theory % Distribution",
                                                 labels={'value': 'Theory %', 'count': 'Students'})
                            st.plotly_chart(fig_t, use_container_width=True)
                        else:
                            st.info("No theory data available for this subject.")

                    with col2:
                        vals2 = subject_data['Subject_%'].dropna()
                        if len(vals2) > 0:
                            fig_s = px.histogram(vals2, nbins=15, title=f"📊 {selected_subject} - Subject % Distribution",
                                                 labels={'value': 'Subject %', 'count': 'Students'})
                            st.plotly_chart(fig_s, use_container_width=True)
                        else:
                            st.info("No subject percentage data available.")

                    # Top and bottom performers in this subject (aligned side-by-side)
                    col3, col4 = st.columns(2)

                    with col3:
                        st.subheader(f"🏆 Top Performers in {selected_subject}")
                        top_subject = subject_data.nlargest(10, 'Subject_%')[['SR.No', 'Name', 'Subject_%']]
                        top_subject['Subject_%'] = top_subject['Subject_%'].round(2)
                        st.dataframe(top_subject, use_container_width=True)

                    with col4:
                        st.subheader(f"⚠️ Need Improvement in {selected_subject}")
                        bottom_subject = subject_data.nsmallest(10, 'Subject_%')[['SR.No', 'Name', 'Subject_%']]
                        bottom_subject['Subject_%'] = bottom_subject['Subject_%'].round(2)
                        st.dataframe(bottom_subject, use_container_width=True)

                    # Category-wise averages (Subject %)
                    cat_df = subject_data.groupby('Category', dropna=True)['Subject_%'].mean().reset_index().dropna()
                    if len(cat_df) > 0:
                        fig_cat = px.bar(cat_df, x='Category', y='Subject_%', title=f"📈 {selected_subject} - Subject % by Category",
                                         color='Category', color_discrete_map={'Bright': '#28a745','Average': '#ffc107','Weak': '#dc3545'})
                        st.plotly_chart(fig_cat, use_container_width=True)
                else:
                    st.warning(f"No assessment data found for {selected_subject}")
        else:
            st.warning("No subjects found in the data.")

    else:
        st.warning("📤 Please upload and process an Excel file first.")

# Performance insights page (REMOVED avg and excellence metrics)
elif page == "🎯 Performance Insights":
    if st.session_state.df_full is not None:
        df_full = st.session_state.df_full
        df_suggestions = st.session_state.df_suggestions

        st.header("🎯 Performance Insights & Recommendations")

        # Overall statistics (REMOVED avg overall and excellence rate)
        st.subheader("📊 Overall Statistics")

        col1, col2, col3 = st.columns(3)

        with col1:
            total_students = len(df_full)
            st.metric("👥 Total Students", total_students)

        with col2:
            if 'Academic_Performance_%' in df_full.columns:
                avg_academic = df_full['Academic_Performance_%'].mean()
                if pd.notna(avg_academic):
                    st.metric("📈 Avg Academic %", f"{avg_academic:.2f}%")
                else:
                    st.metric("📈 Avg Academic %", "N/A")
            else:
                st.metric("📈 Avg Academic %", "N/A")

        with col3:
            if 'Academic_Performance_%' in df_full.columns:
                avg_academic = df_full['Academic_Performance_%'].mean()
                if pd.notna(avg_academic):
                    st.metric("📈 Avg Academic %", f"{avg_academic:.2f}%")
                else:
                    st.metric("📈 Avg Academic %", "N/A")
            else:
                st.metric("📈 Avg Academic %", "N/A")

        # Category analysis
        st.subheader("📈 Category-wise Analysis")

        category_stats = df_suggestions['Category'].value_counts()

        col5, col6 = st.columns(2)

        with col5:
            # Category distribution with recommendations
            fig_donut = px.pie(
                values=category_stats.values,
                names=category_stats.index,
                title="🎯 Student Distribution by Performance Category",
                hole=0.4,
                color_discrete_map={
                    'Bright': '#28a745',
                    'Average': '#ffc107',
                    'Weak': '#dc3545',
                    'Unknown': '#6c757d'
                }
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        with col6:
            st.markdown("### 📋 Category Insights")

            for category, count in category_stats.items():
                percentage = (count / total_students) * 100

                if category == 'Bright':
                    st.success(f"🌟 **Bright Learners**: {count} ({percentage:.1f}%)")
                    st.write("💡 Continue challenging these students with advanced topics")

                elif category == 'Average':
                    st.info(f"📊 **Average Learners**: {count} ({percentage:.1f}%)")
                    st.write("🎯 Focus on targeted improvement strategies")

                elif category == 'Weak':
                    st.warning(f"⚠️ **Weak Learners**: {count} ({percentage:.1f}%)")
                    st.write("🆘 Require immediate attention and support")

        # Subject difficulty analysis
        subjects = get_subject_list(df_full)
        if subjects:
            st.subheader("📚 Subject Difficulty Analysis")

            subject_difficulty = []
            for subject in subjects:
                subject_cols = []
                for col in df_full.columns:
                    col_upper = col.upper()
                    subject_upper = subject.upper()

                    if (col_upper.startswith(subject_upper) or subject_upper in col_upper):
                        exclude_words = ['ACADEMIC', 'PERFORMANCE', 'CODING', 'EXPERTISE', 'CATEGORY']
                        if not any(word in col_upper for word in exclude_words):
                            if df_full[col].dtype in ['int64', 'float64']:
                                subject_cols.append(col)

                if subject_cols:
                    # Calculate percentage for this subject (ISE + MSE + ESE + Practical)
                    theory_cols = [c for c in subject_cols if any(x in c.upper() for x in ['ISE', 'MSE', 'ESE'])]
                    prac_cols = [c for c in subject_cols if _is_practical_col(c)]

                    # Calculate subject percentage for each student
                    subject_percentages = []
                    for _, row in df_full.iterrows():
                        percentage = _row_percentage(row, theory_cols + prac_cols)
                        if pd.notna(percentage):
                            subject_percentages.append(percentage)

                    if subject_percentages:
                        avg_percentage = np.mean(subject_percentages)
                        fail_rate = (np.array(subject_percentages) < 40).sum() / len(subject_percentages) * 100

                        difficulty_level = "Easy" if avg_percentage >= 70 else "Moderate" if avg_percentage >= 50 else "Difficult"

                        subject_difficulty.append({
                            'Subject': subject,
                            'Avg_Percentage': round(avg_percentage, 2),
                            'Fail_Rate': round(fail_rate, 1),
                            'Difficulty': difficulty_level
                        })

            if subject_difficulty:
                difficulty_df = pd.DataFrame(subject_difficulty).sort_values('Avg_Percentage')

                # Difficulty visualization
                fig_difficulty = px.bar(
                    difficulty_df,
                    x='Subject',
                    y='Avg_Percentage',
                    color='Difficulty',
                    title="📊 Subject Difficulty Analysis (by Percentage)",
                    color_discrete_map={
                        'Easy': '#28a745',
                        'Moderate': '#ffc107',
                        'Difficult': '#dc3545'
                    }
                )
                fig_difficulty.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_difficulty, use_container_width=True)

                st.dataframe(difficulty_df, use_container_width=True)
            else:
                st.warning("No subject difficulty data available.")

        # Actionable Recommendations
        st.subheader("💡 Actionable Recommendations")

        recommendations = []

        # Based on weak learners
        weak_count = category_stats.get('Weak', 0)
        if weak_count > total_students * 0.3:
            recommendations.append({
                'Priority': 'High',
                'Area': 'Academic Support',
                'Recommendation': f'{weak_count} students ({weak_count/total_students*100:.1f}%) need immediate academic intervention',
                'Action': 'Implement remedial classes and peer tutoring programs'
            })

        # Based on subject difficulty
        if subjects and subject_difficulty:
            difficult_subjects = [s for s in subject_difficulty if s['Difficulty'] == 'Difficult']
            if difficult_subjects:
                worst_subject = min(difficult_subjects, key=lambda x: x['Avg_Percentage'])
                recommendations.append({
                    'Priority': 'High',
                    'Area': 'Curriculum',
                    'Recommendation': f"{worst_subject['Subject']} shows lowest performance (avg: {worst_subject['Avg_Percentage']:.1f}%)",
                    'Action': 'Review teaching methodology and provide additional resources'
                })

        # Based on pass rate
        if 'Academic_Performance_%' in df_full.columns:
            pass_count = len(df_full[df_full['Academic_Performance_%'] >= 40])
            pass_rate = (pass_count / len(df_full)) * 100 if len(df_full) > 0 else 0

            if pass_rate < 80:
                recommendations.append({
                    'Priority': 'High',
                    'Area': 'Pass Rate',
                    'Recommendation': f'Pass rate is {pass_rate:.1f}% - below acceptable threshold',
                    'Action': 'Implement comprehensive support system and early warning mechanisms'
                })

        # Display recommendations
        if recommendations:
            for i, rec in enumerate(recommendations):
                priority_color = 'error' if rec['Priority'] == 'High' else 'warning' if rec['Priority'] == 'Medium' else 'info'

                with st.container():
                    st.markdown(f"### {i+1}. {rec['Area']} - {rec['Priority']} Priority")
                    if rec['Priority'] == 'High':
                        st.error(f"🚨 **Issue**: {rec['Recommendation']}")
                        st.error(f"🎯 **Action**: {rec['Action']}")
                    elif rec['Priority'] == 'Medium':
                        st.warning(f"⚠️ **Issue**: {rec['Recommendation']}")
                        st.warning(f"🎯 **Action**: {rec['Action']}")
                    else:
                        st.info(f"ℹ️ **Issue**: {rec['Recommendation']}")
                        st.info(f"🎯 **Action**: {rec['Action']}")
                    st.markdown("---")
        else:
            st.success("🎉 Great! No critical issues identified. Keep up the good work!")

        # Export comprehensive report
        st.subheader("📄 Export Comprehensive Report")

        if st.button("📊 Generate Detailed Report", type="primary"):
            # Create comprehensive report
            report_data = {
                'Overall_Statistics': {
                    'Total_Students': total_students,
                    'Pass_Rate': float(pass_rate) if 'Academic_Performance_%' in df_full.columns else None,
                    'Average_Academic_Performance': float(avg_academic) if 'Academic_Performance_%' in df_full.columns and pd.notna(avg_academic) else None
                },
                'Category_Distribution': category_stats.to_dict(),
                'Subject_Analysis': subject_difficulty if subjects and subject_difficulty else [],
                'Recommendations': recommendations
            }

            # Convert to JSON for download
            import json
            report_json = json.dumps(report_data, indent=2, default=str)

            st.download_button(
                "📥 Download Detailed Report (JSON)",
                report_json,
                "comprehensive_analysis_report.json",
                "application/json"
            )

            st.success("✅ Report generated successfully!")

    else:
        st.warning("📤 Please upload and process an Excel file first.")


# NEW PAGE: Threshold-Based Recommendations
elif page == "🎯 Threshold-Based Recommendations":
    if st.session_state.df_full is not None:
        df_full = st.session_state.df_full
        subjects = get_subject_list(df_full)

        st.markdown('<div class="dynamic-rec-header">🎯 Threshold-Based Student Recommendation System</div>', unsafe_allow_html=True)
        st.markdown("""
        
        """, unsafe_allow_html=True)

        if subjects:
            # Subject and threshold input section
            col1, col2 = st.columns([2, 1])

            with col1:
                selected_subject = st.selectbox(
                    "📚 Select Subject",
                    subjects,
                    help="Choose a subject to analyze"
                )

            with col2:
                # Get available exam types for the selected subject
                exam_types = get_subject_exam_types(df_full, selected_subject)
                exam_types.insert(0, "All Exam Types")  # Add option for all types

                selected_exam_type = st.selectbox(
                    "📝 Exam Type (Optional)",
                    exam_types,
                    help="Choose specific exam type or leave as 'All' for comprehensive analysis"
                )

            # Threshold input with validation
            st.markdown("### 📊 Set Threshold Value")

            col3, col4, col5 = st.columns([1, 1, 1])

            with col3:
                threshold_value = st.number_input(
                    "🎯 Threshold Marks",
                    min_value=0,
                    max_value=500,
                    value=40,
                    help="Enter the minimum marks threshold"
                )

            with col4:
                # Dynamically detect max marks from column headers for subject/exam selection
                subject_upper = selected_subject.upper()
                def _matches_exam(col_upper, exam_type_value):
                    if exam_type_value == "All Exam Types":
                        return True
                    et = exam_type_value.upper()
                    return et in col_upper

                detected_max = 0
                for col in df_full.columns:
                    col_upper = str(col).upper()
                    if subject_upper in col_upper and _matches_exam(col_upper, selected_exam_type):
                        detected_max = max(detected_max, extract_max_marks_from_header(col))
                # Fallback if nothing detected
                if detected_max == 0:
                    detected_max = _get_max_marks_for_subject(selected_subject, None if selected_exam_type == "All Exam Types" else selected_exam_type)

                max_marks = detected_max
                st.metric("💯 Max Marks", int(max_marks))

            with col5:
                # Validation indicator
                if threshold_value > max_marks:
                    st.error(f"❌ Threshold exceeds max marks!")
                else:
                    st.success(f"✅ Valid threshold")

            # Generate recommendations button
            if st.button("🚀 Generate Threshold Recommendations", type="primary", use_container_width=True):

                # Validate threshold before processing
                if threshold_value > max_marks:
                    st.error(f"""
                    🚨 **Invalid Threshold Value!**

                    **Threshold**: {threshold_value} marks
                    **Maximum Marks**: {max_marks} marks

                    Please enter a threshold value that is less than or equal to the maximum marks for {selected_subject} {selected_exam_type if selected_exam_type != "All Exam Types" else ""}.
                    """)
                else:
                    # Get recommendations
                    exam_type_param = None if selected_exam_type == "All Exam Types" else selected_exam_type
                    results = get_threshold_based_recommendations(
                        df_full,
                        selected_subject,
                        threshold_value,
                        exam_type_param
                    )

                    if results['error']:
                        st.error(f"❌ {results['message']}")
                    else:
                        # Display results
                        st.success(f"✅ {results['message']}")

                        # Summary metrics
                        col_sum1, col_sum2, col_sum3 = st.columns(3)

                        with col_sum1:
                            st.metric("📊 Students Above Threshold", len(results['students']))

                        with col_sum2:
                            st.metric("👥 Total Students", results['total_students'])

                        with col_sum3:
                            st.metric("📈 Percentage Above Threshold", f"{results['percentage_above_threshold']:.1f}%")

                        # Display students above threshold
                        if results['students']:
                            st.markdown("### 🏆 Students Above Threshold")

                            # Create DataFrame for display
                            display_data = []
                            for student in results['students']:
                                display_data.append({
                                    'Student Name': student['Student Name'],
                                    'Roll No': student['Roll No'],
                                    'Exam Type': student['Exam Type'],
                                    'Marks': student['Marks'],
                                    'Max Marks': int(student.get('Max Marks', max_marks)),
                                    'Threshold': student['Threshold'],
                                    'Performance': student['Performance']
                                })

                            df_display = pd.DataFrame(display_data)
                            st.dataframe(df_display, use_container_width=True)

                            # Download option
                            csv = df_display.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv,
                                file_name=f"{selected_subject}_threshold_{threshold_value}_results.csv",
                                mime="text/csv"
                            )

                            # Performance insights
                            st.markdown("### 📈 Performance Insights")

                            if results['percentage_above_threshold'] >= 80:
                                st.success("🌟 **Excellent Class Performance**: Most students are performing well above the threshold!")
                            elif results['percentage_above_threshold'] >= 60:
                                st.info("👍 **Good Class Performance**: A majority of students are meeting the threshold.")
                            elif results['percentage_above_threshold'] >= 40:
                                st.warning("⚠️ **Moderate Performance**: Many students need improvement to meet the threshold.")
                            else:
                                st.error("🚨 **Low Performance**: Most students are below the threshold and need immediate support.")

                            # Recommendations based on results
                            st.markdown("### 💡 Recommendations")

                            if results['percentage_above_threshold'] < 50:
                                st.markdown("""
                                **Immediate Actions Needed:**
                                - 📚 Review teaching methodology for this subject
                                - 👨‍🏫 Provide additional support sessions
                                - 📊 Analyze common weak areas
                                - 🤝 Implement peer tutoring programs
                                """)
                            elif results['percentage_above_threshold'] < 75:
                                st.markdown("""
                                **Improvement Strategies:**
                                - 📈 Focus on students below threshold
                                - 🎯 Provide targeted practice materials
                                - 📝 Regular assessment and feedback
                                """)
                            else:
                                st.markdown("""
                                **Maintenance & Enhancement:**
                                - 🏆 Challenge high performers with advanced topics
                                - 👥 Encourage peer mentoring
                                - 📚 Introduce enrichment programs
                                """)
                        else:
                            st.warning("No students found above the specified threshold. Consider lowering the threshold value.")
        else:
            st.warning("No subjects found in the uploaded data. Please upload a file first.")
    else:
        st.warning("📤 Please upload and process an Excel file first.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🎓 GradeGraph -</p>
        <p>Built by Kshitij and Shweta for COMPUTER DEPARTMENT</p>
    </div>
    """,
    unsafe_allow_html=True
)




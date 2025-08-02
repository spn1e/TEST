import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import with error handling
try:
    from src.analysis.statistical import EducationalStatisticsAnalyzer
    from src.analysis.time_series import TimeSeriesEducationAnalyzer
    from src.visualization.plots import EducationalVisualizer
    from src.utils.helpers import DataProcessor, StreamlitHelpers, AnalyticsHelpers
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.error("Please ensure all source files are present and __init__.py files exist")
    st.stop()

st.set_page_config(page_title="Overview", page_icon="📊", layout="wide")

st.title("📊 Learning Analytics Overview")

if 'datasets' in st.session_state and st.session_state.datasets:
    datasets = st.session_state.datasets
    
    # Key metrics dashboard
    st.header("Key Performance Indicators")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Engagement metrics
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = datasets['learning_analytics']['engagement_score'].mean() * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Average Engagement Score"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 25], 'color': "lightgray"},
                    {'range': [25, 50], 'color': "gray"},
                    {'range': [50, 75], 'color': "lightblue"},
                    {'range': [75, 100], 'color': "blue"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col2:
        # Learning progression
        avg_progression = datasets['learning_analytics']['skill_progression'].mean()
        st.metric(
            "Average Skill Progression",
            f"{avg_progression:.2%}",
            f"+{avg_progression - 0.5:.2%} vs baseline"
        )
        
        # Mini line chart with error handling
        try:
            progression_over_time = datasets['quests'].groupby(
                pd.to_datetime(datasets['quests']['start_time']).dt.date
            )['score'].mean()
            
            fig_mini = px.line(
                x=progression_over_time.index,
                y=progression_over_time.values,
                title="Daily Average Quest Scores"
            )
            fig_mini.update_layout(height=200, showlegend=False)
            st.plotly_chart(fig_mini, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating progression chart: {e}")
    
    with col3:
        # Collaboration network
        st.metric(
            "Total Collaborations",
            len(datasets['collaborations']),
            f"{len(datasets['collaborations']) / len(datasets['students']):.1f} per student"
        )
        
        # Collaboration types
        try:
            collab_types = datasets['collaborations']['interaction_type'].value_counts()
            fig_pie = px.pie(
                values=collab_types.values,
                names=collab_types.index,
                title="Collaboration Types"
            )
            fig_pie.update_layout(height=200)
            st.plotly_chart(fig_pie, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating collaboration chart: {e}")
    
    st.markdown("---")
    
    # Detailed analysis sections
    st.header("Detailed Performance Analysis")
    
    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["Grade Level Analysis", "Learning Styles", "Temporal Patterns"])
    
    with tab1:
        try:
            # Grade level comparison
            grade_analysis = datasets['students'].merge(
                datasets['learning_analytics'], on='student_id'
            ).groupby('grade_level').agg({
                'engagement_score': 'mean',
                'quest_completion_rate': 'mean',
                'learning_gain': 'mean'
            }).round(3)
            
            fig_grades = go.Figure()
            
            for metric in ['engagement_score', 'quest_completion_rate']:
                fig_grades.add_trace(go.Bar(
                    x=grade_analysis.index,
                    y=grade_analysis[metric],
                    name=metric.replace('_', ' ').title()
                ))
            
            fig_grades.update_layout(
                title="Performance Metrics by Grade Level",
                xaxis_title="Grade Level",
                yaxis_title="Score",
                barmode='group'
            )
            st.plotly_chart(fig_grades, use_container_width=True)
        except Exception as e:
            st.error(f"Error in grade analysis: {e}")
    
    with tab2:
        try:
            # Learning style analysis
            style_analysis = datasets['students'].merge(
                datasets['learning_analytics'], on='student_id'
            )
            
            fig_box = px.box(
                style_analysis,
                x='learning_style',
                y='engagement_score',
                color='learning_style',
                title="Engagement by Learning Style"
            )
            st.plotly_chart(fig_box, use_container_width=True)
        except Exception as e:
            st.error(f"Error in learning style analysis: {e}")
    
    with tab3:
        try:
            # Temporal patterns
            movements_hourly = pd.to_datetime(datasets['movements']['timestamp']).dt.hour.value_counts().sort_index()
            
            fig_temporal = px.bar(
                x=movements_hourly.index,
                y=movements_hourly.values,
                title="Activity Distribution by Hour of Day",
                labels={'x': 'Hour', 'y': 'Number of Activities'}
            )
            st.plotly_chart(fig_temporal, use_container_width=True)
        except Exception as e:
            st.error(f"Error in temporal analysis: {e}")

else:
    st.warning("⚠️ Please generate data first using the sidebar!")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Machine Learning imports
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

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
    st.error("Please ensure all source files are present")
    st.stop()

st.set_page_config(page_title="Predictive Models", page_icon="🤖", layout="wide")

st.title("🤖 Predictive Analytics & Machine Learning")
st.markdown("### Advanced models for predicting student outcomes and identifying at-risk learners")

if 'datasets' in st.session_state and st.session_state.datasets:
    datasets = st.session_state.datasets
    
    # Prepare data
    try:
        full_data = datasets['students'].merge(
            datasets['learning_analytics'], on='student_id'
        )
    except KeyError as e:
        st.error(f"Missing required dataset: {e}")
        st.stop()
    
    # Model selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        model_type = st.selectbox(
            "Select Prediction Task",
            ["STEM Interest Prediction", "Engagement Level Classification", 
             "Quest Completion Prediction", "At-Risk Student Identification"]
        )
    
    with col2:
        st.info("""
        **Model Types:**
        - Linear Models
        - Random Forest
        - Gradient Boosting
        - Neural Networks
        """)
    
    st.markdown("---")
    
    if model_type == "STEM Interest Prediction":
        st.header("Predicting Post-Intervention STEM Interest")
        
        # Feature selection
        available_features = [
            'stem_interest_pre', 'quest_completion_rate', 'avg_attempts_per_quest',
            'building_complexity_avg', 'total_blocks_placed', 'collaboration_events',
            'days_active', 'skill_progression', 'engagement_score'
        ]
        
        # Filter available features based on what's actually in the data
        available_features = [f for f in available_features if f in full_data.columns]
        
        selected_features = st.multiselect(
            "Select Features for Prediction",
            available_features,
            default=available_features[:min(5, len(available_features))]
        )
        
        if st.button("Train Models", type="primary") and len(selected_features) > 0:
            try:
                # Prepare data
                X = full_data[selected_features].fillna(0)  # Handle missing values
                y = full_data['stem_interest_post'].fillna(full_data['stem_interest_post'].mean())
                
                # Check if we have enough data
                if len(X) < 20:
                    st.warning("Not enough data for reliable model training. Generate more students.")
                    st.stop()
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # Train multiple models
                models = {
                    'Ridge Regression': Ridge(alpha=1.0),
                    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
                }
                
                results = {}
                predictions = {}
                
                # Progress bar
                progress_bar = st.progress(0)
                
                for i, (name, model) in enumerate(models.items()):
                    try:
                        # Train model
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        
                        # Calculate metrics
                        results[name] = {
                            'r2': r2_score(y_test, y_pred),
                            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                            'mae': mean_absolute_error(y_test, y_pred)
                        }
                        predictions[name] = y_pred
                    except Exception as e:
                        st.error(f"Error training {name}: {e}")
                        results[name] = {'r2': 0, 'rmse': np.inf, 'mae': np.inf}
                        predictions[name] = np.zeros_like(y_test)
                    
                    progress_bar.progress((i + 1) / len(models))
                
                progress_bar.empty()
                
                # Display results
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Model comparison
                    fig_comparison = go.Figure()
                    
                    for metric in ['r2', 'rmse', 'mae']:
                        values = [results[model][metric] for model in models.keys()]
                        fig_comparison.add_trace(go.Bar(
                            name=metric.upper(),
                            x=list(models.keys()),
                            y=values
                        ))
                    
                    fig_comparison.update_layout(
                        title="Model Performance Comparison",
                        barmode='group',
                        height=400
                    )
                    st.plotly_chart(fig_comparison, use_container_width=True)
                
                with col2:
                    # Best model summary
                    best_model = max(results.items(), key=lambda x: x[1]['r2'])[0]
                    st.metric("Best Model", best_model)
                    st.metric("R² Score", f"{results[best_model]['r2']:.3f}")
                    st.metric("RMSE", f"{results[best_model]['rmse']:.3f}")
                
                # Prediction vs Actual scatter plots
                st.markdown("### Prediction Accuracy Visualization")
                
                fig_scatter = make_subplots(
                    rows=1, cols=3,
                    subplot_titles=list(models.keys())
                )
                
                for i, (name, y_pred) in enumerate(predictions.items()):
                    fig_scatter.add_trace(
                        go.Scatter(
                            x=y_test,
                            y=y_pred,
                            mode='markers',
                            name=name,
                            marker=dict(size=8, opacity=0.6)
                        ),
                        row=1, col=i+1
                    )
                    
                    # Add perfect prediction line
                    fig_scatter.add_trace(
                        go.Scatter(
                            x=[y_test.min(), y_test.max()],
                            y=[y_test.min(), y_test.max()],
                            mode='lines',
                            name='Perfect Prediction',
                            line=dict(dash='dash', color='red'),
                            showlegend=(i==0)
                        ),
                        row=1, col=i+1
                    )
                
                fig_scatter.update_xaxes(title_text="Actual STEM Interest")
                fig_scatter.update_yaxes(title_text="Predicted STEM Interest")
                fig_scatter.update_layout(height=400)
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Feature importance (for tree-based models)
                st.markdown("### Feature Importance Analysis")
                
                if hasattr(models['Random Forest'], 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'feature': selected_features,
                        'importance': models['Random Forest'].feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    fig_importance = px.bar(
                        importance_df,
                        x='importance',
                        y='feature',
                        orientation='h',
                        title="Random Forest Feature Importance"
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error in model training: {e}")
                st.error("Please check your data and try again")
    
    elif model_type == "At-Risk Student Identification":
        st.header("Early Warning System for At-Risk Students")
        
        # Define at-risk criteria
        st.markdown("### Define At-Risk Criteria")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            engagement_threshold = st.slider(
                "Min Engagement Score",
                0.0, 1.0, 0.3,
                help="Students below this threshold are considered at-risk"
            )
        
        with col2:
            completion_threshold = st.slider(
                "Min Quest Completion Rate",
                0.0, 1.0, 0.4,
                help="Students below this threshold are considered at-risk"
            )
        
        with col3:
            days_threshold = st.slider(
                "Min Active Days",
                0, 30, 5,
                help="Students with fewer active days are considered at-risk"
            )
        
        try:
            # Identify at-risk students
            full_data['at_risk'] = (
                (full_data['engagement_score'] < engagement_threshold) |
                (full_data['quest_completion_rate'] < completion_threshold) |
                (full_data['days_active'] < days_threshold)
            ).astype(int)
            
            # Display statistics
            at_risk_count = full_data['at_risk'].sum()
            at_risk_pct = at_risk_count / len(full_data) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("At-Risk Students", at_risk_count)
            with col2:
                st.metric("Percentage At-Risk", f"{at_risk_pct:.1f}%")
            with col3:
                st.metric("Total Students", len(full_data))
            
            # Risk factors analysis
            st.markdown("### Risk Factor Analysis")
            
            # Compare at-risk vs not at-risk
            risk_comparison = full_data.groupby('at_risk')[
                ['engagement_score', 'quest_completion_rate', 'collaboration_events', 
                 'skill_progression', 'days_active']
            ].mean()
            
            fig_risk = go.Figure()
            
            for col in risk_comparison.columns:
                fig_risk.add_trace(go.Bar(
                    name=col.replace('_', ' ').title(),
                    x=['Not At-Risk', 'At-Risk'],
                    y=risk_comparison[col].values
                ))
            
            fig_risk.update_layout(
                title="Average Metrics: At-Risk vs Not At-Risk Students",
                barmode='group',
                height=400
            )
            st.plotly_chart(fig_risk, use_container_width=True)
            
            # Individual student profiles
            st.markdown("### Individual Student Risk Profiles")
            
            at_risk_students = full_data[full_data['at_risk'] == 1].head(10)
            
            if len(at_risk_students) > 0:
                # Create risk score
                at_risk_students = at_risk_students.copy()  # Avoid SettingWithCopyWarning
                at_risk_students['risk_score'] = (
                    (1 - at_risk_students['engagement_score']) * 0.4 +
                    (1 - at_risk_students['quest_completion_rate']) * 0.4 +
                    (1 - at_risk_students['days_active'] / 30) * 0.2
                )
                
                # Display top at-risk students
                display_cols = ['student_id', 'engagement_score', 'quest_completion_rate', 
                               'days_active', 'risk_score']
                
                st.dataframe(
                    at_risk_students[display_cols].sort_values('risk_score', ascending=False)
                    .style.background_gradient(subset=['risk_score'], cmap='Reds')
                    .format({
                        'engagement_score': '{:.2f}',
                        'quest_completion_rate': '{:.2f}',
                        'risk_score': '{:.2f}'
                    })
                )
                
                # Intervention recommendations
                st.markdown("### 🎯 Recommended Interventions")
                
                interventions = {
                    "Low Engagement": "Schedule one-on-one mentoring sessions, provide personalized challenges",
                    "Low Quest Completion": "Adjust difficulty levels, provide scaffolded support",
                    "Low Activity": "Send engagement reminders, create peer study groups",
                    "Multiple Risk Factors": "Comprehensive support plan with weekly check-ins"
                }
                
                for intervention, description in interventions.items():
                    st.info(f"**{intervention}:** {description}")
            else:
                st.success("No at-risk students found with current criteria!")
                
        except Exception as e:
            st.error(f"Error in risk analysis: {e}")

else:
    st.warning("⚠️ Please generate data first using the sidebar!")

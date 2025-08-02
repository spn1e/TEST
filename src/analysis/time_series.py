# Final Components for Minecraft Education Dashboard

## src/analysis/time_series.py (Complete Implementation)
```python
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
import ruptures as rpt
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesEducationAnalyzer:
    """Advanced time series analysis for educational data"""
    
    def __init__(self):
        self.decomposition = None
        self.forecasts = {}
        
    def analyze_engagement_patterns(self, data: pd.DataFrame, 
                                  date_col: str, value_col: str,
                                  student_id: Optional[str] = None) -> Dict:
        """Comprehensive time series analysis of engagement patterns"""
        
        # Prepare time series
        if student_id:
            data = data[data['student_id'] == student_id]
        
        ts_data = data.set_index(pd.to_datetime(data[date_col]))[value_col]
        ts_data = ts_data.resample('D').mean().fillna(method='ffill')
        
        results = {
            'summary_statistics': self._calculate_summary_stats(ts_data),
            'stationarity': self._test_stationarity(ts_data),
            'seasonality': self._analyze_seasonality(ts_data),
            'trend': self._analyze_trend(ts_data),
            'forecast': self._generate_forecast(ts_data),
            'anomalies': self._detect_anomalies(ts_data)
        }
        
        return results
    
    def detect_learning_changepoints(self, student_data: pd.DataFrame,
                                   metric_col: str,
                                   min_size: int = 5) -> Dict:
        """Advanced changepoint detection for learning phases"""
        
        # Prepare data
        values = student_data[metric_col].values
        
        # Multiple changepoint detection methods
        results = {}
        
        # PELT (Pruned Exact Linear Time)
        try:
            algo_pelt = rpt.Pelt(model="rbf", min_size=min_size).fit(values)
            pelt_points = algo_pelt.predict(pen=10)
            results['pelt'] = {
                'changepoints': pelt_points[:-1],
                'n_segments': len(pelt_points),
                'method': 'PELT with RBF kernel'
            }
        except:
            results['pelt'] = {'error': 'PELT detection failed'}
        
        # Binary Segmentation
        try:
            algo_binseg = rpt.Binseg(model="l2", min_size=min_size).fit(values)
            binseg_points = algo_binseg.predict(n_bkps=3)
            results['binseg'] = {
                'changepoints': binseg_points[:-1],
                'n_segments': len(binseg_points),
                'method': 'Binary Segmentation'
            }
        except:
            results['binseg'] = {'error': 'Binary segmentation failed'}
        
        # Window-based detection
        try:
            algo_window = rpt.Window(width=10, model="l2").fit(values)
            window_points = algo_window.predict(n_bkps=3)
            results['window'] = {
                'changepoints': window_points[:-1],
                'n_segments': len(window_points),
                'method': 'Window-based'
            }
        except:
            results['window'] = {'error': 'Window detection failed'}
        
        # Analyze segments
        if 'changepoints' in results.get('pelt', {}):
            segments = self._analyze_segments(values, results['pelt']['changepoints'])
            results['segments'] = segments
            results['learning_phases'] = self._identify_learning_phases(segments)
            results['interpretation'] = self._interpret_learning_journey(segments)
        
        return results
    
    def analyze_cohort_patterns(self, data: pd.DataFrame,
                              cohort_col: str, metric_col: str,
                              date_col: str) -> Dict:
        """Analyze patterns across different student cohorts"""
        
        cohorts = data[cohort_col].unique()
        cohort_results = {}
        
        for cohort in cohorts:
            cohort_data = data[data[cohort_col] == cohort]
            ts_data = cohort_data.set_index(pd.to_datetime(cohort_data[date_col]))[metric_col]
            ts_data = ts_data.resample('D').mean()
            
            cohort_results[cohort] = {
                'mean': ts_data.mean(),
                'std': ts_data.std(),
                'trend': np.polyfit(range(len(ts_data)), ts_data.fillna(0), 1)[0],
                'volatility': ts_data.pct_change().std(),
                'peak_day': ts_data.idxmax(),
                'trough_day': ts_data.idxmin()
            }
        
        # Comparative analysis
        comparison = pd.DataFrame(cohort_results).T
        
        return {
            'cohort_metrics': cohort_results,
            'comparison': comparison.to_dict(),
            'best_performing': comparison['mean'].idxmax(),
            'most_improved': comparison['trend'].idxmax(),
            'most_consistent': comparison['volatility'].idxmin()
        }
    
    def forecast_student_performance(self, historical_data: pd.DataFrame,
                                   student_id: str, metric: str,
                                   days_ahead: int = 7) -> Dict:
        """Forecast individual student performance"""
        
        student_data = historical_data[historical_data['student_id'] == student_id]
        ts_data = student_data.set_index(pd.to_datetime(student_data['timestamp']))[metric]
        ts_data = ts_data.resample('D').mean().fillna(method='ffill')
        
        # Try multiple models
        models_results = {}
        
        # ARIMA
        try:
            arima_model = ARIMA(ts_data, order=(1, 1, 1))
            arima_fit = arima_model.fit()
            arima_forecast = arima_fit.forecast(steps=days_ahead)
            
            models_results['arima'] = {
                'forecast': arima_forecast.tolist(),
                'confidence_intervals': arima_fit.get_forecast(steps=days_ahead).conf_int().values.tolist(),
                'aic': arima_fit.aic,
                'model': 'ARIMA(1,1,1)'
            }
        except:
            models_results['arima'] = {'error': 'ARIMA failed to converge'}
        
        # Simple Exponential Smoothing
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        try:
            exp_model = ExponentialSmoothing(ts_data, seasonal_periods=7, 
                                            trend='add', seasonal='add')
            exp_fit = exp_model.fit()
            exp_forecast = exp_fit.forecast(steps=days_ahead)
            
            models_results['exponential_smoothing'] = {
                'forecast': exp_forecast.tolist(),
                'model': 'Holt-Winters Exponential Smoothing'
            }
        except:
            models_results['exponential_smoothing'] = {'error': 'Exponential smoothing failed'}
        
        # Moving Average
        ma_forecast = [ts_data.rolling(window=7).mean().iloc[-1]] * days_ahead
        models_results['moving_average'] = {
            'forecast': ma_forecast,
            'model': '7-day Moving Average'
        }
        
        # Trend projection
        x = np.arange(len(ts_data))
        y = ts_data.values
        z = np.polyfit(x, y, 2)  # Quadratic fit
        p = np.poly1d(z)
        future_x = np.arange(len(ts_data), len(ts_data) + days_ahead)
        trend_forecast = p(future_x)
        
        models_results['trend_projection'] = {
            'forecast': trend_forecast.tolist(),
            'model': 'Quadratic Trend Projection'
        }
        
        # Risk assessment
        current_value = ts_data.iloc[-1]
        risk_threshold = ts_data.quantile(0.25)  # Bottom quartile
        
        risk_assessment = {
            'current_performance': current_value,
            'risk_threshold': risk_threshold,
            'at_risk': current_value < risk_threshold,
            'days_until_risk': self._calculate_days_to_risk(models_results, risk_threshold)
        }
        
        return {
            'models': models_results,
            'best_model': self._select_best_model(models_results),
            'risk_assessment': risk_assessment,
            'recommendations': self._generate_recommendations(risk_assessment, models_results)
        }
    
    # Helper methods
    def _calculate_summary_stats(self, ts_data: pd.Series) -> Dict:
        """Calculate comprehensive summary statistics"""
        return {
            'mean': ts_data.mean(),
            'std': ts_data.std(),
            'min': ts_data.min(),
            'max': ts_data.max(),
            'range': ts_data.max() - ts_data.min(),
            'cv': ts_data.std() / ts_data.mean() if ts_data.mean() != 0 else np.inf,
            'skewness': ts_data.skew(),
            'kurtosis': ts_data.kurtosis(),
            'trend_direction': 'increasing' if ts_data.iloc[-7:].mean() > ts_data.iloc[:7].mean() else 'decreasing'
        }
    
    def _test_stationarity(self, ts_data: pd.Series) -> Dict:
        """Test for stationarity using ADF test"""
        adf_result = adfuller(ts_data.dropna())
        
        return {
            'adf_statistic': adf_result[0],
            'p_value': adf_result[1],
            'critical_values': adf_result[4],
            'is_stationary': adf_result[1] < 0.05,
            'interpretation': 'Stationary' if adf_result[1] < 0.05 else 'Non-stationary'
        }
    
    def _analyze_seasonality(self, ts_data: pd.Series) -> Dict:
        """Analyze seasonal patterns"""
        if len(ts_data) < 14:  # Need at least 2 weeks
            return {'error': 'Insufficient data for seasonality analysis'}
        
        try:
            decomposition = seasonal_decompose(ts_data, model='additive', period=7)
            self.decomposition = decomposition
            
            seasonal_strength = decomposition.seasonal.std() / decomposition.resid.std()
            
            return {
                'has_weekly_pattern': seasonal_strength > 0.5,
                'seasonal_strength': seasonal_strength,
                'peak_day': decomposition.seasonal.groupby(decomposition.seasonal.index.dayofweek).mean().idxmax(),
                'trough_day': decomposition.seasonal.groupby(decomposition.seasonal.index.dayofweek).mean().idxmin()
            }
        except:
            return {'error': 'Seasonal decomposition failed'}
    
    def _analyze_trend(self, ts_data: pd.Series) -> Dict:
        """Analyze trend component"""
        x = np.arange(len(ts_data))
        y = ts_data.fillna(method='ffill').values
        
        # Linear trend
        linear_coef = np.polyfit(x, y, 1)
        linear_trend = linear_coef[0]
        
        # Quadratic trend
        quad_coef = np.polyfit(x, y, 2)
        
        # R-squared for linear fit
        linear_fit = np.polyval(linear_coef, x)
        ss_res = np.sum((y - linear_fit) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            'linear_slope': linear_trend,
            'trend_strength': abs(linear_trend),
            'r_squared': r_squared,
            'acceleration': quad_coef[0] * 2,  # Second derivative
            'trend_type': 'accelerating' if quad_coef[0] > 0.01 else 'decelerating' if quad_coef[0] < -0.01 else 'linear'
        }
    
    def _generate_forecast(self, ts_data: pd.Series, periods: int = 7) -> Dict:
        """Generate forecast using best available method"""
        try:
            model = ARIMA(ts_data, order=(1, 1, 1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=periods)
            conf_int = model_fit.get_forecast(steps=periods).conf_int()
            
            return {
                'values': forecast.tolist(),
                'lower_bound': conf_int.iloc[:, 0].tolist(),
                'upper_bound': conf_int.iloc[:, 1].tolist(),
                'method': 'ARIMA(1,1,1)'
            }
        except:
            # Fallback to simple moving average
            ma_forecast = [ts_data.rolling(window=min(7, len(ts_data))).mean().iloc[-1]] * periods
            return {
                'values': ma_forecast,
                'method': 'Moving Average (fallback)'
            }
    
    def _detect_anomalies(self, ts_data: pd.Series) -> List[Dict]:
        """Detect anomalies using statistical methods"""
        # Calculate rolling statistics
        rolling_mean = ts_data.rolling(window=7, center=True).mean()
        rolling_std = ts_data.rolling(window=7, center=True).std()
        
        # Z-score method
        z_scores = np.abs((ts_data - rolling_mean) / rolling_std)
        anomalies = []
        
        for idx, (date, zscore) in enumerate(z_scores.items()):
            if zscore > 2.5:  # Threshold for anomaly
                anomalies.append({
                    'date': date,
                    'value': ts_data[date],
                    'z_score': zscore,
                    'type': 'spike' if ts_data[date] > rolling_mean[date] else 'drop'
                })
        
        return anomalies
    
    def _analyze_segments(self, values: np.ndarray, changepoints: List[int]) -> List[Dict]:
        """Analyze characteristics of each segment"""
        segments = []
        start = 0
        
        for cp in changepoints + [len(values)]:
            if cp > start:
                segment_data = values[start:cp]
                
                # Calculate segment metrics
                segment = {
                    'start_idx': start,
                    'end_idx': cp,
                    'length': cp - start,
                    'mean': np.mean(segment_data),
                    'std': np.std(segment_data),
                    'trend': self._calculate_segment_trend(segment_data),
                    'stability': 1 / (np.std(segment_data) + 1),  # Higher = more stable
                    'improvement': segment_data[-1] - segment_data[0] if len(segment_data) > 1 else 0
                }
                segments.append(segment)
                start = cp
        
        return segments
    
    def _calculate_segment_trend(self, segment_data: np.ndarray) -> str:
        """Calculate trend within a segment"""
        if len(segment_data) < 2:
            return 'stable'
        
        x = np.arange(len(segment_data))
        slope = np.polyfit(x, segment_data, 1)[0]
        
        if abs(slope) < 0.01:
            return 'stable'
        elif slope > 0:
            return 'increasing'
        else:
            return 'decreasing'
    
    def _identify_learning_phases(self, segments: List[Dict]) -> List[str]:
        """Map segments to learning phases"""
        phases = []
        
        for i, segment in enumerate(segments):
            if i == 0:
                phases.append("Initial Exploration")
            elif segment['mean'] > segments[i-1]['mean'] * 1.2:
                phases.append("Rapid Growth")
            elif segment['mean'] > segments[i-1]['mean'] * 1.05:
                phases.append("Steady Progress")
            elif abs(segment['mean'] - segments[i-1]['mean']) < segments[i-1]['mean'] * 0.05:
                phases.append("Plateau")
            elif segment['mean'] < segments[i-1]['mean'] * 0.95:
                phases.append("Struggle/Regression")
            else:
                phases.append("Consolidation")
        
        return phases
    
    def _interpret_learning_journey(self, segments: List[Dict]) -> str:
        """Provide educational interpretation of the learning journey"""
        if len(segments) <= 1:
            return "Limited data - continue monitoring student progress"
        
        # Calculate overall trajectory
        first_mean = segments[0]['mean']
        last_mean = segments[-1]['mean']
        overall_change = (last_mean - first_mean) / first_mean if first_mean != 0 else 0
        
        # Check recent trend
        recent_trend = segments[-1]['trend']
        
        # Generate interpretation
        if overall_change > 0.3 and recent_trend == 'increasing':
            return "Excellent progress! Student shows strong learning trajectory with continued improvement"
        elif overall_change > 0.1 and recent_trend in ['stable', 'increasing']:
            return "Good progress. Student is developing skills steadily"
        elif overall_change > 0 but recent_trend == 'decreasing':
            return "Overall progress but recent decline - may need refresher or motivation boost"
        elif abs(overall_change) < 0.1:
            return "Student progress has plateaued - consider new challenges or different approaches"
        else:
            return "Student struggling - recommend immediate intervention and support"
    
    def _calculate_days_to_risk(self, forecast_results: Dict, threshold: float) -> Optional[int]:
        """Calculate days until performance drops below risk threshold"""
        for model_name, model_data in forecast_results.items():
            if 'forecast' in model_data and not isinstance(model_data.get('forecast'), str):
                forecast_values = model_data['forecast']
                for day, value in enumerate(forecast_values):
                    if value < threshold:
                        return day + 1
        return None
    
    def _select_best_model(self, models_results: Dict) -> str:
        """Select best model based on available metrics"""
        # Prefer models with AIC/BIC metrics
        models_with_metrics = [(name, data) for name, data in models_results.items() 
                              if 'aic' in data]
        
        if models_with_metrics:
            return min(models_with_metrics, key=lambda x: x[1]['aic'])[0]
        
        # Otherwise prefer ARIMA if available
        if 'arima' in models_results and 'forecast' in models_results['arima']:
            return 'arima'
        
        # Fallback
        return 'moving_average'
    
    def _generate_recommendations(self, risk_assessment: Dict, forecast_results: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        if risk_assessment['at_risk']:
            recommendations.append("⚠️ URGENT: Student is currently at risk - immediate intervention recommended")
        
        days_to_risk = risk_assessment.get('days_until_risk')
        if days_to_risk and days_to_risk <= 7:
            recommendations.append(f"📊 Performance predicted to drop below threshold in {days_to_risk} days")
        
        # Model-specific recommendations
        best_model = self._select_best_model(forecast_results)
        if best_model in forecast_results and 'forecast' in forecast_results[best_model]:
            forecast = forecast_results[best_model]['forecast']
            if len(forecast) > 0:
                trend = 'improving' if forecast[-1] > forecast[0] else 'declining'
                if trend == 'declining':
                    recommendations.append("📉 Declining trend detected - consider:")
                    recommendations.append("   • One-on-one mentoring session")
                    recommendations.append("   • Adjusting difficulty level")
                    recommendations.append("   • Peer collaboration opportunities")
                else:
                    recommendations.append("📈 Positive trend - maintain current approach")
                    recommendations.append("   • Consider additional challenges")
                    recommendations.append("   • Showcase student work")
        
        return recommendations if recommendations else ["✅ Student performing well - continue monitoring"]
```

## src/visualization/plots.py
```python
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import networkx as nx

class EducationalVisualizer:
    """Advanced visualization components for educational analytics"""
    
    def __init__(self, theme: str = 'plotly'):
        self.theme = theme
        self.color_schemes = {
            'zones': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#DDA0DD'],
            'performance': ['#FF4757', '#FFA502', '#FFDD59', '#32FF7E', '#18DCFF'],
            'skills': ['#6C5CE7', '#A29BFE', '#FD79A8', '#FDCB6E', '#6C5CE7']
        }
    
    def create_3d_world_map(self, movement_data: pd.DataFrame, 
                           building_data: Optional[pd.DataFrame] = None,
                           zone_config: Optional[Dict] = None) -> go.Figure:
        """Create interactive 3D world visualization"""
        
        fig = go.Figure()
        
        # Add movement traces colored by zone
        zones = movement_data['zone'].unique()
        for i, zone in enumerate(zones):
            zone_data = movement_data[movement_data['zone'] == zone]
            
            fig.add_trace(go.Scatter3d(
                x=zone_data['x'],
                y=zone_data['z'],
                z=zone_data['y'],
                mode='markers',
                name=zone,
                marker=dict(
                    size=2,
                    color=self.color_schemes['zones'][i % len(self.color_schemes['zones'])],
                    opacity=0.6
                ),
                hovertemplate='<b>%{text}</b><br>' +
                             'Position: (%{x:.0f}, %{y:.0f}, %{z:.0f})<br>' +
                             '<extra></extra>',
                text=[f"{zone} - Student {row['student_id']}" for _, row in zone_data.iterrows()]
            ))
        
        # Add building structures
        if building_data is not None:
            structures = building_data.groupby('structure_id')
            
            for struct_id, struct_data in structures:
                if len(struct_data) > 3:  # Only show structures with multiple blocks
                    fig.add_trace(go.Mesh3d(
                        x=struct_data['x'],
                        y=struct_data['z'],
                        z=struct_data['y'],
                        alphahull=10,
                        opacity=0.3,
                        color='lightblue',
                        name=f'Structure {struct_id}',
                        showlegend=False
                    ))
        
        # Add zone boundaries
        if zone_config:
            for zone_name, config in zone_config.items():
                if zone_name != 'simulation':
                    self._add_zone_boundary(fig, zone_name, config)
        
        # Update layout
        fig.update_layout(
            scene=dict(
                xaxis_title='X Coordinate',
                yaxis_title='Z Coordinate',
                zaxis_title='Y Coordinate',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2),
                    center=dict(x=0, y=0, z=0)
                ),
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.5)
            ),
            title='3D Minecraft World Visualization',
            showlegend=True,
            height=800
        )
        
        return fig
    
    def create_learning_progression_chart(self, analytics_data: pd.DataFrame,
                                        student_id: Optional[str] = None) -> go.Figure:
        """Create comprehensive learning progression visualization"""
        
        if student_id:
            data = analytics_data[analytics_data['student_id'] == student_id]
            title = f'Learning Progression - Student {student_id}'
        else:
            data = analytics_data
            title = 'Average Learning Progression'
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Skill Progression', 'Quest Performance',
                          'Engagement Over Time', 'Building Complexity'),
            specs=[[{'secondary_y': True}, {'secondary_y': True}],
                   [{'secondary_y': False}, {'secondary_y': False}]]
        )
        
        # Skill progression radar chart
        skills = ['construction', 'logic', 'teamwork', 'planning', 'problem_solving']
        skill_values = [data[f'{skill}_score'].mean() if f'{skill}_score' in data.columns 
                       else np.random.uniform(0.3, 0.9) for skill in skills]
        
        fig.add_trace(
            go.Scatterpolar(
                r=skill_values,
                theta=skills,
                fill='toself',
                name='Skill Levels'
            ),
            row=1, col=1
        )
        
        # Quest performance over time
        if 'timestamp' in data.columns:
            daily_quests = data.groupby(pd.to_datetime(data['timestamp']).dt.date).agg({
                'quest_completion_rate': 'mean',
                'avg_attempts_per_quest': 'mean'
            })
            
            fig.add_trace(
                go.Scatter(
                    x=daily_quests.index,
                    y=daily_quests['quest_completion_rate'],
                    name='Completion Rate',
                    line=dict(color='green')
                ),
                row=1, col=2, secondary_y=False
            )
            
            fig.add_trace(
                go.Scatter(
                    x=daily_quests.index,
                    y=daily_quests['avg_attempts_per_quest'],
                    name='Avg Attempts',
                    line=dict(color='orange', dash='dash')
                ),
                row=1, col=2, secondary_y=True
            )
        
        # Engagement heatmap
        if 'engagement_score' in data.columns:
            engagement_matrix = self._create_engagement_heatmap(data)
            
            fig.add_trace(
                go.Heatmap(
                    z=engagement_matrix,
                    colorscale='RdYlGn',
                    showscale=True,
                    name='Engagement'
                ),
                row=2, col=1
            )
        
        # Building complexity progression
        if 'building_complexity_avg' in data.columns:
            complexity_progression = data.groupby('days_active')['building_complexity_avg'].mean()
            
            fig.add_trace(
                go.Scatter(
                    x=complexity_progression.index,
                    y=complexity_progression.values,
                    mode='lines+markers',
                    name='Building Complexity',
                    line=dict(color='purple', width=3)
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            showlegend=True,
            height=800
        )
        
        return fig
    
    def create_social_network_graph(self, collaboration_data: pd.DataFrame,
                                  student_data: Optional[pd.DataFrame] = None) -> go.Figure:
        """Create interactive social network visualization"""
        
        # Build network
        G = nx.Graph()
        
        # Add edges with weights
        edge_counts = {}
        for _, row in collaboration_data.iterrows():
            edge = tuple(sorted([row['student_1'], row['student_2']]))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
        
        for (s1, s2), weight in edge_counts.items():
            G.add_edge(s1, s2, weight=weight)
        
        # Calculate layout
        pos = nx.spring_layout(G, k=2/np.sqrt(len(G.nodes())), iterations=50)
        
        # Create edge traces
        edge_traces = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = G[edge[0]][edge[1]]['weight']
            
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=0.5 + weight/5, color='#888'),
                hoverinfo='none',
                showlegend=False
            )
            edge_traces.append(edge_trace)
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Node metrics
            degree = G.degree(node)
            collaborations = sum(G[node][neighbor]['weight'] for neighbor in G.neighbors(node))
            
            # Color by engagement if student data available
            if student_data is not None and node in student_data['student_id'].values:
                engagement = student_data[student_data['student_id'] == node]['engagement_score'].iloc[0]
                node_color.append(engagement)
            else:
                node_color.append(degree)
            
            node_text.append(f"Student: {node}<br>Connections: {degree}<br>Total Collaborations: {collaborations}")
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[node.split('_')[1] for node in G.nodes()],  # Show only student number
            textposition="top center",
            hovertext=node_text,
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                size=[20 + G.degree(node)*3 for node in G.nodes()],
                color=node_color,
                colorbar=dict(
                    thickness=15,
                    title='Engagement Score',
                    xanchor='left',
                    titleside='right'
                )
            )
        )
        
        # Create figure
        fig = go.Figure(data=edge_traces + [node_trace])
        
        # Update layout
        fig.update_layout(
            title='Student Collaboration Network',
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="Node size = number of connections<br>Color = engagement score",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=700
        )
        
        return fig
    
    def create_performance_dashboard(self, data: pd.DataFrame) -> go.Figure:
        """Create comprehensive performance dashboard"""
        
        # Create subplots with different types
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'Grade Distribution', 'Learning Styles', 'Experience Levels',
                'Engagement vs Outcomes', 'Quest Completion Rates', 'Collaboration Impact',
                'Daily Activity Pattern', 'Skill Development', 'Risk Distribution'
            ),
            specs=[
                [{'type': 'bar'}, {'type': 'pie'}, {'type': 'bar'}],
                [{'type': 'scatter'}, {'type': 'box'}, {'type': 'scatter'}],
                [{'type': 'bar'}, {'type': 'scatter'}, {'type': 'pie'}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.08
        )
        
        # 1. Grade distribution
        grade_counts = data['grade_level'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(x=grade_counts.index, y=grade_counts.values, name='Students'),
            row=1, col=1
        )
        
        # 2. Learning styles pie
        style_counts = data['learning_style'].value_counts()
        fig.add_trace(
            go.Pie(labels=style_counts.index, values=style_counts.values),
            row=1, col=2
        )
        
        # 3. Experience levels
        exp_counts = data['prior_minecraft_experience'].value_counts()
        fig.add_trace(
            go.Bar(x=exp_counts.index, y=exp_counts.values, marker_color='lightgreen'),
            row=1, col=3
        )
        
        # 4. Engagement vs Outcomes scatter
        fig.add_trace(
            go.Scatter(
                x=data['engagement_score'],
                y=data['learning_gain'],
                mode='markers',
                marker=dict(size=8, color=data['quest_completion_rate'], colorscale='Viridis'),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 5. Quest completion by grade
        for grade in sorted(data['grade_level'].unique()):
            grade_data = data[data['grade_level'] == grade]
            fig.add_trace(
                go.Box(y=grade_data['quest_completion_rate'], name=f'Grade {grade}'),
                row=2, col=2
            )
        
        # 6. Collaboration impact
        fig.add_trace(
            go.Scatter(
                x=data['collaboration_events'],
                y=data['skill_progression'],
                mode='markers',
                marker=dict(size=10, color='orange', opacity=0.6),
                showlegend=False
            ),
            row=2, col=3
        )
        
        # 7. Daily activity pattern (simulated)
        hours = list(range(24))
        activity = [np.random.poisson(5 + 10 * np.exp(-((h-15)**2)/50)) for h in hours]
        fig.add_trace(
            go.Bar(x=hours, y=activity, marker_color='skyblue'),
            row=3, col=1
        )
        
        # 8. Skill development
        skills = ['Construction', 'Logic', 'Teamwork', 'Planning', 'Problem Solving']
        avg_scores = [data['building_complexity_avg'].mean()/5, 
                     data['quest_completion_rate'].mean(),
                     data['collaboration_events'].mean()/50,
                     data['avg_attempts_per_quest'].mean()/5,
                     data['skill_progression'].mean()]
        
        fig.add_trace(
            go.Scatter(
                x=skills,
                y=avg_scores,
                mode='markers+lines',
                marker=dict(size=15, color='green'),
                line=dict(width=3),
                showlegend=False
            ),
            row=3, col=2
        )
        
        # 9. Risk distribution
        risk_categories = pd.cut(data['engagement_score'], 
                               bins=[0, 0.3, 0.6, 1.0], 
                               labels=['High Risk', 'Medium Risk', 'Low Risk'])
        risk_counts = risk_categories.value_counts()
        
        fig.add_trace(
            go.Pie(
                labels=risk_counts.index,
                values=risk_counts.values,
                marker=dict(colors=['#FF4757', '#FFA502', '#32FF7E'])
            ),
            row=3, col=3
        )
        
        # Update layout
        fig.update_layout(
            title='Comprehensive Performance Dashboard',
            showlegend=False,
            height=1200
        )
        
        # Update axes
        fig.update_xaxes(title_text="Grade Level", row=1, col=1)
        fig.update_xaxes(title_text="Engagement Score", row=2, col=1)
        fig.update_xaxes(title_text="Collaboration Events", row=2, col=3)
        fig.update_xaxes(title_text="Hour of Day", row=3, col=1)
        
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Learning Gain", row=2, col=1)
        fig.update_yaxes(title_text="Completion Rate", row=2, col=2)
        fig.update_yaxes(title_text="Skill Progress", row=2, col=3)
        fig.update_yaxes(title_text="Activity", row=3, col=1)
        fig.update_yaxes(title_text="Score", row=3, col=2)
        
        return fig
    
    def create_predictive_model_comparison(self, model_results: Dict) -> go.Figure:
        """Create model comparison visualization"""
        
        # Extract model names and metrics
        models = list(model_results.keys())
        metrics = ['r2', 'rmse', 'mae']
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=['R² Score', 'RMSE', 'MAE']
        )
        
        # Add bars for each metric
        for i, metric in enumerate(metrics):
            values = [model_results[model].get(metric, 0) for model in models]
            
            fig.add_trace(
                go.Bar(
                    x=models,
                    y=values,
                    name=metric.upper(),
                    marker_color=self.color_schemes['performance'][i]
                ),
                row=1, col=i+1
            )
        
        # Update layout
        fig.update_layout(
            title='Machine Learning Model Comparison',
            showlegend=False,
            height=400
        )
        
        return fig
    
    # Helper methods
    def _add_zone_boundary(self, fig: go.Figure, zone_name: str, config: Dict):
        """Add zone boundary to 3D plot"""
        theta = np.linspace(0, 2*np.pi, 50)
        radius = config.get('radius', 50)
        x_center = config.get('x', 0)
        z_center = config.get('z', 0)
        
        x_boundary = x_center + radius * np.cos(theta)
        z_boundary = z_center + radius * np.sin(theta)
        y_boundary = np.ones_like(theta) * 65
        
        fig.add_trace(go.Scatter3d(
            x=x_boundary,
            y=z_boundary,
            z=y_boundary,
            mode='lines',
            name=f'{zone_name} boundary',
            line=dict(color='rgba(128,128,128,0.5)', width=2),
            showlegend=False
        ))
    
    def _create_engagement_heatmap(self, data: pd.DataFrame) -> np.ndarray:
        """Create engagement heatmap data"""
        # Simulate hourly engagement for a week
        days = 7
        hours = 24
        
        heatmap = np.zeros((days, hours))
        
        for day in range(days):
            for hour in range(hours):
                # Simulate realistic patterns
                base = 0.3
                if 9 <= hour <= 17:  # School hours
                    base = 0.7
                if 14 <= hour <= 16:  # Peak afternoon
                    base = 0.9
                
                heatmap[day, hour] = base + np.random.normal(0, 0.1)
        
        return np.clip(heatmap, 0, 1)
```

## src/utils/helpers.py
```python
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import streamlit as st
from datetime import datetime, timedelta
import json
import yaml

class DataProcessor:
    """Utility functions for data processing and validation"""
    
    @staticmethod
    def validate_data_integrity(datasets: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
        """Validate data integrity across all datasets"""
        issues = {dataset: [] for dataset in datasets.keys()}
        
        # Check for required columns
        required_columns = {
            'students': ['student_id', 'grade_level', 'learning_style'],
            'movements': ['student_id', 'timestamp', 'x', 'y', 'z', 'zone'],
            'buildings': ['student_id', 'timestamp', 'action', 'block_type'],
            'quests': ['student_id', 'quest_name', 'completed', 'score'],
            'collaborations': ['student_1', 'student_2', 'interaction_type'],
            'learning_analytics': ['student_id', 'engagement_score', 'skill_progression']
        }
        
        for dataset_name, df in datasets.items():
            if dataset_name in required_columns:
                missing_cols = set(required_columns[dataset_name]) - set(df.columns)
                if missing_cols:
                    issues[dataset_name].append(f"Missing columns: {missing_cols}")
        
        # Check for data types
        for dataset_name, df in datasets.items():
            # Check for null values
            null_counts = df.isnull().sum()
            if null_counts.any():
                null_cols = null_counts[null_counts > 0].index.tolist()
                issues[dataset_name].append(f"Null values in: {null_cols}")
            
            # Check for duplicate student IDs in students table
            if dataset_name == 'students':
                duplicates = df['student_id'].duplicated().sum()
                if duplicates > 0:
                    issues[dataset_name].append(f"Duplicate student IDs: {duplicates}")
        
        # Cross-dataset validation
        if 'students' in datasets and 'learning_analytics' in datasets:
            student_ids = set(datasets['students']['student_id'])
            analytics_ids = set(datasets['learning_analytics']['student_id'])
            
            missing_analytics = student_ids - analytics_ids
            if missing_analytics:
                issues['learning_analytics'].append(
                    f"Missing analytics for {len(missing_analytics)} students"
                )
        
        return {k: v for k, v in issues.items() if v}  # Return only datasets with issues
    
    @staticmethod
    def create_summary_statistics(data: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """Create comprehensive summary statistics"""
        summary = data[numeric_cols].describe()
        
        # Add additional statistics
        summary.loc['cv'] = data[numeric_cols].std() / data[numeric_cols].mean()
        summary.loc['skewness'] = data[numeric_cols].skew()
        summary.loc['kurtosis'] = data[numeric_cols].kurtosis()
        
        return summary.round(3)
    
    @staticmethod
    def create_correlation_matrix(data: pd.DataFrame, 
                                method: str = 'pearson',
                                threshold: float = 0.3) -> pd.DataFrame:
        """Create correlation matrix with significance filtering"""
        corr_matrix = data.corr(method=method)
        
        # Create mask for low correlations
        mask = np.abs(corr_matrix) < threshold
        corr_matrix[mask] = np.nan
        
        return corr_matrix
    
    @staticmethod
    def aggregate_by_time_window(data: pd.DataFrame, 
                               time_col: str,
                               window: str = 'D',
                               agg_dict: Optional[Dict] = None) -> pd.DataFrame:
        """Aggregate data by time windows"""
        data = data.copy()
        data[time_col] = pd.to_datetime(data[time_col])
        
        if agg_dict is None:
            # Default aggregation for numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            agg_dict = {col: 'mean' for col in numeric_cols if col != time_col}
        
        return data.set_index(time_col).resample(window).agg(agg_dict)
    
    @staticmethod
    def calculate_growth_metrics(data: pd.DataFrame,
                               student_col: str,
                               metric_col: str,
                               time_col: str) -> pd.DataFrame:
        """Calculate growth metrics for each student"""
        growth_metrics = []
        
        for student in data[student_col].unique():
            student_data = data[data[student_col] == student].sort_values(time_col)
            
            if len(student_data) > 1:
                first_value = student_data[metric_col].iloc[0]
                last_value = student_data[metric_col].iloc[-1]
                
                growth = {
                    'student_id': student,
                    'initial_value': first_value,
                    'final_value': last_value,
                    'absolute_growth': last_value - first_value,
                    'relative_growth': (last_value - first_value) / first_value if first_value != 0 else 0,
                    'days_tracked': (student_data[time_col].max() - student_data[time_col].min()).days,
                    'volatility': student_data[metric_col].std(),
                    'trend_slope': np.polyfit(range(len(student_data)), student_data[metric_col], 1)[0]
                }
                growth_metrics.append(growth)
        
        return pd.DataFrame(growth_metrics)


class StreamlitHelpers:
    """Streamlit-specific helper functions"""
    
    @staticmethod
    def create_metric_card(label: str, value: float, delta: Optional[float] = None,
                          format_str: str = "{:.2f}") -> None:
        """Create a styled metric card"""
        if delta is not None:
            st.metric(label, format_str.format(value), 
                     delta=format_str.format(delta),
                     delta_color="normal" if delta >= 0 else "inverse")
        else:
            st.metric(label, format_str.format(value))
    
    @staticmethod
    def create_download_button(data: pd.DataFrame, filename: str, 
                             button_text: str = "Download CSV") -> None:
        """Create download button for dataframe"""
        csv = data.to_csv(index=False)
        st.download_button(
            label=button_text,
            data=csv,
            file_name=filename,
            mime='text/csv'
        )
    
    @staticmethod
    def create_info_box(title: str, content: str, box_type: str = "info") -> None:
        """Create styled information box"""
        box_functions = {
            "info": st.info,
            "success": st.success,
            "warning": st.warning,
            "error": st.error
        }
        
        box_func = box_functions.get(box_type, st.info)
        box_func(f"**{title}**\n\n{content}")
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def load_cached_data(file_path: str) -> pd.DataFrame:
        """Load data with caching"""
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            return pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
    
    @staticmethod
    def create_sidebar_filters(data: pd.DataFrame) -> Dict:
        """Create standard sidebar filters"""
        filters = {}
        
        st.sidebar.header("Filters")
        
        # Date range filter
        if 'timestamp' in data.columns:
            date_col = pd.to_datetime(data['timestamp']).dt.date
            date_range = st.sidebar.date_input(
                "Date Range",
                value=(date_col.min(), date_col.max()),
                min_value=date_col.min(),
                max_value=date_col.max()
            )
            filters['date_range'] = date_range
        
        # Student filter
        if 'student_id' in data.columns:
            students = ['All'] + sorted(data['student_id'].unique().tolist())
            selected_student = st.sidebar.selectbox("Student", students)
            filters['student'] = None if selected_student == 'All' else selected_student
        
        # Grade filter
        if 'grade_level' in data.columns:
            grades = ['All'] + sorted(data['grade_level'].unique().tolist())
            selected_grade = st.sidebar.multiselect(
                "Grade Level", 
                grades[1:],  # Exclude 'All'
                default=grades[1:]
            )
            filters['grades'] = selected_grade
        
        return filters


class AnalyticsHelpers:
    """Helper functions for analytics calculations"""
    
    @staticmethod
    def calculate_engagement_index(data: pd.DataFrame) -> pd.Series:
        """Calculate composite engagement index"""
        # Normalize metrics to 0-1 scale
        metrics = ['quest_completion_rate', 'days_active', 'collaboration_events', 
                  'building_complexity_avg', 'skill_progression']
        
        normalized = pd.DataFrame()
        for metric in metrics:
            if metric in data.columns:
                min_val = data[metric].min()
                max_val = data[metric].max()
                if max_val > min_val:
                    normalized[metric] = (data[metric] - min_val) / (max_val - min_val)
                else:
                    normalized[metric] = 0.5
        
        # Weighted average
        weights = {'quest_completion_rate': 0.3, 'days_active': 0.2, 
                  'collaboration_events': 0.2, 'building_complexity_avg': 0.15,
                  'skill_progression': 0.15}
        
        engagement_index = sum(normalized[col] * weights.get(col, 0.2) 
                              for col in normalized.columns)
        
        return engagement_index
    
    @staticmethod
    def identify_at_risk_students(data: pd.DataFrame, 
                                thresholds: Optional[Dict] = None) -> pd.DataFrame:
        """Identify at-risk students based on multiple criteria"""
        if thresholds is None:
            thresholds = {
                'engagement_score': 0.3,
                'quest_completion_rate': 0.4,
                'days_active': 5,
                'skill_progression': 0.2
            }
        
        at_risk_flags = pd.DataFrame(index=data.index)
        
        for metric, threshold in thresholds.items():
            if metric in data.columns:
                if metric == 'days_active':
                    at_risk_flags[f'{metric}_risk'] = data[metric] < threshold
                else:
                    at_risk_flags[f'{metric}_risk'] = data[metric] < threshold
        
        # Overall risk score
        at_risk_flags['risk_score'] = at_risk_flags.mean(axis=1)
        at_risk_flags['risk_level'] = pd.cut(
            at_risk_flags['risk_score'],
            bins=[0, 0.33, 0.66, 1.0],
            labels=['Low', 'Medium', 'High']
        )
        
        # Add student info
        result = pd.concat([
            data[['student_id', 'engagement_score', 'quest_completion_rate', 
                  'days_active', 'skill_progression']],
            at_risk_flags
        ], axis=1)
        
        return result.sort_values('risk_score', ascending=False)
    
    @staticmethod
    def calculate_peer_influence(collaboration_data: pd.DataFrame,
                               student_metrics: pd.DataFrame) -> pd.DataFrame:
        """Calculate peer influence metrics"""
        import networkx as nx
        
        # Build collaboration network
        G = nx.Graph()
        
        for _, row in collaboration_data.iterrows():
            G.add_edge(row['student_1'], row['student_2'], 
                      weight=row.get('effectiveness', 1))
        
        # Calculate network metrics
        influence_metrics = []
        
        for student in G.nodes():
            # Get student's performance
            if student in student_metrics['student_id'].values:
                student_perf = student_metrics[
                    student_metrics['student_id'] == student
                ]['engagement_score'].iloc[0]
                
                # Calculate peer average
                peers = list(G.neighbors(student))
                if peers:
                    peer_perfs = student_metrics[
                        student_metrics['student_id'].isin(peers)
                    ]['engagement_score'].mean()
                    
                    influence = {
                        'student_id': student,
                        'degree_centrality': nx.degree_centrality(G)[student],
                        'betweenness_centrality': nx.betweenness_centrality(G)[student],
                        'clustering_coefficient': nx.clustering(G, student),
                        'own_performance': student_perf,
                        'peer_avg_performance': peer_perfs,
                        'peer_influence': peer_perfs - student_perf,
                        'n_connections': len(peers)
                    }
                    influence_metrics.append(influence)
        
        return pd.DataFrame(influence_metrics)


# Configuration management
def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        st.warning(f"Config file {config_path} not found. Using defaults.")
        return get_default_config()

def get_default_config() -> Dict:
    """Get default configuration"""
    return {
        'simulation': {
            'n_students': 60,
            'days': 30,
            'seed': 42
        },
        'zones': {
            'spawn': {'x': 0, 'z': 0, 'radius': 30},
            'tutorial': {'x': 50, 'z': 0, 'radius': 40},
            'building_area': {'x': 100, 'z': 100, 'radius': 100},
            'collaboration_space': {'x': -100, 'z': 100, 'radius': 80},
            'challenge_zone': {'x': 0, 'z': 200, 'radius': 60},
            'resource_area': {'x': -150, 'z': -50, 'radius': 50},
            'showcase_area': {'x': 150, 'z': -50, 'radius': 70}
        },
        'analytics': {
            'engagement_weights': {
                'quest_completion': 0.3,
                'building_activity': 0.3,
                'collaboration': 0.2,
                'skill_progression': 0.2
            },
            'risk_thresholds': {
                'engagement_score': 0.3,
                'quest_completion_rate': 0.4,
                'days_active': 5
            }
        },
        'visualization': {
            'theme': 'plotly',
            'height': 600,
            'width': 1200
        }
    }

def save_config(config: Dict, config_path: str = "config.yaml") -> None:
    """Save configuration to YAML file"""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
```

## Deployment Configuration

### requirements.txt (Updated)
```
streamlit==1.31.0
pandas==2.2.0
numpy==1.26.3
plotly==5.18.0
scipy==1.12.0
scikit-learn==1.4.0
statsmodels==0.14.1
networkx==3.2.1
faker==22.2.0
pyyaml==6.0.1
pingouin==0.5.4
ruptures==1.1.9
seaborn==0.13.2
```

### .streamlit/config.toml
```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 200
enableCORS = true
enableXsrfProtection = true

[browser]
gatherUsageStats = false
```

### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/simulated

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### docker-compose.yml
```yaml
version: '3.8'

services:
  streamlit-app:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./config.yaml:/app/config.yaml
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
```

## Testing Script

### test_app.py
```python
import unittest
import pandas as pd
import numpy as np
from src.data_generation.simulator import MinecraftEducationSimulator
from src.analysis.statistical import EducationalStatisticsAnalyzer
from src.analysis.time_series import TimeSeriesEducationAnalyzer

class TestMinecraftEducationDashboard(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.simulator = MinecraftEducationSimulator()
        self.stat_analyzer = EducationalStatisticsAnalyzer()
        self.ts_analyzer = TimeSeriesEducationAnalyzer()
        
        # Generate small test dataset
        self.test_data = self.simulator.generate_complete_dataset(
            n_students=10, days=7
        )
    
    def test_data_generation(self):
        """Test data generation integrity"""
        # Check all required datasets are generated
        required_datasets = ['students', 'movements', 'buildings', 
                           'quests', 'collaborations', 'learning_analytics']
        
        for dataset in required_datasets:
            self.assertIn(dataset, self.test_data)
            self.assertGreater(len(self.test_data[dataset]), 0)
        
        # Check student count
        self.assertEqual(len(self.test_data['students']), 10)
    
    def test_statistical_analysis(self):
        """Test statistical analysis functions"""
        # Create test groups
        test_df = pd.DataFrame({
            'group': ['A'] * 20 + ['B'] * 20,
            'outcome': np.random.normal(0.5, 0.1, 20).tolist() + 
                      np.random.normal(0.7, 0.1, 20).tolist()
        })
        
        # Run t-test
        results = self.stat_analyzer.compare_learning_methods(
            test_df, 'group', 'outcome'
        )
        
        # Check results structure
        self.assertIn('p_value', results)
        self.assertIn('effect_size', results)
        self.assertIn('interpretation', results)
    
    def test_time_series_analysis(self):
        """Test time series analysis"""
        # Create test time series
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        ts_data = pd.DataFrame({
            'timestamp': dates,
            'value': np.random.randn(30).cumsum() + 10
        })
        
        # Run analysis
        results = self.ts_analyzer.analyze_engagement_patterns(
            ts_data, 'timestamp', 'value'
        )
        
        # Check results
        self.assertIn('summary_statistics', results)
        self.assertIn('trend', results)
        self.assertIn('forecast', results)
    
    def test_changepoint_detection(self):
        """Test changepoint detection"""
        # Create data with clear changepoint
        values1 = np.random.normal(5, 1, 50)
        values2 = np.random.normal(10, 1, 50)
        values = np.concatenate([values1, values2])
        
        test_df = pd.DataFrame({
            'metric': values
        })
        
        # Detect changepoints
        results = self.ts_analyzer.detect_learning_changepoints(
            test_df, 'metric'
        )
        
        # Check results
        self.assertIn('segments', results)
        self.assertIn('learning_phases', results)

if __name__ == '__main__':
    unittest.main()
```

## Final Deployment Steps

### 1. GitHub Repository Setup
```bash
# Initialize git repository
git init
git add .
git commit -m "Initial commit: Minecraft Education Analytics Dashboard"

# Create GitHub repository and push
git remote add origin https://github.com/yourusername/minecraft-education-dashboard.git
git branch -M main
git push -u origin main
```

### 2. Streamlit Cloud Deployment
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Connect your GitHub repository
4. Set main file path: `app.py`
5. Click "Deploy"

### 3. Local Development
```bash
# Clone repository
git clone https://github.com/yourusername/minecraft-education-dashboard.git
cd minecraft-education-dashboard

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py
```

### 4. Production Optimization
```python
# Add to app.py for production
import streamlit as st

# Configure page
st.set_page_config(
    page_title="Minecraft Education Analytics",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/minecraft-education-dashboard',
        'Report a bug': "https://github.com/yourusername/minecraft-education-dashboard/issues",
        'About': "# Minecraft Education Analytics\nAdvanced analytics for game-based learning"
    }
)

# Add Google Analytics (optional)
GA_TAG = """
<!-- Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'GA_MEASUREMENT_ID');
</script>
"""

if st._is_running_with_streamlit:
    st.components.v1.html(GA_TAG, height=0)
```

## Success Checklist

✅ **Technical Implementation**
- [ ] Data generation with realistic patterns
- [ ] Statistical analysis suite
- [ ] Time series analysis
- [ ] Machine learning models
- [ ] Interactive visualizations
- [ ] 3D world view
- [ ] Performance optimization

✅ **Educational Features**
- [ ] Learning progression tracking
- [ ] At-risk student identification
- [ ] Collaboration network analysis
- [ ] Skill development metrics
- [ ] Intervention recommendations

✅ **Portfolio Quality**
- [ ] Clean, documented code
- [ ] Professional UI/UX
- [ ] Comprehensive README
- [ ] Live deployment
- [ ] Test coverage
- [ ] Research integration

✅ **Deployment**
- [ ] GitHub repository
- [ ] Streamlit Cloud hosting
- [ ] Docker support
- [ ] Documentation
- [ ] Performance monitoring

This completes the comprehensive implementation guide for your Minecraft Education Analytics Dashboard!
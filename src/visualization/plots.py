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

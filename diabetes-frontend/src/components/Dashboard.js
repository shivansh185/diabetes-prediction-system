import React, { useEffect, useState } from 'react';
import axios from 'axios';
import {
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer
} from 'recharts';
import './Dashboard.css';

const Dashboard = () => {
  const [performance, setPerformance] = useState(null);
  const [featureImportance, setFeatureImportance] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      const [perfRes, featRes] = await Promise.all([
        axios.get('http://localhost:5000/api/model-performance'),
        axios.get('http://localhost:5000/api/feature-importance')
      ]);
      
      setPerformance(perfRes.data);
      setFeatureImportance(featRes.data);
      setLoading(false);
    } catch (error) {
      console.error('Error fetching data:', error);
      setLoading(false);
    }
  };

  if (loading) {
    return <div className="loading">Loading dashboard data...</div>;
  }

  // Prepare data for charts
  const modelComparisonData = performance ? [
    { name: 'Ensemble', Accuracy: performance.ensemble.accuracy, Precision: performance.ensemble.precision, Recall: performance.ensemble.recall, F1: performance.ensemble.f1 },
    { name: 'XGBoost', Accuracy: performance.xgb.accuracy, Precision: performance.xgb.precision, Recall: performance.xgb.recall, F1: performance.xgb.f1 },
    { name: 'MLP', Accuracy: performance.mlp.accuracy, Precision: performance.mlp.precision, Recall: performance.mlp.recall, F1: performance.mlp.f1 }
  ] : [];

  const aucData = performance ? [
    { name: 'Ensemble', AUC: performance.ensemble.auc },
    { name: 'XGBoost', AUC: performance.xgb.auc },
    { name: 'MLP', AUC: performance.mlp.auc }
  ] : [];

  const COLORS = ['#4CAF50', '#FFC107', '#F44336', '#2196F3'];

  return (
    <div className="dashboard">
      <h1>Model Performance Dashboard</h1>
      
      <div className="metrics-grid">
        <div className="metric-card">
          <h3>Ensemble Model</h3>
          <div className="metric-value">{performance?.ensemble.accuracy}%</div>
          <div className="metric-label">Accuracy</div>
        </div>
        
        <div className="metric-card">
          <h3>Ensemble Model</h3>
          <div className="metric-value">{performance?.ensemble.precision}%</div>
          <div className="metric-label">Precision</div>
        </div>
        
        <div className="metric-card">
          <h3>Ensemble Model</h3>
          <div className="metric-value">{performance?.ensemble.recall}%</div>
          <div className="metric-label">Recall</div>
        </div>
        
        <div className="metric-card">
          <h3>Ensemble Model</h3>
          <div className="metric-value">{performance?.ensemble.f1}%</div>
          <div className="metric-label">F1 Score</div>
        </div>
      </div>
      
      <div className="charts-grid">
        <div className="chart-card">
          <h3>Model Performance Comparison</h3>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={modelComparisonData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="Accuracy" fill="#4CAF50" />
              <Bar dataKey="Precision" fill="#2196F3" />
              <Bar dataKey="Recall" fill="#FFC107" />
              <Bar dataKey="F1" fill="#F44336" />
            </BarChart>
          </ResponsiveContainer>
        </div>
        
        <div className="chart-card">
          <h3>AUC-ROC Scores</h3>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={aucData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis domain={[0, 1]} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="AUC" stroke="#4CAF50" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        
        {featureImportance && (
          <div className="chart-card">
            <h3>Feature Importance (XGBoost)</h3>
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={featureImportance} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis dataKey="feature" type="category" width={150} />
                <Tooltip />
                <Bar dataKey="importance" fill="#4CAF50" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;
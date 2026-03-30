import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './ChartDisplay.css';

const ChartDisplay = () => {
  const [charts, setCharts] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeChart, setActiveChart] = useState('main');

  useEffect(() => {
    fetchCharts();
  }, []);

  const fetchCharts = async () => {
    try {
      setLoading(true);
      const response = await axios.get('http://localhost:5000/api/charts');
      
      if (response.data.success) {
        setCharts(response.data.charts);
      } else {
        setError(response.data.error);
      }
    } catch (error) {
      console.error('Error fetching charts:', error);
      setError('Failed to load charts. Please make sure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="chart-loading">
        <div className="spinner"></div>
        <p>Loading charts...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="chart-error">
        <h3>Error Loading Charts</h3>
        <p>{error}</p>
        <button onClick={fetchCharts} className="retry-btn">
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="chart-container">
      <h2>Model Performance Visualization</h2>
      
      <div className="chart-tabs">
        <button 
          className={`tab-btn ${activeChart === 'main' ? 'active' : ''}`}
          onClick={() => setActiveChart('main')}
        >
          Model Analysis
        </button>
        <button 
          className={`tab-btn ${activeChart === 'comparison' ? 'active' : ''}`}
          onClick={() => setActiveChart('comparison')}
        >
          Model Comparison
        </button>
      </div>
      
      <div className="chart-display">
        {activeChart === 'main' && charts?.main_chart && (
          <div className="chart-card">
            <div className="chart-image-container">
              <img 
                src={`data:image/png;base64,${charts.main_chart}`} 
                alt="Model Performance Charts"
                className="chart-image"
              />
            </div>
          </div>
        )}
        
        {activeChart === 'comparison' && charts?.model_comparison && (
          <div className="chart-card">
            <div className="chart-image-container">
              <img 
                src={`data:image/png;base64,${charts.model_comparison}`} 
                alt="Model Comparison Chart"
                className="chart-image"
              />
            </div>
          </div>
        )}
      </div>
      
      <div className="chart-info">
        <h3>Understanding the Charts</h3>
        <div className="info-grid">
          <div className="info-item">
            <strong>Confusion Matrix</strong>
            <p>Shows true vs predicted values. Diagonal cells represent correct predictions.</p>
          </div>
          <div className="info-item">
            <strong>ROC Curve</strong>
            <p>Shows model's ability to distinguish between classes. Higher AUC indicates better performance.</p>
          </div>
          <div className="info-item">
            <strong>Precision-Recall Curve</strong>
            <p>Useful for imbalanced datasets. Shows trade-off between precision and recall.</p>
          </div>
          <div className="info-item">
            <strong>Probability Distribution</strong>
            <p>Shows how confident the model is in its predictions for each class.</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChartDisplay;
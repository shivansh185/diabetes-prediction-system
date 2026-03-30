import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useInView } from 'react-intersection-observer';
import { gsap } from 'gsap';
import axios from 'axios';
import confetti from 'canvas-confetti';
import { toast } from 'react-toastify';
import { 
  FaUserMd, FaStethoscope, FaHeartbeat, FaWeight, FaRuler, 
  FaCalendarAlt, FaVial, FaTint, FaMicroscope, FaChartLine 
} from 'react-icons/fa';
import './PredictionForm.css';

const PredictionForm = () => {
  const [formData, setFormData] = useState({
    pregnancies: '',
    glucose: '',
    bloodPressure: '',
    skinThickness: '',
    insulin: '',
    bmi: '',
    dpf: '',
    age: ''
  });
  
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [mlResults, setMlResults] = useState(null);
  const formRef = useRef(null);
  const resultRef = useRef(null);
  const [ref] = useInView({
    triggerOnce: true,
    threshold: 0.1,
  });

  useEffect(() => {
    // GSAP animation for form entrance
    if (formRef.current) {
      gsap.fromTo(formRef.current,
        { opacity: 0, y: 50 },
        { opacity: 1, y: 0, duration: 0.8, ease: "power3.out" }
      );
    }
  }, []);

  useEffect(() => {
    // Animate result when it appears
    if (result && resultRef.current) {
      gsap.fromTo(resultRef.current,
        { opacity: 0, scale: 0.9, y: 20 },
        { opacity: 1, scale: 1, y: 0, duration: 0.6, ease: "back.out(1.2)" }
      );
    }
  }, [result]);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const triggerConfetti = (probability) => {
    // Basic confetti
    confetti({
      particleCount: 100,
      spread: 70,
      origin: { y: 0.6 },
      colors: ['#4CAF50', '#667eea', '#764ba2']
    });
    
    // Extra fireworks for high confidence predictions
    if (probability > 80) {
      setTimeout(() => {
        confetti({
          particleCount: 200,
          spread: 100,
          origin: { y: 0.5 },
          startVelocity: 25,
          decay: 0.9,
          colors: ['#ff6b6b', '#4CAF50', '#667eea', '#ffd93d']
        });
      }, 200);
      
      // Second burst
      setTimeout(() => {
        confetti({
          particleCount: 150,
          spread: 80,
          origin: { y: 0.4, x: 0.3 },
          startVelocity: 30,
        });
        confetti({
          particleCount: 150,
          spread: 80,
          origin: { y: 0.4, x: 0.7 },
          startVelocity: 30,
        });
      }, 500);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Validate inputs with animation
    for (let key in formData) {
      if (formData[key] === '') {
        toast.error(`Please fill in ${key.replace(/([A-Z])/g, ' $1').toLowerCase()} field`);
        // Shake animation for empty field
        const input = document.querySelector(`input[name="${key}"]`);
        if (input) {
          gsap.to(input, {
            x: [0, -10, 10, -5, 5, 0],
            duration: 0.5,
            ease: "power2.out"
          });
        }
        return;
      }
      if (isNaN(formData[key]) || formData[key] < 0) {
        toast.error(`Please enter a valid positive number for ${key.replace(/([A-Z])/g, ' $1').toLowerCase()}`);
        return;
      }
    }
    
    setLoading(true);
    
    try {
      const response = await axios.post('http://localhost:5000/api/predict', formData, {
        headers: { 'Content-Type': 'application/json' },
        timeout: 10000
      });
      
      if (response.data.success) {
        setResult(response.data);
        setMlResults(response.data.ml_predictions);
        toast.success('Prediction completed successfully! 🎉');
        
        // Trigger confetti celebration
        triggerConfetti(response.data.probability);
      } else {
        toast.error(response.data.error || 'Error making prediction');
      }
    } catch (error) {
      console.error('Error:', error);
      toast.error('Cannot connect to server. Make sure the backend is running on port 5000');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    // GSAP animation for reset
    gsap.to(formRef.current, {
      scale: 0.98,
      duration: 0.2,
      yoyo: true,
      repeat: 1,
      onComplete: () => {
        setFormData({
          pregnancies: '',
          glucose: '',
          bloodPressure: '',
          skinThickness: '',
          insulin: '',
          bmi: '',
          dpf: '',
          age: ''
        });
        setResult(null);
        setMlResults(null);
        toast.info('Form has been reset');
      }
    });
  };

  const inputFields = [
    { name: 'pregnancies', label: 'Pregnancies', icon: FaUserMd, placeholder: 'Number of pregnancies', min: 0, max: 20, step: 1 },
    { name: 'glucose', label: 'Glucose Level', icon: FaTint, placeholder: 'mg/dL', min: 0, max: 300, step: 1, unit: 'mg/dL' },
    { name: 'bloodPressure', label: 'Blood Pressure', icon: FaHeartbeat, placeholder: 'mm Hg', min: 0, max: 150, step: 1, unit: 'mm Hg' },
    { name: 'skinThickness', label: 'Skin Thickness', icon: FaRuler, placeholder: 'mm', min: 0, max: 100, step: 1, unit: 'mm' },
    { name: 'insulin', label: 'Insulin Level', icon: FaVial, placeholder: 'mu U/ml', min: 0, max: 500, step: 1, unit: 'mu U/ml' },
    { name: 'bmi', label: 'BMI', icon: FaWeight, placeholder: 'kg/m²', min: 10, max: 50, step: 0.1, unit: 'kg/m²' },
    { name: 'dpf', label: 'Diabetes Pedigree', icon: FaMicroscope, placeholder: 'DPF value', min: 0, max: 2.5, step: 0.001 },
    { name: 'age', label: 'Age', icon: FaCalendarAlt, placeholder: 'Years', min: 0, max: 120, step: 1, unit: 'years' }
  ];

  return (
    <div className="prediction-container">
      <motion.div
        ref={formRef}
        className="form-section"
        initial={{ opacity: 0, y: 50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <motion.div 
          className="form-header"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
        >
          <FaStethoscope className="header-icon" />
          <h2>Patient Health Assessment</h2>
          <p>Enter patient information for AI-powered diabetes prediction</p>
        </motion.div>

        <form onSubmit={handleSubmit}>
          <div className="form-grid">
            {inputFields.map((field, index) => (
              <motion.div
                key={field.name}
                className="form-group"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.05 }}
                whileHover={{ scale: 1.02 }}
              >
                <label>
                  <field.icon className="input-icon" />
                  {field.label}
                  {field.unit && <span className="unit">({field.unit})</span>}
                </label>
                <input
                  type="number"
                  name={field.name}
                  value={formData[field.name]}
                  onChange={handleChange}
                  required
                  step={field.step}
                  min={field.min}
                  max={field.max}
                  placeholder={field.placeholder}
                />
                <motion.div
                  className="input-focus"
                  initial={{ scaleX: 0 }}
                  whileFocus={{ scaleX: 1 }}
                  transition={{ duration: 0.3 }}
                />
              </motion.div>
            ))}
          </div>
          
          <motion.div 
            className="button-group"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
          >
            <motion.button
              type="submit"
              disabled={loading}
              className="predict-btn"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              animate={loading ? { scale: [1, 1.02, 1] } : {}}
              transition={{ repeat: loading ? Infinity : 0, duration: 0.5 }}
            >
              {loading ? (
                <>
                  <span className="spinner"></span>
                  Analyzing Data...
                </>
              ) : (
                <>
                  <FaChartLine /> Predict Diabetes
                </>
              )}
            </motion.button>
            
            <motion.button
              type="button"
              onClick={handleReset}
              className="reset-btn"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              Reset Form
            </motion.button>
          </motion.div>
        </form>
      </motion.div>
      
      <AnimatePresence>
        {result && (
          <motion.div
            ref={resultRef}
            className="results-section"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            transition={{ type: "spring", stiffness: 200, damping: 20 }}
          >
            <motion.div 
              className={`result-card ${result.prediction === 'Diabetic' ? 'diabetic' : 'non-diabetic'}`}
              initial={{ backgroundPosition: "0% 50%" }}
              animate={{ backgroundPosition: "100% 50%" }}
              transition={{ duration: 0.5 }}
            >
              <h3>
                {result.prediction === 'Diabetic' ? '⚠️ Diabetes Detected' : '✅ Healthy Assessment'}
              </h3>
              <div className="prediction-value">{result.prediction}</div>
              <div className="probability">
                Confidence: {result.probability}%
                <motion.div 
                  className="confidence-bar"
                  initial={{ width: 0 }}
                  animate={{ width: `${result.probability}%` }}
                  transition={{ duration: 1, ease: "easeOut" }}
                />
              </div>
              <div className="recommendation">
                <FaStethoscope className="rec-icon" />
                {result.recommendation}
              </div>
            </motion.div>
            
            <motion.div 
              className="ensemble-results"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
            >
              <h3>Ensemble Model Analysis</h3>
              <div className="model-details">
                <motion.div 
                  className="model-item"
                  whileHover={{ scale: 1.05 }}
                  transition={{ type: "spring", stiffness: 400 }}
                >
                  <strong>XGBoost:</strong>
                  <div className="model-confidence">
                    {result.xgb_probability}% confidence
                    <div className="mini-bar">
                      <motion.div 
                        className="mini-fill"
                        initial={{ width: 0 }}
                        animate={{ width: `${result.xgb_probability}%` }}
                        transition={{ duration: 0.8 }}
                      />
                    </div>
                  </div>
                </motion.div>
                <motion.div 
                  className="model-item"
                  whileHover={{ scale: 1.05 }}
                  transition={{ type: "spring", stiffness: 400 }}
                >
                  <strong>MLP Neural Network:</strong>
                  <div className="model-confidence">
                    {result.mlp_probability}% confidence
                    <div className="mini-bar">
                      <motion.div 
                        className="mini-fill"
                        initial={{ width: 0 }}
                        animate={{ width: `${result.mlp_probability}%` }}
                        transition={{ duration: 0.8, delay: 0.2 }}
                      />
                    </div>
                  </div>
                </motion.div>
              </div>
            </motion.div>
            
            {mlResults && Object.keys(mlResults).length > 0 && (
              <motion.div 
                className="ml-comparison"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.4 }}
              >
                <h3>Individual Model Predictions</h3>
                <div className="model-grid">
                  {Object.entries(mlResults).map(([model, data], index) => (
                    <motion.div
                      key={model}
                      className="model-card"
                      initial={{ opacity: 0, scale: 0.9 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ delay: index * 0.1 }}
                      whileHover={{ scale: 1.05, y: -5 }}
                    >
                      <h4>{model}</h4>
                      <div className={`prediction ${data.prediction === 'Diabetic' ? 'diabetic-text' : 'non-diabetic-text'}`}>
                        {data.prediction}
                      </div>
                      <div className="confidence">{data.probability}% confidence</div>
                    </motion.div>
                  ))}
                </div>
              </motion.div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default PredictionForm;
import React, { useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { ToastContainer } from 'react-toastify';
import { motion, AnimatePresence } from 'framer-motion';
import { FaHeartbeat, FaChartLine, FaChartBar, FaRobot, FaBrain, FaMicroscope } from 'react-icons/fa';
import PredictionForm from './components/PredictionForm';
import Dashboard from './components/Dashboard';
import ChartDisplay from './components/ChartDisplay';
import 'react-toastify/dist/ReactToastify.css';
import './App.css';

// Animated page transitions
const pageVariants = {
  initial: {
    opacity: 0,
    y: 20,
  },
  in: {
    opacity: 1,
    y: 0,
  },
  out: {
    opacity: 0,
    y: -20,
  },
};

const pageTransition = {
  type: 'tween',
  ease: 'anticipate',
  duration: 0.5,
};

const AnimatedRoutes = () => {
  const location = useLocation();

  return (
    <AnimatePresence mode="wait">
      <Routes location={location} key={location.pathname}>
        <Route
          path="/"
          element={
            <motion.div
              initial="initial"
              animate="in"
              exit="out"
              variants={pageVariants}
              transition={pageTransition}
            >
              <PredictionForm />
            </motion.div>
          }
        />
        <Route
          path="/dashboard"
          element={
            <motion.div
              initial="initial"
              animate="in"
              exit="out"
              variants={pageVariants}
              transition={pageTransition}
            >
              <Dashboard />
            </motion.div>
          }
        />
        <Route
          path="/charts"
          element={
            <motion.div
              initial="initial"
              animate="in"
              exit="out"
              variants={pageVariants}
              transition={pageTransition}
            >
              <ChartDisplay />
            </motion.div>
          }
        />
      </Routes>
    </AnimatePresence>
  );
};

function App() {
  return (
    <Router>
      <div className="App">
        <Navbar />
        <div className="container">
          <AnimatedRoutes />
        </div>
        <ToastContainer 
          position="bottom-right" 
          autoClose={5000}
          theme="colored"
          toastClassName="custom-toast"
        />
        <FloatingParticles />
      </div>
    </Router>
  );
}

// Enhanced Navbar with animations
const Navbar = () => {
  const [scrolled, setScrolled] = React.useState(false);
  const location = useLocation();

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 50);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const navItems = [
    { path: '/', name: 'Predict', icon: <FaHeartbeat /> },
    { path: '/dashboard', name: 'Dashboard', icon: <FaChartLine /> },
    { path: '/charts', name: 'Charts', icon: <FaChartBar /> },
  ];

  return (
    <motion.nav
      className={`navbar ${scrolled ? 'scrolled' : ''}`}
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ type: 'spring', stiffness: 100, damping: 20 }}
    >
      <div className="nav-container">
        <motion.div
          className="logo"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <Link to="/">
            <FaRobot className="logo-icon" />
            <h1>
              Diabetes<span>Predictor</span>
              <motion.span
                className="ai-badge"
                initial={{ opacity: 0, scale: 0 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.5 }}
              >
                AI Powered
              </motion.span>
            </h1>
          </Link>
        </motion.div>

        <ul className="nav-menu">
          {navItems.map((item, index) => (
            <motion.li
              key={item.path}
              className="nav-item"
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
            >
              <Link
                to={item.path}
                className={`nav-link ${location.pathname === item.path ? 'active' : ''}`}
              >
                <span className="nav-icon">{item.icon}</span>
                <span className="nav-text">{item.name}</span>
                {location.pathname === item.path && (
                  <motion.div
                    className="active-indicator"
                    layoutId="activeIndicator"
                    transition={{ type: 'spring', stiffness: 300, damping: 30 }}
                  />
                )}
              </Link>
            </motion.li>
          ))}
        </ul>

        <motion.div
          className="ai-status"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
        >
          <FaBrain className="brain-icon" />
          <span className="status-text">ML Models Active</span>
          <span className="pulse-dot"></span>
        </motion.div>
      </div>
    </motion.nav>
  );
};

// Floating particles background effect
const FloatingParticles = () => {
  useEffect(() => {
    const createParticle = () => {
      const particle = document.createElement('div');
      particle.className = 'floating-particle';
      particle.style.left = Math.random() * 100 + '%';
      particle.style.animationDuration = Math.random() * 20 + 10 + 's';
      particle.style.animationDelay = Math.random() * 10 + 's';
      particle.style.opacity = Math.random() * 0.3;
      particle.style.width = Math.random() * 5 + 2 + 'px';
      particle.style.height = particle.style.width;
      document.body.appendChild(particle);
      
      setTimeout(() => {
        particle.remove();
      }, 20000);
    };
    
    const interval = setInterval(createParticle, 2000);
    return () => clearInterval(interval);
  }, []);
  
  return null;
};

export default App;
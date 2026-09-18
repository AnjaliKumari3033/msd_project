import React from 'react';
import { HashRouter, Routes, Route } from 'react-router-dom';
import './index.css';

import Navbar from './components/Navbar';
import Classification from './pages/Classification';
import Regression from './pages/Regression';
import Clustering from './pages/Clustering';

const App = () => {
  return (
    <HashRouter>
      <div className="app">
        <Navbar />
        <Routes>
          <Route path="/" element={<Classification />} />
          <Route path="/classification" element={<Classification />} />
          <Route path="/regression" element={<Regression />} />
          <Route path="/clustering" element={<Clustering />} />
        </Routes>
      </div>
    </HashRouter>
  );
};

export default App;

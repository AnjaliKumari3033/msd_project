import React from 'react';
import { NavLink } from 'react-router-dom';

const Navbar = () => {
  return (
    <nav className="navbar">
      <div className="container nav-container">
        <div className="nav-brand">Team 13: IMDB Sentiment Analysis</div>
        <div className="nav-links">
          <NavLink to="/" end className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}>
            Home
          </NavLink>
          <NavLink to="/classification" className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}>
            Binary Classification
          </NavLink>
          <NavLink to="/regression" className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}>
            Continuous Regression
          </NavLink>
          <NavLink to="/clustering" className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}>
            Unsupervised Clustering
          </NavLink>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;

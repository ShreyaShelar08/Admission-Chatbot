import React, { useState, useEffect } from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import Login from "./components/Login";
import Signup from "./components/Signup";
import Dashboard from "./components/Dashboard";

function App() {
  const [user, setUser] = useState(null);

  useEffect(() => {
    const currentUser = JSON.parse(localStorage.getItem("currentUser"));
    if (currentUser) setUser(currentUser);
  }, []);

  return (
    <Routes>
      <Route
        path="/"
        element={user ? <Navigate to="/dashboard" /> : <Login setUser={setUser} />}
      />
      <Route path="/signup" element={<Signup setUser={setUser} />} />
      <Route
        path="/dashboard"
        element={user ? <Dashboard user={user} /> : <Navigate to="/" />}
      />
    </Routes>
  );
}

export default App;

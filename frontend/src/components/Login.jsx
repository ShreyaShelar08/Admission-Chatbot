import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import "./Login.css";

const Login = ({ setUser }) => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const navigate = useNavigate();

  const handleLogin = (e) => {
    e.preventDefault();
    const users = JSON.parse(localStorage.getItem("users")) || [];
    const existingUser = users.find(u => u.email === email && u.password === password);

    if (existingUser) {
      setUser(existingUser);
      localStorage.setItem("currentUser", JSON.stringify(existingUser));
      navigate("/dashboard");
    } else {
      alert("Invalid credentials");
    }
  };

  const handleSocialLogin = (provider) => {
    alert(`${provider} authentication would require OAuth setup with backend.\n\nFor now, please use email login.\n\nTo enable ${provider} login, you need to:\n1. Create app in ${provider} Developer Console\n2. Get API keys\n3. Implement OAuth in your backend\n4. Connect to your React app`);
  };

  const handleForgotPassword = () => {
    alert("Forgot password functionality would require:\n1. Email service setup (SendGrid, AWS SES, etc.)\n2. Password reset endpoints in backend\n3. Database update logic\n\nFor now, please contact admin to reset password.");
  };

  return (
    <div className="login-wrapper">
      {/* LEFT SIDE - SAME AS SIGNUP */}
      <div className="login-left">
        <div className="logo">AI Chatbot</div>

        <div className="bot-circle">
          <div className="bot-face">
            <span className="eye"></span>
            <span className="eye"></span>
            <span className="smile"></span>
          </div>
        </div>

        <h1>Meet your new assistant.</h1>
        <p>
          Personalized conversations, tailored for your productivity and
          creative needs.
        </p>
      </div>

      {/* RIGHT SIDE - LOGIN FORM */}
      <div className="login-right">
        <p className="top-signup">
          Don't have an account?{" "}
          <span onClick={() => navigate("/signup")}>Sign Up</span>
        </p>

        <h2>Welcome Back</h2>

        <form onSubmit={handleLogin} className="login-form">
          <div className="form-group">
            <label>Email Address</label>
            <input
              type="email"
              placeholder="name@example.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </div>

          <div className="form-group">
            <label>Password</label>
            <input
              type="password"
              placeholder="Enter your password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>

          <div className="forgot-password">
            <a href="#" onClick={(e) => { e.preventDefault(); handleForgotPassword(); }}>
              Forgot Password?
            </a>
          </div>

          <button type="submit" className="login-btn">
            Login
          </button>
        </form>

        <div className="divider">OR CONTINUE WITH</div>

        <div className="social-buttons">
          <button 
            className="social-btn google" 
            onClick={() => handleSocialLogin("Google")}
          >
            Google
          </button>
          <button 
            className="social-btn github" 
            onClick={() => handleSocialLogin("GitHub")}
          >
            GitHub
          </button>
        </div>
      </div>
    </div>
  );
};

export default Login;
import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import "./Signup.css";

const Signup = () => {
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const navigate = useNavigate();

  const handleSignup = (e) => {
    e.preventDefault();

    const users = JSON.parse(localStorage.getItem("users")) || [];

    if (users.find((u) => u.email === email)) {
      alert("User already exists");
      return;
    }

    users.push({ fullName, email, password });
    localStorage.setItem("users", JSON.stringify(users));

    alert("Signup successful! Please login.");
    navigate("/");
  };

  const handleSocialLogin = (provider) => {
    alert(`${provider} authentication would require OAuth setup with backend.\n\nFor now, please use email signup.\n\nTo enable ${provider} login, you need to:\n1. Create app in ${provider} Developer Console\n2. Get API keys\n3. Implement OAuth in your backend\n4. Connect to your React app`);
  };

  const handleForgotPassword = () => {
    alert("Forgot password functionality would require:\n1. Email service setup (SendGrid, AWS SES, etc.)\n2. Password reset endpoints in backend\n3. Database update logic\n\nFor now, please contact admin to reset password.");
  };

  return (
    <div className="signup-wrapper">
      {/* LEFT SIDE */}
      <div className="signup-left">
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

      {/* RIGHT SIDE */}
      <div className="signup-right">
        <p className="top-login">
          Already have an account?{" "}
          <span onClick={() => navigate("/")}>Log In</span>
        </p>

        <h2>Create Account</h2>

        <form onSubmit={handleSignup} className="signup-form">
          <div className="form-group">
            <label>Full Name</label>
            <input
              type="text"
              placeholder="John Doe"
              value={fullName}
              onChange={(e) => setFullName(e.target.value)}
              required
            />
          </div>

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
              placeholder="Create a password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              minLength="6"
            />
          </div>

          <div className="checkbox-container">
            <input type="checkbox" id="terms" required />
            <label htmlFor="terms">
              I agree to the <a href="#" onClick={(e) => { e.preventDefault(); alert("Terms of Service page would go here"); }}>Terms of Service</a> and{" "}
              <a href="#" onClick={(e) => { e.preventDefault(); alert("Privacy Policy page would go here"); }}>Privacy Policy</a>.
            </label>
          </div>

          <button type="submit" className="signup-btn">
            Sign Up
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

export default Signup;
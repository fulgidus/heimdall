import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Eye, EyeOff, Loader } from 'lucide-react';
import { useAuthStore } from '../store';
import './Login.css';

const Login: React.FC = () => {
    const navigate = useNavigate();
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [showPassword, setShowPassword] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState('');
    const { login } = useAuthStore();

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setError('');
        setIsLoading(true);

        try {
            await login(email, password);
            navigate('/dashboard');
        } catch (err) {
            console.error('Login error:', err);
            setError(err instanceof Error ? err.message : 'Invalid credentials. Please try again.');
        } finally {
            setIsLoading(false);
        }
    };

    const adminEmail = import.meta.env.VITE_ADMIN_EMAIL || 'admin@heimdall.local';
    const adminPassword = import.meta.env.VITE_ADMIN_PASSWORD || 'admin';

    return (
        <div className="login-container">
            {/* Animated Background Elements */}
            <div className="login-background">
                <div className="login-background-orb-1"></div>
                <div className="login-background-orb-2"></div>
            </div>

            {/* Main Container */}
            <div className="login-main-container">
                {/* Header - Logo & Branding */}
                <div className="login-header">
                    <div className="login-logo-container">
                        <div className="login-logo">🚀</div>
                    </div>
                    <h1 className="login-title">
                        Heimdall
                    </h1>
                    <p className="login-subtitle">RF Source Localization Platform</p>
                </div>

                {/* Login Card */}
                <div className="login-card">
                    {/* Card Title */}
                    <div className="login-card-header">
                        <h2 className="login-card-title">Sign In</h2>
                        <p className="login-card-description">Access your RF localization dashboard</p>
                    </div>

                    {/* Form */}
                    <form onSubmit={handleSubmit} className="login-form">
                        {/* Email Input */}
                        <div className="login-form-field" style={{ animationDelay: '0.1s' }}>
                            <label htmlFor="email" className="login-form-label">
                                <span className="login-form-label-inner">
                                    <span className="login-form-icon">@</span>
                                    Email Address
                                </span>
                            </label>
                            <input
                                id="email"
                                type="email"
                                placeholder="admin@heimdall.local"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                required
                                className="login-form-input"
                            />
                        </div>

                        {/* Password Input */}
                        <div className="login-form-field" style={{ animationDelay: '0.15s' }}>
                            <label htmlFor="password" className="login-form-label">
                                <span className="login-form-label-inner">
                                    <span className="login-form-icon">🔐</span>
                                    Password
                                </span>
                            </label>
                            <div className="login-password-container">
                                <input
                                    id="password"
                                    type={showPassword ? 'text' : 'password'}
                                    placeholder="••••••••"
                                    value={password}
                                    onChange={(e) => setPassword(e.target.value)}
                                    required
                                    className="login-form-input login-password-input"
                                />
                                <button
                                    type="button"
                                    onClick={() => setShowPassword(!showPassword)}
                                    className="login-password-toggle"
                                    aria-label={showPassword ? 'Hide password' : 'Show password'}
                                >
                                    {showPassword ? <EyeOff size={20} /> : <Eye size={20} />}
                                </button>
                            </div>
                        </div>

                        {/* Error Message */}
                        {error && (
                            <div className="login-error" role="alert">
                                <span className="login-error-content">
                                    <span className="login-error-icon">⚠️</span>
                                    {error}
                                </span>
                            </div>
                        )}

                        {/* Submit Button */}
                        <button
                            type="submit"
                            disabled={isLoading}
                            className="login-submit-button"
                            style={{ animationDelay: '0.3s' }}
                        >
                            {isLoading ? (
                                <span className="login-submit-loading">
                                    <Loader size={20} className="animate-spin" />
                                    <span>Signing In...</span>
                                </span>
                            ) : (
                                <span>Sign In</span>
                            )}
                        </button>
                    </form>

                    {/* Demo Credentials Info */}
                    <div className="login-demo-section" style={{ animationDelay: '0.35s' }}>
                        <p className="login-demo-title">📋 Demo Credentials</p>
                        <div className="login-demo-credentials">
                            <div>
                                <span className="login-demo-credential-label">Email:</span>{' '}
                                <span className="login-demo-credential-value">{adminEmail}</span>
                            </div>
                            <div>
                                <span className="login-demo-credential-label">Password:</span>{' '}
                                <span className="login-demo-credential-value">{adminPassword}</span>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Footer Info */}
                <div className="login-footer" style={{ animationDelay: '0.4s' }}>
                    <p>🛡️ Secure RF Localization System • Phase 7 Alpha</p>
                </div>
            </div>
        </div>
    );
};

export default Login;

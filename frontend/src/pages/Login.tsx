import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Eye, EyeOff } from 'lucide-react';
import { useAuthStore } from '../store';

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
    const adminPassword = import.meta.env.VITE_ADMIN_PASSWORD || 'Admin123!@#';

    return (
        <div className="min-h-screen bg-gradient-to-br from-oxford-blue via-sea-green to-oxford-blue flex items-center justify-center p-4">
            {/* Animated Background Gradient */}
            <div className="fixed inset-0 bg-gradient-radial-neon opacity-5 -z-10"></div>

            <div className="max-w-md">
                {/* Logo */}
                <div className="text-center mb-12">
                    <h1 className="text-5xl font-bold text-light-green mb-2">
                        🚀 Heimdall
                    </h1>
                    <p className="text-sea-green font-semibold text-lg">RF Localization Platform</p>
                </div>

                {/* Login Card - Solid Colors with AA Contrast */}
                <div className="bg-oxford-blue rounded-xl shadow-2xl p-8 border-2 border-sea-green">
                    <h2 className="text-2xl font-bold text-light-green mb-2">Welcome Back</h2>
                    <p className="text-sea-green mb-8 font-medium">Sign in to your account</p>

                    <form onSubmit={handleSubmit} className="space-y-5">
                        {/* Email */}
                        <div>
                            <label htmlFor="email" className="block text-light-green font-semibold mb-2 text-sm">
                                Email Address
                            </label>
                            <input
                                id="email"
                                type="email"
                                placeholder="admin@heimdall.local"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                required
                                className="w-full px-4 py-3 bg-french-gray bg-opacity-20 border-2 border-sea-green text-white placeholder-sea-green placeholder-opacity-60 rounded-lg focus:outline-none focus:border-light-green focus:bg-opacity-30 transition-all"
                            />
                        </div>

                        {/* Password */}
                        <div>
                            <label htmlFor="password" className="block text-light-green font-semibold mb-2 text-sm">
                                Password
                            </label>
                            <div className="relative">
                                <input
                                    id="password"
                                    type={showPassword ? 'text' : 'password'}
                                    placeholder="••••••••"
                                    value={password}
                                    onChange={(e) => setPassword(e.target.value)}
                                    required
                                    className="w-full px-4 py-3 bg-french-gray bg-opacity-20 border-2 border-sea-green text-white placeholder-sea-green placeholder-opacity-60 rounded-lg focus:outline-none focus:border-light-green focus:bg-opacity-30 transition-all"
                                />
                                <button
                                    type="button"
                                    onClick={() => setShowPassword(!showPassword)}
                                    className="absolute right-3 top-3 text-sea-green hover:text-light-green transition-colors font-semibold"
                                    aria-label={showPassword ? 'Hide password' : 'Show password'}
                                >
                                    {showPassword ? <EyeOff size={20} /> : <Eye size={20} />}
                                </button>
                            </div>
                        </div>

                        {/* Error Message - High Contrast */}
                        {error && (
                            <div className="p-4 bg-red-600 border-2 border-red-400 rounded-lg text-white font-semibold text-sm" role="alert">
                                ⚠️ {error}
                            </div>
                        )}

                        {/* Remember Me & Forgot Password */}
                        <div className="flex items-center justify-between text-sm">
                            <label className="flex items-center gap-2 cursor-pointer">
                                <input
                                    type="checkbox"
                                    className="w-5 h-5 rounded border-2 border-sea-green accent-light-green cursor-pointer bg-french-gray bg-opacity-20"
                                    defaultChecked
                                />
                                <span className="text-sea-green font-medium hover:text-light-green">Remember me</span>
                            </label>
                            <a href="#" className="text-light-green hover:text-neon-blue font-semibold transition-colors">
                                Forgot password?
                            </a>
                        </div>

                        {/* Submit Button */}
                        <button
                            type="submit"
                            disabled={isLoading}
                            className="w-full py-3 mt-8 bg-light-green hover:bg-neon-blue disabled:opacity-50 text-oxford-blue font-bold rounded-lg transition-all transform hover:scale-105 active:scale-95 shadow-lg"
                        >
                            {isLoading ? '⏳ Signing In...' : 'Sign In'}
                        </button>
                    </form>

                    {/* Divider */}
                    <div className="my-8 flex items-center gap-3">
                        <div className="flex-1 h-px bg-sea-green"></div>
                        <span className="text-sea-green font-semibold text-sm">OR</span>
                        <div className="flex-1 h-px bg-sea-green"></div>
                    </div>

                    {/* Social Login */}
                    <div className="grid grid-cols-3 gap-3">
                        {['Google', 'GitHub', 'Microsoft'].map((provider) => (
                            <button
                                key={provider}
                                type="button"
                                className="py-2 px-3 bg-sea-green hover:bg-light-green text-oxford-blue font-bold rounded-lg transition-all text-lg"
                            >
                                {provider === 'Google' ? '🔍' : provider === 'GitHub' ? '🐙' : '⊞'}
                            </button>
                        ))}
                    </div>

                    {/* Sign Up Link */}
                    <p className="text-center text-sea-green text-sm mt-8">
                        Don't have an account?{' '}
                        <a href="#" className="text-light-green hover:text-neon-blue font-bold transition-colors">
                            Sign up
                        </a>
                    </p>
                </div>

                {/* Credentials Info - High Contrast */}
                <div className="mt-6 bg-sea-green bg-opacity-20 border-2 border-sea-green rounded-lg p-4">
                    <p className="text-sm text-light-green text-center font-semibold mb-2">
                        📋 Demo Credentials
                    </p>
                    <p className="text-xs text-sea-green text-center font-mono">
                        Email: <span className="text-white font-bold">{adminEmail}</span><br />
                        Password: <span className="text-white font-bold">{adminPassword}</span>
                    </p>
                </div>
            </div>
        </div>
    );
};

export default Login;

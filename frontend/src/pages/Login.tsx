import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Eye, EyeOff, Loader } from 'lucide-react';
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
        <div className="min-h-screen bg-linear-to-br from-purple-600 via-purple-500 to-indigo-600 flex flex-col items-center justify-center p-4 overflow-hidden">
            {/* Animated Background Elements */}
            <div className="fixed inset-0 overflow-hidden pointer-events-none">
                <div className="absolute top-20 left-10 w-72 h-72 bg-white rounded-full mix-blend-multiply filter blur-3xl opacity-10 animate-pulse"></div>
                <div className="absolute bottom-20 right-10 w-72 h-72 bg-indigo-400 rounded-full mix-blend-multiply filter blur-3xl opacity-10 animate-pulse" style={{ animationDelay: '2s' }}></div>
            </div>

            {/* Main Container */}
            <div className="w-full max-w-md relative z-10">
                {/* Header - Logo & Branding */}
                <div className="text-center mb-12 animate-fade-in">
                    <div className="inline-block mb-4">
                        <div className="text-6xl font-black text-white drop-shadow-lg">🚀</div>
                    </div>
                    <h1 className="text-4xl md:text-5xl font-black text-white mb-2 tracking-tight">
                        Heimdall
                    </h1>
                    <p className="text-purple-200 font-medium text-sm md:text-base">RF Source Localization Platform</p>
                </div>

                {/* Login Card */}
                <div className="bg-white rounded-3xl shadow-2xl p-8! md:p-10! backdrop-blur-xl animate-slide-up">
                    {/* Card Title */}
                    <div className="mb-8">
                        <h2 className="text-2xl md:text-3xl font-bold text-gray-900 mb-2">Sign In</h2>
                        <p className="text-gray-600 text-sm md:text-base">Access your RF localization dashboard</p>
                    </div>

                    {/* Form */}
                    <form onSubmit={handleSubmit} className="space-y-6">
                        {/* Email Input */}
                        <div className="space-y-2 animate-fade-in" style={{ animationDelay: '0.1s' }}>
                            <label htmlFor="email" className="block text-sm font-semibold text-gray-900">
                                <span className="inline-flex items-center gap-2">
                                    <span className="text-purple-600 font-bold">@</span>
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
                                className="w-full px-4 py-3 bg-gray-50 border-2 border-gray-200 text-gray-900 placeholder-gray-500 rounded-xl focus:outline-none focus:border-purple-600 focus:bg-purple-50 transition-all duration-200"
                            />
                        </div>

                        {/* Password Input */}
                        <div className="space-y-2 animate-fade-in" style={{ animationDelay: '0.15s' }}>
                            <label htmlFor="password" className="block text-sm font-semibold text-gray-900">
                                <span className="inline-flex items-center gap-2">
                                    <span className="text-purple-600 font-bold">🔐</span>
                                    Password
                                </span>
                            </label>
                            <div className="relative">
                                <input
                                    id="password"
                                    type={showPassword ? 'text' : 'password'}
                                    placeholder="••••••••"
                                    value={password}
                                    onChange={(e) => setPassword(e.target.value)}
                                    required
                                    className="w-full px-4 py-3 bg-gray-50 border-2 border-gray-200 text-gray-900 placeholder-gray-500 rounded-xl focus:outline-none focus:border-purple-600 focus:bg-purple-50 transition-all duration-200 pr-12"
                                />
                                <button
                                    type="button"
                                    onClick={() => setShowPassword(!showPassword)}
                                    className="absolute right-4 top-1/2 -translate-y-1/2 text-gray-600 hover:text-purple-600 transition-colors"
                                    aria-label={showPassword ? 'Hide password' : 'Show password'}
                                >
                                    {showPassword ? <EyeOff size={20} /> : <Eye size={20} />}
                                </button>
                            </div>
                        </div>

                        {/* Error Message */}
                        {error && (
                            <div className="p-4 bg-red-50 border-2 border-red-300 rounded-xl text-red-700 text-sm font-medium animate-shake" role="alert">
                                <span className="inline-flex items-center gap-2">
                                    <span className="text-lg">⚠️</span>
                                    {error}
                                </span>
                            </div>
                        )}

                        {/* Submit Button */}
                        <button
                            type="submit"
                            disabled={isLoading}
                            className="w-full py-3 mt-8 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 disabled:opacity-60 disabled:cursor-not-allowed text-white font-bold rounded-xl transition-all duration-200 transform hover:scale-105 active:scale-95 shadow-lg hover:shadow-xl flex items-center justify-center gap-2 animate-fade-in"
                            style={{ animationDelay: '0.3s' }}
                        >
                            {isLoading ? (
                                <>
                                    <Loader size={20} className="animate-spin" />
                                    <span>Signing In...</span>
                                </>
                            ) : (
                                <span>Sign In</span>
                            )}
                        </button>
                    </form>

                    {/* Demo Credentials Info */}
                    <div className="mt-8 pt-8 border-t-2 border-gray-200 space-y-3 animate-fade-in" style={{ animationDelay: '0.35s' }}>
                        <p className="text-xs font-semibold text-gray-600 uppercase tracking-wide">📋 Demo Credentials</p>
                        <div className="space-y-2 text-sm font-mono bg-gray-50 p-3 rounded-lg">
                            <div>
                                <span className="text-gray-600">Email:</span>{' '}
                                <span className="text-gray-900 font-bold">{adminEmail}</span>
                            </div>
                            <div>
                                <span className="text-gray-600">Password:</span>{' '}
                                <span className="text-gray-900 font-bold">{adminPassword}</span>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Footer Info */}
                <div className="text-center mt-8 text-purple-200 text-xs animate-fade-in" style={{ animationDelay: '0.4s' }}>
                    <p>🛡️ Secure RF Localization System • Phase 7 Alpha</p>
                </div>
            </div>

            {/* Tailwind Animations */}
            <style>{`
                @keyframes fade-in {
                    from {
                        opacity: 0;
                    }
                    to {
                        opacity: 1;
                    }
                }
                
                @keyframes slide-up {
                    from {
                        opacity: 0;
                        transform: translateY(20px);
                    }
                    to {
                        opacity: 1;
                        transform: translateY(0);
                    }
                }
                
                @keyframes shake {
                    0%, 100% {
                        transform: translateX(0);
                    }
                    10%, 30%, 50%, 70%, 90% {
                        transform: translateX(-5px);
                    }
                    20%, 40%, 60%, 80% {
                        transform: translateX(5px);
                    }
                }
                
                .animate-fade-in {
                    animation: fade-in 0.6s ease-out forwards;
                    opacity: 0;
                }
                
                .animate-slide-up {
                    animation: slide-up 0.6s ease-out forwards;
                }
                
                .animate-shake {
                    animation: shake 0.5s ease-in-out;
                }
            `}</style>
        </div>
    );
};

export default Login;

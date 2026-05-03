import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Mail, Lock, User, Fingerprint, Eye, EyeOff } from 'lucide-react';
import { authAPI } from '../api/auth.api';
import { setToken } from '../utils/token';
import Loader from '../components/Loader';

const Register = () => {
  const navigate = useNavigate();
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: '',
    confirmPassword: '',
  });
  const [role, setRole] = useState("investigator");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const validatePassword = (password) => {
    const minLength = 8;
    const hasUpperCase = /[A-Z]/.test(password);
    const hasLowerCase = /[a-z]/.test(password);
    const hasNumbers = /\d/.test(password);
    const hasSpecialChar = /[!@#$%^&*(),.?":{}|<>]/.test(password);

    if (password.length < minLength) {
      return 'Password must be at least 8 characters long';
    }
    if (!hasUpperCase) {
      return 'Password must contain at least one uppercase letter';
    }
    if (!hasLowerCase) {
      return 'Password must contain at least one lowercase letter';
    }
    if (!hasNumbers) {
      return 'Password must contain at least one number';
    }
    if (!hasSpecialChar) {
      return 'Password must contain at least one special character';
    }
    return null;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    // Validate password strength
    const passwordError = validatePassword(formData.password);
    if (passwordError) {
      setError(passwordError);
      return;
    }

    if (formData.password !== formData.confirmPassword) {
      setError('Passwords do not match');
      return;
    }

    setLoading(true);

    try {
      const response = await authAPI.register({
        name: formData.name,
        email: formData.email,
        password: formData.password,
        role: role,
      });
      
      // Backend returns access_token
      if (response.access_token) {
        setToken(response.access_token);
      } else {
        // Auto-login after registration
        const loginResponse = await authAPI.login({
          email: formData.email,
          password: formData.password,
        });
        setToken(loginResponse.access_token);
      }
      
      if (role === 'admin') {
        navigate('/admin');
      } else {
        navigate('/match');
      }
    } catch (err) {
      setError(err.response?.data?.detail || err.response?.data?.message || 'Registration failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 flex items-center justify-center p-4 pt-16">
      <div className="absolute inset-0 opacity-30" 
           style={{
             backgroundImage: `repeating-linear-gradient(0deg, transparent, transparent 1px, rgba(59, 130, 246, 0.03) 1px, rgba(59, 130, 246, 0.03) 2px),
                              repeating-linear-gradient(90deg, transparent, transparent 1px, rgba(59, 130, 246, 0.03) 1px, rgba(59, 130, 246, 0.03) 2px)`
           }} />
      
      {/* Background Glow */}
      <div className="absolute top-1/3 left-1/2 -translate-x-1/2 w-[600px] h-[600px] bg-blue-600/5 rounded-full blur-3xl" />

      <div className="relative z-10 w-full max-w-md">
        {/* Logo */}
        <div className="text-center mb-8">
          <Link to="/" className="inline-flex items-center gap-2">
            <Fingerprint className="h-10 w-10 text-blue-500" />
            <span className="font-mono font-bold text-2xl">
              <span className="text-blue-500">FOREN</span>
              <span className="text-white">SIX</span>
            </span>
          </Link>
          <p className="text-slate-400 mt-2">Create your secure account</p>
        </div>

        {/* Register Card */}
        <div className="p-8 rounded-xl bg-slate-900/50 border border-slate-800/50 backdrop-blur-sm">
          <h2 className="text-2xl font-bold text-white mb-2 text-center">
            Create Account
          </h2>
          <p className="text-center text-slate-400 mb-6">
            Join Forensix today
          </p>

          {error && (
            <div className="bg-red-500/10 border border-red-500/50 text-red-400 px-4 py-3 rounded-lg mb-6 text-sm">
              {error}
            </div>
          )}

          <div className="mb-6">
            <label className="block text-slate-300 font-medium mb-2 text-sm">
              Select Role
            </label>
            <div className="flex gap-4">
              <button
                type="button"
                onClick={() => setRole("admin")}
                className={`flex-1 py-2 px-4 rounded-lg border transition-all ${
                  role === "admin"
                    ? "border-blue-500 bg-blue-500/10 text-blue-500"
                    : "border-slate-700/50 text-slate-400 bg-slate-800"
                }`}
              >
                Admin
              </button>
              <button
                type="button"
                onClick={() => setRole("investigator")}
                className={`flex-1 py-2 px-4 rounded-lg border transition-all ${
                  role === "investigator"
                    ? "border-cyan-500 bg-cyan-500/10 text-cyan-500"
                    : "border-slate-700/50 text-slate-400 bg-slate-800"
                }`}
              >
                Investigator
              </button>
            </div>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-slate-300 font-medium mb-2 text-sm">
                Full Name
              </label>
              <div className="relative">
                <User className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400 w-5 h-5" />
                <input
                  type="text"
                  name="name"
                  value={formData.name}
                  onChange={handleChange}
                  required
                  className="w-full pl-10 pr-4 py-3 bg-slate-800 border border-slate-700/50 rounded-lg focus:outline-none focus:border-blue-500/50 text-white placeholder:text-slate-500"
                  placeholder="John Doe"
                />
              </div>
            </div>

            <div>
              <label className="block text-slate-300 font-medium mb-2 text-sm">
                Email
              </label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400 w-5 h-5" />
                <input
                  type="email"
                  name="email"
                  value={formData.email}
                  onChange={handleChange}
                  required
                  className="w-full pl-10 pr-4 py-3 bg-slate-800 border border-slate-700/50 rounded-lg focus:outline-none focus:border-blue-500/50 text-white placeholder:text-slate-500"
                  placeholder="your.email@example.com"
                />
              </div>
            </div>

            <div>
              <label className="block text-slate-300 font-medium mb-2 text-sm">
                Password
              </label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400 w-5 h-5" />
                <input
                  type={showPassword ? "text" : "password"}
                  name="password"
                  value={formData.password}
                  onChange={handleChange}
                  required
                  className="w-full pl-10 pr-12 py-3 bg-slate-800 border border-slate-700/50 rounded-lg focus:outline-none focus:border-blue-500/50 text-white placeholder:text-slate-500"
                  placeholder="••••••••"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-white"
                >
                  {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                </button>
              </div>
              <p className="text-xs text-slate-500 mt-1">
                Must be 8+ characters with uppercase, lowercase, number & special character
              </p>
            </div>

            <div>
              <label className="block text-slate-300 font-medium mb-2 text-sm">
                Confirm Password
              </label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400 w-5 h-5" />
                <input
                  type={showConfirmPassword ? "text" : "password"}
                  name="confirmPassword"
                  value={formData.confirmPassword}
                  onChange={handleChange}
                  required
                  className="w-full pl-10 pr-12 py-3 bg-slate-800 border border-slate-700/50 rounded-lg focus:outline-none focus:border-blue-500/50 text-white placeholder:text-slate-500"
                  placeholder="••••••••"
                />
                <button
                  type="button"
                  onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-white"
                >
                  {showConfirmPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                </button>
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full bg-gradient-to-r from-blue-600 to-cyan-600 text-white py-3 rounded-lg font-semibold hover:from-blue-700 hover:to-cyan-700 transition disabled:opacity-50 shadow-lg shadow-blue-500/20"
            >
              {loading ? <Loader message="" /> : 'Create Account'}
            </button>
          </form>

          <p className="text-center text-slate-400 mt-6 text-sm">
            Already have an account?{' '}
            <Link to="/login" className="text-blue-500 font-semibold hover:underline">
              Sign in
            </Link>
          </p>
        </div>

        {/* Back Link */}
        <p className="text-center mt-6">
          <Link to="/" className="text-sm text-blue-500 hover:underline">
            ← Back to Home
          </Link>
        </p>
      </div>
    </div>
  );
};

export default Register;
import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Shield, User, Lock, Fingerprint, Eye, EyeOff } from "lucide-react";
import { authAPI } from '../api/auth.api';
import { setToken } from '../utils/token';
import Loader from '../components/Loader';

export default function Login() {
  const [role, setRole] = useState("admin");
  const [showPassword, setShowPassword] = useState(false);
  const [formData, setFormData] = useState({
    email: '',
    password: '',
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      const response = await authAPI.login(formData);
      setToken(response.access_token);

      try {
        const userProfile = await authAPI.getProfile();
        if (userProfile.role === 'admin') {
          navigate('/admin');
        } else {
          navigate('/match');
        }
      } catch (err) {
        navigate('/match');
      }
    } catch (err) {
      setError(err.response?.data?.detail || err.response?.data?.message || 'Login failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen flex items-center justify-center pt-16 pb-8 bg-slate-950">
      <div className="absolute inset-0 opacity-30" 
           style={{
             backgroundImage: `repeating-linear-gradient(0deg, transparent, transparent 1px, rgba(59, 130, 246, 0.03) 1px, rgba(59, 130, 246, 0.03) 2px),
                              repeating-linear-gradient(90deg, transparent, transparent 1px, rgba(59, 130, 246, 0.03) 1px, rgba(59, 130, 246, 0.03) 2px)`
           }} />
      
      {/* Background Glow */}
      <div className="absolute top-1/3 left-1/2 -translate-x-1/2 w-[600px] h-[600px] bg-blue-600/5 rounded-full blur-3xl" />

      <div className="relative z-10 w-full max-w-md px-4">
        {/* Logo */}
        <div className="text-center mb-8">
          <Link to="/" className="inline-flex items-center gap-2">
            <Fingerprint className="h-10 w-10 text-blue-500" />
            <span className="font-mono font-bold text-2xl">
              <span className="text-blue-500">FOREN</span>
              <span className="text-white">SIX</span>
            </span>
          </Link>
          <p className="text-slate-400 mt-2">Secure Authentication Portal</p>
        </div>

        {/* Login Card */}
        <div className="p-8 rounded-xl bg-slate-900/50 border border-slate-800/50 backdrop-blur-sm">
          {/* Role Selector */}
          <div className="mb-6">
            <label className="block text-sm font-medium mb-3 text-slate-400">
              Select Role
            </label>
            <div className="grid grid-cols-2 gap-3">
              <button
                type="button"
                onClick={() => setRole("admin")}
                className={`p-4 rounded-lg border transition-all ${
                  role === "admin"
                    ? "border-blue-500 bg-blue-500/10 text-blue-500"
                    : "border-slate-800/50 text-slate-400 hover:border-blue-500/30"
                }`}
              >
                <Shield className="w-6 h-6 mx-auto mb-2" />
                <span className="text-sm font-medium">Admin</span>
              </button>
              <button
                type="button"
                onClick={() => setRole("investigator")}
                className={`p-4 rounded-lg border transition-all ${
                  role === "investigator"
                    ? "border-cyan-500 bg-cyan-500/10 text-cyan-500"
                    : "border-slate-800/50 text-slate-400 hover:border-cyan-500/30"
                }`}
              >
                <User className="w-6 h-6 mx-auto mb-2" />
                <span className="text-sm font-medium">Investigator</span>
              </button>
            </div>
          </div>

          {/* Role Description */}
          <div className="mb-6 p-3 rounded-lg bg-slate-800/50 text-sm text-slate-400">
            {role === "admin" ? (
              <p>Full control: Manage sketches, images, and system data.</p>
            ) : (
              <p>Query access: Search and view match results only.</p>
            )}
          </div>

          {error && (
            <div className="mb-6 p-3 rounded-lg bg-red-500/10 border border-red-500/50 text-red-400 text-sm">
              {error}
            </div>
          )}

          {/* Form */}
          <form className="space-y-4" onSubmit={handleSubmit}>
            {/* Email */}
            <div>
              <label className="block text-sm font-medium mb-2 text-slate-300">Email</label>
              <div className="relative">
                <User className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
                <input
                  type="email"
                  name="email"
                  value={formData.email}
                  onChange={handleChange}
                  placeholder="Enter your email"
                  required
                  className="w-full pl-10 pr-4 py-3 rounded-lg bg-slate-800 border border-slate-700/50 text-white placeholder:text-slate-500 focus:outline-none focus:border-blue-500/50"
                />
              </div>
            </div>

            {/* Password */}
            <div>
              <label className="block text-sm font-medium mb-2 text-slate-300">Password</label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
                <input
                  type={showPassword ? "text" : "password"}
                  name="password"
                  value={formData.password}
                  onChange={handleChange}
                  placeholder="Enter your password"
                  required
                  className="w-full pl-10 pr-12 py-3 rounded-lg bg-slate-800 border border-slate-700/50 text-white placeholder:text-slate-500 focus:outline-none focus:border-blue-500/50"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-white"
                >
                  {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                </button>
              </div>
            </div>

            {/* Submit */}
            <button
              type="submit"
              disabled={loading}
              className={`w-full py-3 rounded-lg font-semibold transition-all flex items-center justify-center gap-2 ${
                role === "admin"
                  ? "bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-700 hover:to-blue-600 text-white shadow-lg shadow-blue-500/20"
                  : "bg-gradient-to-r from-cyan-600 to-cyan-500 hover:from-cyan-700 hover:to-cyan-600 text-white shadow-lg shadow-cyan-500/20"
              } disabled:opacity-50`}
            >
              {loading ? (
                <Loader message="" />
              ) : (
                <>
                  <Shield className="w-4 h-4" />
                  Secure Login
                </>
              )}
            </button>
          </form>

          {/* Footer */}
          <p className="mt-6 text-center text-sm text-slate-400">
            Don't have an account?{" "}
            <Link to="/register" className="text-blue-500 hover:underline">
              Register
            </Link>
          </p>
          <p className="mt-2 text-center text-xs text-slate-500">
            Role-based access ensures system integrity and prevents misuse.
          </p>
        </div>

        {/* Back Link */}
        <p className="text-center mt-6">
          <Link to="/" className="text-sm text-blue-500 hover:underline">
            ← Back to Home
          </Link>
        </p>
      </div>
    </main>
  );
}
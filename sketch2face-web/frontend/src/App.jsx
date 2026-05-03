import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import Login from './pages/Login';
import Register from './pages/Register';
import Match from './pages/Match';
import History from './pages/History';
import Admin from './pages/Admin';
import HowItWorks from './pages/HowItWorks';
import Features from './pages/Features';
import RequireAuth from './auth/requireAuth';

function App() {
  try {
    return (
      <Router>
        <div className="min-h-screen flex flex-col bg-background">
          <Navbar />
          <div className="flex-1">
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/login" element={<Login />} />
              <Route path="/register" element={<Register />} />
              <Route path="/how-it-works" element={<HowItWorks />} />
              <Route path="/features" element={<Features />} />
              <Route path="/match" element={<Match />} />
              <Route
                path="/history"
                element={
                  <RequireAuth>
                    <History />
                  </RequireAuth>
                }
              />
              <Route
                path="/admin"
                element={
                  <RequireAuth>
                    <Admin />
                  </RequireAuth>
                }
              />
            </Routes>
          </div>
        </div>
      </Router>
    );
  } catch (error) {
    console.error('App Error:', error);
    return (
      <div style={{ padding: '20px', color: 'white' }}>
        <h1>Error Loading App</h1>
        <p>{error.message}</p>
      </div>
    );
  }
}

export default App;

import { useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { Menu, X, Shield, Fingerprint } from "lucide-react";
import { Button } from "./ui/Button";
import { cn } from "../lib/utils";
import { removeToken, isAuthenticated, getUserRole } from '../utils/token';

const navItems = [
  { name: "Home", path: "/" },
  { name: "How It Works", path: "/how-it-works" },
  { name: "Features", path: "/features" },
  { name: "Demo", path: "/match" },
  { name: "History", path: "/history" },
];

export function Navbar() {
  const [isOpen, setIsOpen] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();
  const isAuth = isAuthenticated();
  const userRole = getUserRole();

  const handleLogout = () => {
    removeToken();
    navigate('/login');
  };

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 border-b border-border/50 bg-background/80 backdrop-blur-xl">
      <div className="container mx-auto px-4">
        <div className="flex h-16 items-center justify-between">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-2 group">
            <div className="relative">
              <Fingerprint className="h-8 w-8 text-primary transition-all duration-300 group-hover:scale-110" />
              <div className="absolute inset-0 blur-lg bg-primary/30 opacity-0 group-hover:opacity-100 transition-opacity" />
            </div>
            <span className="font-mono font-bold text-lg tracking-tight">
              <span className="text-primary">FOREN</span>
              <span className="text-foreground">SIX</span>
            </span>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden lg:flex items-center gap-1">
            {navItems.map((item) => (
              <Link
                key={item.path}
                to={item.path}
                className={cn(
                  "px-4 py-2 text-sm font-medium rounded-md transition-all duration-300",
                  location.pathname === item.path
                    ? "text-primary bg-primary/10"
                    : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
                )}
              >
                {item.name}
              </Link>
            ))}
          </div>

          {/* Auth Buttons */}
          <div className="hidden lg:flex items-center gap-3">
            {isAuth ? (
              <>
                {userRole === 'admin' && (
                  <Link to="/admin">
                    <Button variant="default" size="sm" className="bg-indigo-600 hover:bg-indigo-700 text-white gap-2">
                      <Shield className="h-4 w-4" />
                      Dashboard
                    </Button>
                  </Link>
                )}
                <Button variant="outline" size="sm" onClick={handleLogout} className="gap-2">
                  <Shield className="h-4 w-4" />
                  Logout
                </Button>
              </>
            ) : (
              <Link to="/login">
                <Button variant="outline" size="sm" className="gap-2">
                  <Shield className="h-4 w-4" />
                  Login
                </Button>
              </Link>
            )}
          </div>

          {/* Mobile Menu Button */}
          <button
            className="lg:hidden p-2 text-muted-foreground hover:text-foreground"
            onClick={() => setIsOpen(!isOpen)}
          >
            {isOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
          </button>
        </div>

        {/* Mobile Navigation */}
        {isOpen && (
          <div className="lg:hidden py-4 border-t border-border/50">
            <div className="flex flex-col gap-2">
              {navItems.map((item) => (
                <Link
                  key={item.path}
                  to={item.path}
                  onClick={() => setIsOpen(false)}
                  className={cn(
                    "px-4 py-3 text-sm font-medium rounded-md transition-all",
                    location.pathname === item.path
                      ? "text-primary bg-primary/10"
                      : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
                  )}
                >
                  {item.name}
                </Link>
              ))}
              {isAuth ? (
                <div className="flex flex-col gap-2 mt-2">
                  {userRole === 'admin' && (
                    <Link to="/admin" onClick={() => setIsOpen(false)}>
                      <Button variant="default" className="w-full bg-indigo-600 hover:bg-indigo-700 text-white gap-2">
                        <Shield className="h-4 w-4" />
                        Admin Dashboard
                      </Button>
                    </Link>
                  )}
                  <Button variant="outline" className="w-full gap-2" onClick={handleLogout}>
                    <Shield className="h-4 w-4" />
                    Logout
                  </Button>
                </div>
              ) : (
                <Link to="/login" onClick={() => setIsOpen(false)}>
                  <Button variant="outline" className="w-full mt-2 gap-2">
                    <Shield className="h-4 w-4" />
                    Login
                  </Button>
                </Link>
              )}
            </div>
          </div>
        )}
      </div>
    </nav>
  );
}

export default Navbar;

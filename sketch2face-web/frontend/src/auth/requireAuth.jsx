import { Navigate } from 'react-router-dom';
import { isAuthenticated } from '../utils/token';

const RequireAuth = ({ children, fallback = null }) => {
  if (!isAuthenticated()) {
    // If fallback provided, show it (for testing)
    // Otherwise redirect to login
    if (fallback) {
      return fallback;
    }
    return <Navigate to="/login" replace />;
  }

  return children;
};

export default RequireAuth;
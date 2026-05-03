import { useState, useEffect } from 'react';
import { Clock, CheckCircle, XCircle, AlertCircle } from 'lucide-react';
import { matchAPI } from '../api/match.api';
import { API_BASE_URL } from '../api/axios';
import Loader from '../components/Loader';

const History = () => {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    try {
      const response = await matchAPI.getHistory();
      // Backend returns array directly, not wrapped in matches
      const historyData = Array.isArray(response) ? response : (response.matches || []);
      setHistory(historyData);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to load history');
    } finally {
      setLoading(false);
    }
  };

  const getStatusIcon = (decision) => {
    if (decision === 'ACCEPTED' || decision === 'HIGH_CONFIDENCE_MATCH') {
      return <CheckCircle className="w-6 h-6 text-success" />;
    } else if (decision === 'MEDIUM_CONFIDENCE') {
      return <AlertCircle className="w-6 h-6 text-warning" />;
    } else {
      return <XCircle className="w-6 h-6 text-destructive" />;
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen pt-24 flex items-center justify-center">
        <Loader message="Loading history..." />
      </div>
    );
  }

  return (
    <main className="min-h-screen pt-24 pb-16">
      <div className="absolute inset-0 grid-pattern opacity-30" />
      
      <div className="container mx-auto px-4 relative z-10 max-w-6xl">
        <div className="flex items-center gap-3 mb-8">
          <Clock className="w-8 h-8 text-primary" />
          <h1 className="text-4xl font-bold">Match History</h1>
        </div>

        {error && (
          <div className="bg-destructive/10 border border-destructive/50 text-destructive px-4 py-3 rounded-lg mb-6">
            {error}
          </div>
        )}

        {history.length === 0 ? (
          <div className="bg-card rounded-lg border border-border/50 shadow-lg p-12 text-center">
            <Clock className="w-16 h-16 text-muted-foreground mx-auto mb-4 opacity-50" />
            <h3 className="text-xl font-semibold mb-2">
              No Match History Yet
            </h3>
            <p className="text-muted-foreground">
              Your face matching history will appear here
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            {history.map((match) => {
              const topMatch = match.top_matches?.[0];
              const decision = match.decision_intelligence?.final_decision;
              
              return (
                <div
                  key={match.query_id}
                  className="bg-card rounded-lg border border-border/50 shadow-md p-6 hover:shadow-lg transition"
                >
                  <div className="flex items-start gap-6">
                    {match.input_image && (
                      <img
                        src={match.input_image.startsWith('http') 
                          ? match.input_image 
                          : `${API_BASE_URL}/${match.input_image.replace(/\\/g, '/')}`}
                        alt="Uploaded"
                        className="w-24 h-24 object-cover rounded-lg border border-border/50"
                        onError={(e) => {
                          e.target.style.display = 'none';
                        }}
                      />
                    )}

                    <div className="flex-1">
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-2">
                          {getStatusIcon(decision)}
                          <h3 className="text-xl font-bold">
                            {topMatch?.name || 'No Match Found'}
                          </h3>
                        </div>
                        <span className="text-sm text-muted-foreground">
                          {new Date(match.timestamp).toLocaleString()}
                        </span>
                      </div>

                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        {topMatch && (
                          <div className="bg-muted/50 rounded p-3">
                            <p className="text-xs text-muted-foreground mb-1">Top Match Similarity</p>
                            <p className="text-lg font-bold text-primary">
                              {(topMatch.similarity_score * 100).toFixed(2)}%
                            </p>
                          </div>
                        )}

                        {match.decision_intelligence && (
                          <>
                            <div className="bg-muted/50 rounded p-3">
                              <p className="text-xs text-muted-foreground mb-1">Reliability</p>
                              <p className="text-lg font-bold text-success">
                                {match.decision_intelligence.reliability_score}
                              </p>
                            </div>

                            <div className="bg-muted/50 rounded p-3">
                              <p className="text-xs text-muted-foreground mb-1">Density Risk</p>
                              <p className="text-lg font-semibold text-foreground">
                                {match.decision_intelligence.density_risk}
                              </p>
                            </div>

                            <div className="bg-muted/50 rounded p-3">
                              <p className="text-xs text-muted-foreground mb-1">Decision</p>
                              <p className="text-lg font-semibold text-success">
                                {match.decision_intelligence.final_decision}
                              </p>
                            </div>
                          </>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </main>
  );
};

export default History;
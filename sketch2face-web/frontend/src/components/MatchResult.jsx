import { CheckCircle, XCircle, AlertCircle, Image as ImageIcon } from 'lucide-react';
import { API_BASE_URL } from '../api/axios';

const MatchResult = ({ result }) => {
  if (!result) return null;

  // Backend returns: query_id, generated_image, top_matches, decision_intelligence
  const topMatch = result.top_matches?.[0];
  const similarity = topMatch?.similarity_score || 0;
  const decision = result.decision_intelligence?.final_decision || 'UNKNOWN';

  const getStatusIcon = () => {
    if (decision === 'ACCEPTED' || decision === 'HIGH_CONFIDENCE_MATCH') {
      return <CheckCircle className="w-16 h-16 text-green-500" />;
    } else if (decision === 'MEDIUM_CONFIDENCE') {
      return <AlertCircle className="w-16 h-16 text-yellow-500" />;
    } else {
      return <XCircle className="w-16 h-16 text-red-500" />;
    }
  };

  const getStatusText = () => {
    if (decision === 'ACCEPTED' || decision === 'HIGH_CONFIDENCE_MATCH') {
      return 'High Confidence Match';
    } else if (decision === 'MEDIUM_CONFIDENCE') {
      return 'Medium Confidence Match';
    } else {
      return 'Low Confidence Match';
    }
  };

  const getStatusColor = () => {
    if (decision === 'ACCEPTED' || decision === 'HIGH_CONFIDENCE_MATCH') {
      return 'text-green-600';
    } else if (decision === 'MEDIUM_CONFIDENCE') {
      return 'text-yellow-600';
    } else {
      return 'text-red-600';
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-8">
      <div className="text-center mb-6">
        <div className="flex justify-center mb-4">{getStatusIcon()}</div>
        <h2 className={`text-3xl font-bold mb-4 ${getStatusColor()}`}>
          {getStatusText()}
        </h2>
      </div>

      {/* Generated Image */}
      {result.generated_image && (
        <div className="mb-6">
          <h3 className="text-lg font-semibold text-gray-700 mb-2 flex items-center gap-2">
            <ImageIcon className="w-5 h-5" />
            Generated Face
          </h3>
          <img
            src={result.generated_image.startsWith('http') 
              ? result.generated_image 
              : `${API_BASE_URL}/${result.generated_image.replace(/\\/g, '/')}`}
            alt="Generated face"
            className="w-full max-w-md mx-auto rounded-lg border-2 border-gray-200"
            onError={(e) => {
              e.target.style.display = 'none';
            }}
          />
        </div>
      )}

      {/* Top Match Info */}
      {topMatch && (
        <div className="space-y-3 text-left max-w-md mx-auto">
          <div className="flex justify-between items-center p-3 bg-gray-50 rounded">
            <span className="font-semibold text-gray-700">Top Match Similarity:</span>
            <span className={`font-bold ${getStatusColor()}`}>
              {(similarity * 100).toFixed(2)}%
            </span>
          </div>

          {topMatch.name && (
            <div className="flex justify-between items-center p-3 bg-gray-50 rounded">
              <span className="font-semibold text-gray-700">Matched Identity:</span>
              <span className="font-bold text-indigo-600 truncate ml-2">
                {topMatch.name}
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default MatchResult;
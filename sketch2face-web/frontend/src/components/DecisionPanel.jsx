import { ThumbsUp, ThumbsDown, MessageSquare, CheckCircle, AlertCircle, XCircle } from 'lucide-react';
import { useState } from 'react';

const DecisionPanel = ({ decisionIntelligence, onFeedback }) => {
  const [feedback, setFeedback] = useState('');
  const [selectedDecision, setSelectedDecision] = useState(null);

  const handleSubmit = () => {
    if (selectedDecision && onFeedback) {
      onFeedback({
        decision: selectedDecision,
        comment: feedback,
      });
      setFeedback('');
      setSelectedDecision(null);
    }
  };

  if (!decisionIntelligence) {
    return null;
  }

  const { reliability_score, density_risk, consistency_verdict, final_decision } = decisionIntelligence;

  const getDecisionColor = (decision) => {
    if (decision === 'ACCEPTED' || decision === 'HIGH_CONFIDENCE_MATCH') {
      return 'text-green-600 bg-green-50 border-green-200';
    } else if (decision === 'MEDIUM_CONFIDENCE') {
      return 'text-yellow-600 bg-yellow-50 border-yellow-200';
    } else {
      return 'text-red-600 bg-red-50 border-red-200';
    }
  };

  const getDecisionIcon = (decision) => {
    if (decision === 'ACCEPTED' || decision === 'HIGH_CONFIDENCE_MATCH') {
      return <CheckCircle className="w-6 h-6" />;
    } else if (decision === 'MEDIUM_CONFIDENCE') {
      return <AlertCircle className="w-6 h-6" />;
    } else {
      return <XCircle className="w-6 h-6" />;
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h3 className="text-xl font-bold text-gray-800 mb-4">
        🧠 Decision Intelligence Report
      </h3>

      {/* Decision Intelligence Metrics */}
      <div className="space-y-4 mb-6">
        {/* Final Decision */}
        <div className={`p-4 rounded-lg border-2 ${getDecisionColor(final_decision)}`}>
          <div className="flex items-center gap-3">
            {getDecisionIcon(final_decision)}
            <div>
              <div className="font-semibold">Final Decision: {final_decision}</div>
              <div className="text-sm opacity-75">
                {final_decision === 'ACCEPTED' || final_decision === 'HIGH_CONFIDENCE_MATCH'
                  ? 'High confidence match - proceed with investigation'
                  : 'Low confidence - requires manual review'}
              </div>
            </div>
          </div>
        </div>

        {/* Reliability Score */}
        <div className="grid grid-cols-3 gap-4">
          <div className="p-3 bg-blue-50 rounded-lg border border-blue-200">
            <div className="text-sm text-gray-600">Reliability</div>
            <div className="text-2xl font-bold text-blue-600">{reliability_score}</div>
            <div className="text-xs text-gray-500">/ 100</div>
          </div>

          <div className="p-3 bg-purple-50 rounded-lg border border-purple-200">
            <div className="text-sm text-gray-600">Density Risk</div>
            <div className="text-lg font-bold text-purple-600">{density_risk}</div>
          </div>

          <div className="p-3 bg-indigo-50 rounded-lg border border-indigo-200">
            <div className="text-sm text-gray-600">Consistency</div>
            <div className="text-lg font-bold text-indigo-600">{consistency_verdict}</div>
          </div>
        </div>
      </div>

      {/* Feedback Section */}
      {onFeedback && (
        <>
          <div className="border-t pt-4 mb-4">
            <h4 className="text-lg font-semibold text-gray-800 mb-4">
              Was this match accurate?
            </h4>
            <div className="flex gap-4 mb-6">
              <button
                onClick={() => setSelectedDecision('correct')}
                className={`flex-1 flex items-center justify-center gap-2 py-4 rounded-lg border-2 transition ${
                  selectedDecision === 'correct'
                    ? 'border-green-500 bg-green-50 text-green-700'
                    : 'border-gray-300 hover:border-green-400'
                }`}
              >
                <ThumbsUp className="w-6 h-6" />
                <span className="font-semibold">Correct</span>
              </button>

              <button
                onClick={() => setSelectedDecision('incorrect')}
                className={`flex-1 flex items-center justify-center gap-2 py-4 rounded-lg border-2 transition ${
                  selectedDecision === 'incorrect'
                    ? 'border-red-500 bg-red-50 text-red-700'
                    : 'border-gray-300 hover:border-red-400'
                }`}
              >
                <ThumbsDown className="w-6 h-6" />
                <span className="font-semibold">Incorrect</span>
              </button>
            </div>
          </div>
        </>
      )}

      <div className="mb-4">
        <label className="flex items-center gap-2 text-gray-700 font-semibold mb-2">
          <MessageSquare className="w-5 h-5" />
          Additional Comments (Optional)
        </label>
        <textarea
          value={feedback}
          onChange={(e) => setFeedback(e.target.value)}
          placeholder="Share any feedback about this match..."
          className="w-full border border-gray-300 rounded-lg p-3 focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
          rows="3"
        />
      </div>

      <button
        onClick={handleSubmit}
        disabled={!selectedDecision}
        className={`w-full py-3 rounded-lg font-semibold transition ${
          selectedDecision
            ? 'bg-indigo-600 text-white hover:bg-indigo-700'
            : 'bg-gray-300 text-gray-500 cursor-not-allowed'
        }`}
      >
        Submit Feedback
      </button>
    </div>
  );
};

export default DecisionPanel;
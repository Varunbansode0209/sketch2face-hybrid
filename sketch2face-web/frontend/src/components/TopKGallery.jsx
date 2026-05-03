import { API_BASE_URL } from '../api/axios';

const TopKGallery = ({ matches }) => {
  if (!matches || matches.length === 0) return null;

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h3 className="text-2xl font-bold text-gray-800 mb-6">Top-5 Matches</h3>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
        {matches.map((match, index) => {
          // Backend returns: similarity_score, image_path, name, match_id
          const similarity = match.similarity_score || match.score || 0;
          const imagePath = match.image_path || match.image || '/placeholder.jpg';
          const name = match.name || `Match ${index + 1}`;
          
          return (
            <div
              key={match.match_id || match.id || index}
              className={`border-2 rounded-lg overflow-hidden hover:shadow-lg transition ${
                index === 0 ? 'border-green-500' : 'border-gray-300'
              }`}
            >
              <div className="relative">
                <img
                  src={imagePath.startsWith('http') ? imagePath : `${API_BASE_URL}/${imagePath.replace(/\\/g, '/')}`}
                  alt={name}
                  className="w-full h-48 object-cover"
                  onError={(e) => {
                    e.target.src = '/placeholder.jpg';
                  }}
                />
                <div className={`absolute top-2 left-2 px-3 py-1 rounded-full text-sm font-bold ${
                  index === 0 ? 'bg-green-600 text-white' : 'bg-indigo-600 text-white'
                }`}>
                  #{index + 1}
                </div>
              </div>
              
              <div className="p-4">
                <h4 className="font-bold text-lg text-gray-800 mb-2 truncate" title={name}>
                  {name}
                </h4>
                
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Similarity:</span>
                    <span className="font-bold text-indigo-600">
                      {(similarity * 100).toFixed(1)}%
                    </span>
                  </div>
                  
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-indigo-600 h-2 rounded-full transition-all"
                      style={{ width: `${similarity * 100}%` }}
                    />
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default TopKGallery;
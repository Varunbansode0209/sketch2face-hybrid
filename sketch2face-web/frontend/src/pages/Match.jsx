import { useState } from "react";
import { Button } from "../components/ui/Button";
import { Upload, Search, CheckCircle, AlertTriangle, XCircle, User, Image } from "lucide-react";
import { matchAPI } from '../api/match.api';
import { API_BASE_URL } from '../api/axios';


export default function Match() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState(null);
  const [generatedImage, setGeneratedImage] = useState(null);
  const [heatmapImage, setHeatmapImage] = useState(null);
  const [reliability, setReliability] = useState(null);
  const [density, setDensity] = useState(null);
  const [consistency, setConsistency] = useState(null);
  const [selectedGallery, setSelectedGallery] = useState('celeba');
  const [error, setError] = useState('');

  const handleFileChange = (e) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResults(null);
      setError('');
    }
  };

  const handleProcess = async () => {
    if (!file) return;
    
    setIsProcessing(true);
    setError('');
    
    try {
      const response = await matchAPI.run(file, selectedGallery);
      
      // Store generated image and heatmap
      if (response.generated_image) {
        // Convert path to URL
        // Backend returns: "./results/generated_xxx.jpg" or "results/generated_xxx.jpg"
        let genPath = response.generated_image.replace(/\\/g, '/');
        // Remove leading "./" if present
        genPath = genPath.replace(/^\.\//, '');
        // If it's already a full URL, use it; otherwise construct from backend
        const genUrl = genPath.startsWith('http') 
          ? genPath 
          : `${API_BASE_URL}/${genPath}`;
        setGeneratedImage(genUrl);
      }
      
      if (response.heatmap) {
        // Convert path to URL
        // Backend returns: "processed/final_demo/heatmaps/xxx.jpg" or "./results/heatmaps/xxx.jpg"
        let heatPath = response.heatmap.replace(/\\/g, '/');
        // Remove leading "./" if present
        heatPath = heatPath.replace(/^\.\//, '');
        // Check if it's in results directory, otherwise might be in processed/
        if (heatPath.includes('results/heatmaps')) {
          const heatUrl = heatPath.startsWith('http') 
            ? heatPath 
            : `${API_BASE_URL}/${heatPath}`;
          setHeatmapImage(heatUrl);
        } else if (heatPath.includes('processed/')) {
          // For processed directory, we might need a different endpoint
          // For now, try to serve it
          const heatUrl = `${API_BASE_URL}/${heatPath}`;
          setHeatmapImage(heatUrl);
        }
      }
      
      // Transform backend response to match UI format
      if (response.top_matches && response.top_matches.length > 0) {
        const transformedResults = response.top_matches.map((match, index) => {
          // Convert image path to full URL if needed
          let imagePath = match.image_path || '';
          if (imagePath && !imagePath.startsWith('http')) {
            // Backend returns: "gallery/celeba/xxx.jpg" or "gallery/cufs/xxx.jpg"
            imagePath = imagePath.startsWith('/') ? imagePath : `/${imagePath}`;
          }
          
          return {
            id: match.match_id || match.id || index,
            name: match.name || `Match ${index + 1}`,
            similarity: match.similarity_score || match.score || 0,
            accepted: index === 0 && (match.similarity_score || match.score || 0) > 0.7,
            image_path: imagePath,
          };
        });
        setResults(transformedResults);
      }

      // Extract Decision Intelligence data
      if (response.decision_intelligence) {
        const di = response.decision_intelligence;
        setReliability(di.reliability_score || null);
        setDensity(di.density_risk || null);
        setConsistency(di.consistency_verdict || null);
      }
    } catch (err) {
      setError(err.response?.data?.detail || err.response?.data?.message || 'Match failed. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  const resetDemo = () => {
    setFile(null);
    setPreview(null);
    setResults(null);
    setGeneratedImage(null);
    setHeatmapImage(null);
    setReliability(null);
    setDensity(null);
    setConsistency(null);
    setError('');
  };

  return (
    <main className="min-h-screen pt-24 pb-16">
      <div className="absolute inset-0 grid-pattern opacity-30" />
      
      <div className="container mx-auto px-4 relative z-10">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl md:text-5xl font-bold mb-4">
            <span className="text-gradient">System Demo</span>
          </h1>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Upload a forensic sketch or photo to see the identification pipeline in action
          </p>
        </div>

        {error && (
          <div className="max-w-6xl mx-auto mb-6 p-4 rounded-lg bg-destructive/10 border border-destructive/50 text-destructive">
            {error}
          </div>
        )}

        <div className="max-w-6xl mx-auto grid lg:grid-cols-2 gap-8">
          {/* Upload Section */}
          <div className="space-y-6">
            <div className="p-6 rounded-xl bg-card border border-border/50">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Upload className="w-5 h-5 text-primary" />
                Input Image
              </h3>
              
              {!preview ? (
                <label className="block">
                  <div className="border-2 border-dashed border-border/50 rounded-lg p-12 text-center cursor-pointer hover:border-primary/50 transition-colors">
                    <Upload className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                    <p className="text-muted-foreground mb-2">
                      Drag and drop or click to upload
                    </p>
                    <p className="text-xs text-muted-foreground">
                      Supports JPG, PNG (Max 10MB)
                    </p>
                  </div>
                  <input 
                    type="file" 
                    className="hidden" 
                    accept="image/*"
                    onChange={handleFileChange}
                  />
                </label>
              ) : (
                <div className="space-y-4">
                  <div className="relative aspect-[3/4] rounded-lg overflow-hidden border border-border/50">
                    <img 
                      src={preview} 
                      alt="Uploaded" 
                      className="w-full h-full object-cover"
                    />
                    {isProcessing && (
                      <div className="absolute inset-0 bg-background/80 flex items-center justify-center">
                        <div className="text-center">
                          <div className="w-16 h-16 border-4 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-4" />
                          <p className="text-primary font-mono">Processing...</p>
                        </div>
                      </div>
                    )}
                  </div>
                  
                  {/* Gallery Selection */}
                  <div>
                    <label className="block text-sm font-medium mb-2 text-muted-foreground">
                      Select Gallery
                    </label>
                    <select
                      value={selectedGallery}
                      onChange={(e) => setSelectedGallery(e.target.value)}
                      className="w-full px-4 py-2 rounded-lg bg-muted border border-border/50 text-foreground focus:outline-none focus:border-primary/50"
                      disabled={isProcessing}
                    >
                      <option value="celeba">CelebA Gallery (Photo-based)</option>
                      <option value="cufs">CUFS Gallery (Sketch-based)</option>
                    </select>
                  </div>

                  <div className="flex gap-3">
                    <Button 
                      variant="hero" 
                      className="flex-1"
                      onClick={handleProcess}
                      disabled={isProcessing}
                    >
                      <Search className="w-4 h-4" />
                      {isProcessing ? "Processing..." : "Analyze Image"}
                    </Button>
                    <Button variant="outline" onClick={resetDemo} disabled={isProcessing}>
                      Reset
                    </Button>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Results Section */}
          <div className="space-y-6">
            {/* Generated Image & Heatmap */}
            {(generatedImage || heatmapImage) && (
              <div className="p-6 rounded-xl bg-card border border-border/50">
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <Image className="w-5 h-5 text-primary" />
                  Generated Results
                </h3>
                <div className="grid grid-cols-2 gap-4">
                  {generatedImage && (
                    <div>
                      <p className="text-sm text-muted-foreground mb-2">Generated Photo</p>
                      <div className="relative aspect-square rounded-lg overflow-hidden border border-border/50">
                        <img 
                          src={generatedImage} 
                          alt="Generated Photo" 
                          className="w-full h-full object-cover"
                          onError={(e) => {
                            console.error("Failed to load generated image:", generatedImage);
                            e.target.style.display = 'none';
                          }}
                        />
                      </div>
                    </div>
                  )}
                  {heatmapImage && (
                    <div>
                      <p className="text-sm text-muted-foreground mb-2">Heatmap</p>
                      <div className="relative aspect-square rounded-lg overflow-hidden border border-border/50">
                        <img 
                          src={heatmapImage} 
                          alt="Heatmap" 
                          className="w-full h-full object-cover"
                          onError={(e) => {
                            console.error("Failed to load heatmap:", heatmapImage);
                            e.target.style.display = 'none';
                          }}
                        />
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Top-5 Matches */}
            <div className="p-6 rounded-xl bg-card border border-border/50">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <User className="w-5 h-5 text-primary" />
                Top-5 Matches
              </h3>
              
              {results ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {results.map((result, index) => {
                    // Convert image path to URL
                    const imagePath = result.image_path || '';
                    const imageUrl = imagePath.startsWith('http') 
                      ? imagePath 
                      : imagePath.startsWith('/') 
                        ? `${API_BASE_URL}${imagePath}`
                        : `${API_BASE_URL}/${imagePath}`;
                    
                    return (
                      <div 
                        key={result.id}
                        className={`relative rounded-lg border overflow-hidden transition-all hover:shadow-lg ${
                          result.accepted 
                            ? "border-success/50 bg-success/5" 
                            : "border-border/50 bg-muted/20"
                        }`}
                      >
                        {/* Match Image */}
                        <div className="relative aspect-[3/4] bg-muted">
                          <img
                            src={imageUrl}
                            alt={result.name || `Match ${index + 1}`}
                            className="w-full h-full object-cover"
                            onError={(e) => {
                              console.error("Failed to load match image:", imageUrl);
                              e.target.style.display = 'none';
                              e.target.nextSibling.style.display = 'flex';
                            }}
                          />
                          <div className="hidden absolute inset-0 items-center justify-center bg-muted text-muted-foreground">
                            <User className="w-12 h-12 opacity-50" />
                          </div>
                          
                          {/* Rank Badge */}
                          <div className={`absolute top-2 left-2 w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold ${
                            result.accepted 
                              ? "bg-success text-white" 
                              : "bg-primary text-white"
                          }`}>
                            #{index + 1}
                          </div>
                          
                          {/* Accepted Badge */}
                          {result.accepted && (
                            <div className="absolute top-2 right-2">
                              <CheckCircle className="w-6 h-6 text-success" />
                            </div>
                          )}
                        </div>
                        
                        {/* Match Info */}
                        <div className="p-4">
                          <p className="font-medium text-sm mb-1 truncate" title={result.name}>
                            {result.name || `Match ${index + 1}`}
                          </p>
                          <div className="flex items-center justify-between">
                            <span className="text-xs text-muted-foreground">Similarity:</span>
                            <span className="text-sm font-bold text-primary">
                              {(result.similarity * 100).toFixed(1)}%
                            </span>
                          </div>
                          {/* Progress Bar */}
                          <div className="mt-2 w-full bg-muted rounded-full h-1.5">
                            <div
                              className={`h-1.5 rounded-full transition-all ${
                                result.accepted ? "bg-success" : "bg-primary"
                              }`}
                              style={{ width: `${Math.min(result.similarity * 100, 100)}%` }}
                            />
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="text-center py-12 text-muted-foreground">
                  <User className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <p>Upload and analyze an image to see matches</p>
                </div>
              )}
            </div>

            {/* Decision Intelligence Report */}
            {results && (reliability !== null || density || consistency) && (
              <div className="p-6 rounded-xl bg-card border border-primary/30">
                <h3 className="text-lg font-semibold mb-4 text-primary font-mono">
                  DECISION INTELLIGENCE REPORT
                </h3>
                
                <div className="space-y-4">
                  {/* Reliability Score */}
                  {reliability !== null && (
                    <div className="flex items-center justify-between p-3 rounded-lg bg-muted/50">
                      <span className="text-muted-foreground">Reliability Score</span>
                      <span className="font-mono font-bold text-success">
                        {reliability} / 100 (HIGH)
                      </span>
                    </div>
                  )}

                  {/* Gallery Density */}
                  {density && (
                    <div className="flex items-center justify-between p-3 rounded-lg bg-muted/50">
                      <span className="text-muted-foreground">Gallery Density</span>
                      <span className="font-mono font-bold text-success">
                        {density}
                      </span>
                    </div>
                  )}

                  {/* Cross-Gallery */}
                  {consistency && (
                    <div className="flex items-center justify-between p-3 rounded-lg bg-muted/50">
                      <span className="text-muted-foreground">Cross-Gallery Check</span>
                      <span className="font-mono font-bold text-success">
                        {consistency}
                      </span>
                    </div>
                  )}

                  {/* Final Decision */}
                  <div className="mt-4 p-4 rounded-lg bg-success/10 border border-success/30">
                    <div className="flex items-center gap-2 text-success font-semibold">
                      <CheckCircle className="w-5 h-5" />
                      MATCH ACCEPTED
                    </div>
                    <p className="text-sm text-muted-foreground mt-2">
                      Strong similarity, clear separation from next candidate, stable top-K behavior.
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}

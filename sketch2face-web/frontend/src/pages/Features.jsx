import { ArrowUp, Grid, Link2, CheckCircle, AlertTriangle } from "lucide-react";

export default function Features() {
  return (
    <main className="min-h-screen pt-24 pb-16">
      <div className="absolute inset-0 grid-pattern opacity-30" />
      
      <div className="container mx-auto px-4 relative z-10 max-w-7xl">
        {/* Header */}
        <div className="text-center mb-16">
          <h1 className="text-4xl md:text-5xl font-bold mb-4">
            <span className="text-gradient">AI Features & Decision Intelligence</span>
          </h1>
          <p className="text-lg text-muted-foreground max-w-3xl mx-auto">
            Most systems say "Match Found". Our system says "Match Found, and here's why it's reliable."
          </p>
        </div>

        {/* Feature Cards */}
        <div className="grid md:grid-cols-3 gap-6 mb-12">
          {/* MRS Card */}
          <div className="p-6 rounded-xl bg-card border border-border/50">
            <div className="w-12 h-12 rounded-full bg-primary/20 flex items-center justify-center mb-4">
              <ArrowUp className="w-6 h-6 text-primary" />
            </div>
            <h3 className="text-xl font-semibold mb-3">Match Reliability Scoring (MRS)</h3>
            <p className="text-sm text-muted-foreground mb-4">
              Converts raw similarity scores into a 0-100 reliability score based on multiple factors.
            </p>
            
            <div className="space-y-2 mb-4">
              <div className="flex items-center gap-2 text-sm">
                <ArrowUp className="w-4 h-4 text-primary" />
                <span>Top-1 similarity</span>
              </div>
              <div className="flex items-center gap-2 text-sm">
                <ArrowUp className="w-4 h-4 text-primary" />
                <span>Confidence margin</span>
              </div>
              <div className="flex items-center gap-2 text-sm">
                <ArrowUp className="w-4 h-4 text-primary" />
                <span>Top-K consistency</span>
              </div>
              <div className="flex items-center gap-2 text-sm">
                <ArrowUp className="w-4 h-4 text-primary" />
                <span>Gallery size penalty</span>
              </div>
            </div>

            <div className="p-4 rounded-lg bg-muted/50 border border-border/30">
              <p className="text-sm text-muted-foreground mb-2">Reliability Score:</p>
              <p className="text-lg font-mono font-bold text-success">84 / 100 (HIGH)</p>
              <div className="mt-3 space-y-1 text-xs text-muted-foreground">
                <p>• Strong similarity</p>
                <p>• Clear separation from next candidate</p>
                <p>• Stable top-K behavior</p>
              </div>
            </div>
          </div>

          {/* GDA Card */}
          <div className="p-6 rounded-xl bg-card border border-border/50">
            <div className="w-12 h-12 rounded-full bg-primary/20 flex items-center justify-center mb-4">
              <Grid className="w-6 h-6 text-primary" />
            </div>
            <h3 className="text-xl font-semibold mb-3">Gallery Density Awareness (GDA)</h3>
            <p className="text-sm text-muted-foreground mb-4">
              Detects whether the identity lies in a crowded facial region to assess false positive risk.
            </p>
            
            <div className="space-y-2 mb-4">
              <div className="flex items-center gap-2 text-sm">
                <AlertTriangle className="w-4 h-4 text-warning" />
                <span>Dense clusters → higher false positive risk</span>
              </div>
              <div className="flex items-center gap-2 text-sm">
                <CheckCircle className="w-4 h-4 text-success" />
                <span>Sparse regions → safer decisions</span>
              </div>
            </div>

            <div className="p-4 rounded-lg bg-muted/50 border border-border/30">
              <p className="text-sm text-muted-foreground mb-2">Gallery Density:</p>
              <p className="text-lg font-mono font-bold text-success">LOW</p>
              <p className="text-xs text-muted-foreground mt-2">Risk: Minimal identity confusion</p>
            </div>
          </div>

          {/* CGCC Card */}
          <div className="p-6 rounded-xl bg-card border border-border/50">
            <div className="w-12 h-12 rounded-full bg-primary/20 flex items-center justify-center mb-4">
              <Link2 className="w-6 h-6 text-primary" />
            </div>
            <h3 className="text-xl font-semibold mb-3">Cross-Gallery Consistency Check (CGCC)</h3>
            <p className="text-sm text-muted-foreground mb-4">
              Verifies whether identity behaves consistently across different datasets.
            </p>
            
            <div className="mb-4 p-3 rounded-lg bg-muted/30 border border-border/20">
              <p className="text-xs text-muted-foreground mb-2">Advanced Validation:</p>
              <p className="text-xs">If identity is inconsistent across galleries → system warns, doesn't blindly accept.</p>
            </div>

            <div className="p-4 rounded-lg bg-muted/50 border border-border/30">
              <p className="text-xs text-muted-foreground mb-2">Similarity Details:</p>
              <div className="space-y-1 text-xs font-mono">
                <p>CelebA similarity: 0.74</p>
                <p>CUFS similarity: 0.68</p>
                <p>Gap: 0.06</p>
                <p className="text-success font-bold mt-2">→ CONSISTENT</p>
              </div>
            </div>
          </div>
        </div>

        {/* Final Decision Logic */}
        <div className="p-8 rounded-xl bg-card border border-primary/30">
          <h2 className="text-3xl font-bold mb-6">
            <span className="text-gradient">Final Decision Logic</span>
          </h2>
          
          <div className="grid md:grid-cols-2 gap-8">
            {/* Match Accepted When */}
            <div>
              <h3 className="text-xl font-semibold text-success mb-4 flex items-center gap-2">
                <CheckCircle className="w-6 h-6" />
                Match Accepted When:
              </h3>
              <div className="space-y-3">
                <div className="flex items-center gap-3 p-3 rounded-lg bg-success/10 border border-success/30">
                  <CheckCircle className="w-5 h-5 text-success flex-shrink-0" />
                  <span className="text-sm">Reliability ≥ 70</span>
                </div>
                <div className="flex items-center gap-3 p-3 rounded-lg bg-success/10 border border-success/30">
                  <CheckCircle className="w-5 h-5 text-success flex-shrink-0" />
                  <span className="text-sm">Density Risk ≠ HIGH</span>
                </div>
                <div className="flex items-center gap-3 p-3 rounded-lg bg-success/10 border border-success/30">
                  <CheckCircle className="w-5 h-5 text-success flex-shrink-0" />
                  <span className="text-sm">Cross-Gallery ≠ INCONSISTENT</span>
                </div>
              </div>
            </div>

            {/* Otherwise System Will */}
            <div>
              <h3 className="text-xl font-semibold text-warning mb-4 flex items-center gap-2">
                <AlertTriangle className="w-6 h-6" />
                Otherwise System Will:
              </h3>
              <div className="space-y-3">
                <div className="flex items-center gap-3 p-3 rounded-lg bg-warning/10 border border-warning/30">
                  <AlertTriangle className="w-5 h-5 text-warning flex-shrink-0" />
                  <span className="text-sm">Show results with caution</span>
                </div>
                <div className="flex items-center gap-3 p-3 rounded-lg bg-warning/10 border border-warning/30">
                  <AlertTriangle className="w-5 h-5 text-warning flex-shrink-0" />
                  <span className="text-sm">Flag uncertainty explicitly</span>
                </div>
                <div className="flex items-center gap-3 p-3 rounded-lg bg-warning/10 border border-warning/30">
                  <AlertTriangle className="w-5 h-5 text-warning flex-shrink-0" />
                  <span className="text-sm">Prevent blind trust</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}

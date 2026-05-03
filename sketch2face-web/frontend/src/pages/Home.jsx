import { Link } from "react-router-dom";
import { Button } from "../components/ui/Button";
import { Search, Brain, ArrowRight, PenTool, ArrowLeftRight, BrainCircuit, Shield } from "lucide-react";

const customStyles = `
  @keyframes scan-line {
    0% { top: -5%; opacity: 0; }
    15% { opacity: 1; }
    85% { opacity: 1; }
    100% { top: 105%; opacity: 0; }
  }
  @keyframes sketch-fade {
    0%, 35% { opacity: 1; filter: sepia(0.2) contrast(1.1); }
    65%, 100% { opacity: 0; filter: sepia(0) contrast(1); }
  }
`;

export function HeroSection() {
  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden pt-16">
      <style>{customStyles}</style>
      {/* Background Grid Pattern */}
      <div className="absolute inset-0 grid-pattern opacity-50" />
      
      {/* Animated Background Orbs */}
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary/10 rounded-full blur-3xl animate-pulse" />
      <div className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-accent/10 rounded-full blur-3xl animate-pulse" style={{ animationDelay: "1s" }} />

      <div className="container mx-auto px-4 relative z-10">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          {/* Left: Text Content */}
          <div className="text-center lg:text-left space-y-8">
            {/* Badge */}
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-primary/30 bg-primary/5 text-primary text-sm font-mono">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-primary" />
              </span>
              AI-POWERED BIOMETRIC SYSTEM
            </div>

            {/* Title */}
            <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold leading-tight">
              <span className="text-foreground">Hybrid </span>
              <span className="text-gradient">Sketch-to-Face</span>
              <br />
              <span className="text-foreground">Recognition System</span>
            </h1>

            {/* Subtitle */}
            <p className="text-lg md:text-xl text-muted-foreground max-w-xl mx-auto lg:mx-0">
              An AI-powered biometric system for identifying individuals from forensic sketches and images with{" "}
              <span className="text-primary font-medium">explainable confidence</span> and{" "}
              <span className="text-accent font-medium">reliability analysis</span>.
            </p>

            {/* CTA Buttons */}
            <div className="flex flex-col sm:flex-row gap-4 justify-center lg:justify-start">
              <Link to="/match">
                <Button variant="hero" size="xl" className="w-full sm:w-auto">
                  <Search className="h-5 w-5" />
                  Try Demo
                </Button>
              </Link>
              <Link to="/how-it-works">
                <Button variant="heroOutline" size="xl" className="w-full sm:w-auto">
                  <Brain className="h-5 w-5" />
                  How It Works
                  <ArrowRight className="h-4 w-4" />
                </Button>
              </Link>
            </div>
          </div>

          {/* Right: Face Scanning Interface */}
          <div className="relative flex justify-center">
            <div className="relative w-80 h-96 lg:w-96 lg:h-[450px]">
              {/* Scanner Frame */}
              <div className="absolute inset-0 border-2 border-primary/40 rounded-lg overflow-hidden bg-card/50 backdrop-blur-sm">
                {/* Corner Markers */}
                <div className="absolute top-2 left-2 w-6 h-6 border-l-2 border-t-2 border-primary z-10" />
                <div className="absolute top-2 right-2 w-6 h-6 border-r-2 border-t-2 border-primary z-10" />
                <div className="absolute bottom-2 left-2 w-6 h-6 border-l-2 border-b-2 border-primary z-10" />
                <div className="absolute bottom-2 right-2 w-6 h-6 border-r-2 border-b-2 border-primary z-10" />

                {/* Sketch-to-Face Reveal Animation Display */}
                <div className="absolute inset-4 flex items-center justify-center overflow-hidden rounded bg-black/60 ring-1 ring-primary/40 shadow-inner">
                  
                  {/* Photo Base Layer (Revealed when sketch fades) */}
                  <img 
                    src="/images/real_photo.png" 
                    alt="Matched Target"
                    className="absolute inset-0 w-full h-full object-cover"
                  />

                  {/* Sketch Layer (Fades out mid-scan) */}
                  <img 
                    src="/images/forensic_sketch.png" 
                    alt="Forensic Sketch"
                    className="absolute inset-0 w-full h-full object-cover"
                    style={{ animation: 'sketch-fade 4s ease-in-out infinite alternate' }}
                  />

                  {/* Scanning Laser Line Tracker */}
                  <div className="absolute w-full h-[2px] bg-cyan-400 shadow-[0_0_20px_4px_rgba(34,211,238,0.8)] z-20 left-0"
                       style={{ animation: 'scan-line 4s linear infinite alternate' }} />
                       
                  {/* Digital Biometric Grid Overlay Filter */}
                  <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyMCIgaGVpZ2h0PSIyMCI+CiAgPHJlY3Qgd2lkdGg9IjIwIiBoZWlnaHQ9IjIwIiBmaWxsPSJub25lIiBzdHJva2U9InJnYmEoOTksIDEwMiwgMjQxLCAwLjE1KSIgc3Ryb2tlLXdpZHRoPSIxIiAvPgo8L3N2Zz4=')] mix-blend-screen opacity-50 z-10 pointer-events-none" />
                </div>

                {/* Status Display */}
                <div className="absolute bottom-4 left-4 right-4 z-10">
                  <div className="bg-background/90 rounded px-3 py-2 font-mono text-xs border border-primary/20">
                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">STATUS:</span>
                      <span className="text-primary animate-pulse">SCANNING...</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Glow Effect */}
              <div className="absolute -inset-4 bg-primary/20 blur-2xl rounded-full opacity-50 animate-pulse" />
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

export function HighlightsSection() {
  const capabilities = [
    {
      icon: <PenTool className="w-8 h-8" />,
      title: "Sketch-to-Face Recognition",
      description: "Converts forensic sketches into identifiable faces using AI.",
      color: "text-primary"
    },
    {
      icon: <ArrowLeftRight className="w-8 h-8" />,
      title: "Cross-Domain Matching",
      description: "Works across forensic (CUFS) and real-world (CelebA) datasets.",
      color: "text-accent"
    },
    {
      icon: <BrainCircuit className="w-8 h-8" />,
      title: "Explainable AI Decisions",
      description: "Every match is justified with reliability and risk analysis.",
      color: "text-primary"
    },
    {
      icon: <Shield className="w-8 h-8" />,
      title: "Law-Enforcement Ready",
      description: "Role-based access and case-based identification workflow.",
      color: "text-accent",
      highlighted: true
    },
  ];

  return (
    <section className="py-24 relative">
      <div className="container mx-auto px-4">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold mb-4">
            <span className="text-gradient">Key Capabilities</span>
          </h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Advanced forensic identification powered by state-of-the-art AI
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 max-w-7xl mx-auto">
          {capabilities.map((cap, index) => (
            <div
              key={index}
              className={`p-6 rounded-xl bg-card border transition-all ${
                cap.highlighted
                  ? "border-primary/50 bg-primary/5 shadow-lg shadow-primary/20"
                  : "border-border/50 hover:border-primary/30"
              }`}
            >
              <div className={`${cap.color} mb-4`}>
                {cap.icon}
              </div>
              <h3 className="text-lg font-semibold mb-2">{cap.title}</h3>
              <p className="text-sm text-muted-foreground">{cap.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

export default function Home() {
  return (
    <main>
      <HeroSection />
      <HighlightsSection />
    </main>
  );
}

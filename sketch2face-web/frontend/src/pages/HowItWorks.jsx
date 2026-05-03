import { Upload, FileCheck, Image, Scan, Fingerprint, Network, Brain, MessageSquare } from "lucide-react";

const steps = [
  {
    number: 1,
    icon: <Upload className="w-6 h-6" />,
    title: "Input Image (Sketch / Photo)",
    description: "Accepts forensic sketches, witness sketches, or photos from both controlled (CUFS) and uncontrolled (CelebA) domains."
  },
  {
    number: 2,
    icon: <FileCheck className="w-6 h-6" />,
    title: "Style Validation & Normalization",
    description: "Automatically checks whether the sketch follows forensic standards. Non-standard sketches are normalized for consistency."
  },
  {
    number: 3,
    icon: <Image className="w-6 h-6" />,
    title: "Face Generation (CUFS only)",
    description: "For forensic sketches, a Pix2Pix-based model generates a photo-like face. For real photos (CelebA), generation is skipped to avoid distortion."
  },
  {
    number: 4,
    icon: <Scan className="w-6 h-6" />,
    title: "Face Detection & Alignment",
    description: "Detects facial region and aligns it for biometric comparison. Uses fallback cropping to prevent rejection due to detector failure."
  },
  {
    number: 5,
    icon: <Fingerprint className="w-6 h-6" />,
    title: "ArcFace Embedding Extraction",
    description: "Uses ArcFace to extract a 512-D identity embedding. Normalized embeddings ensure scale-invariant comparison."
  },
  {
    number: 6,
    icon: <Network className="w-6 h-6" />,
    title: "Similarity Matching (Cosine)",
    description: "Computes cosine similarity against gallery embeddings. Retrieves Top-K candidates instead of a single guess."
  },
  {
    number: 7,
    icon: <Brain className="w-6 h-6" />,
    title: "Decision Intelligence Layer",
    description: "AI-powered analysis that evaluates match reliability, gallery density, and cross-gallery consistency before final decision."
  },
  {
    number: 8,
    icon: <MessageSquare className="w-6 h-6" />,
    title: "Final Result + Explanation",
    description: "Presents results with full transparency, showing confidence scores, risk factors, and reasoning behind each match."
  },
];

export default function HowItWorks() {
  return (
    <main className="min-h-screen pt-24 pb-16">
      <div className="absolute inset-0 grid-pattern opacity-30" />
      
      <div className="container mx-auto px-4 relative z-10">
        <div className="text-center mb-16">
          <h1 className="text-4xl md:text-5xl font-bold mb-4">
            <span className="text-gradient">How Our System Works</span>
          </h1>
          <p className="text-lg text-muted-foreground max-w-3xl mx-auto">
            A comprehensive pipeline from forensic sketch input to explainable identification results
          </p>
        </div>

        <div className="max-w-4xl mx-auto space-y-8">
          {steps.map((step, index) => (
            <div key={step.number} className="flex gap-6">
              {/* Step Number & Icon */}
              <div className="flex-shrink-0">
                <div className="w-16 h-16 rounded-full bg-primary/20 border-2 border-primary/50 flex items-center justify-center relative">
                  <div className="absolute inset-0 rounded-full bg-primary/10 animate-pulse" />
                  <div className="relative z-10 text-primary">
                    {step.icon}
                  </div>
                  <div className="absolute -top-2 -right-2 w-8 h-8 rounded-full bg-primary text-primary-foreground flex items-center justify-center text-sm font-bold border-2 border-background">
                    {step.number}
                  </div>
                </div>
                {index < steps.length - 1 && (
                  <div className="w-0.5 h-16 bg-primary/30 mx-auto mt-4" />
                )}
              </div>

              {/* Step Content */}
              <div className="flex-1 pb-8">
                <div className="p-6 rounded-xl bg-card border border-border/50">
                  <h3 className="text-xl font-semibold mb-3">{step.title}</h3>
                  <p className="text-muted-foreground">{step.description}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </main>
  );
}

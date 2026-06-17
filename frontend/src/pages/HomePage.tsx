import { useState, useCallback } from "react";
import { Upload, FileAudio, Check, AlertCircle, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { cn, formatDuration } from "@/lib/utils";
import { api } from "@/lib/api";
import { useAuthStore } from "@/stores/authStore";

const MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024; // 2GB
const ALLOWED_FORMATS = ["audio/wav", "audio/flac", "audio/aiff", "audio/mp3", "audio/x-wav", "audio/x-aiff"];

function calculateCredits(durationSeconds: number): number {
  if (durationSeconds <= 600) return 1;
  if (durationSeconds <= 1200) return 2;
  if (durationSeconds <= 1800) return 3;
  return Math.floor(durationSeconds / 600) + 1;
}

export function HomePage() {
  const { user, isAuthenticated } = useAuthStore();
  const [file, setFile] = useState<File | null>(null);
  const [duration, setDuration] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadedJobId, setUploadedJobId] = useState<string | null>(null);

  const estimatedCredits = duration ? calculateCredits(duration) : 1;
  const canProcess = isAuthenticated && user && user.creditBalance >= estimatedCredits && file && !isUploading;

  const handleFile = useCallback(async (f: File) => {
    setError(null);

    if (f.size > MAX_FILE_SIZE) {
      setError("File too large. Maximum size is 2GB.");
      return;
    }

    const audioTypes = ["audio/wav", "audio/flac", "audio/aiff", "audio/mp3", "audio/x-wav", "audio/x-aiff", "audio/wave"];
    if (!audioTypes.includes(f.type) && !f.name.match(/\.(wav|flac|aiff|mp3)$/i)) {
      setError("Invalid format. Use WAV, FLAC, AIFF, or MP3.");
      return;
    }

    setFile(f);

    // Get duration using audio element
    const audio = new Audio();
    audio.src = URL.createObjectURL(f);
    audio.onloadedmetadata = () => {
      setDuration(audio.duration);
    };
    audio.onerror = () => {
      // If we can't get duration, assume 1 credit minimum
      setDuration(60);
    };
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }, [handleFile]);

  const handleSubmit = async () => {
    if (!canProcess || !file || !user) return;

    setIsUploading(true);
    setUploadProgress(0);

    try {
      const { job_id } = await api.submitJob(user.id, file);
      setUploadedJobId(job_id);
      setUploadProgress(100);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative py-20 lg:py-32 px-4">
        <div className="container mx-auto max-w-4xl text-center">
          <Badge variant="secondary" className="mb-4">
            AI-Powered Audio Processing
          </Badge>
          <h1 className="text-4xl font-bold tracking-tight lg:text-6xl">
            Stem separation + MIDI for guitar & bass
          </h1>
          <p className="mt-6 text-xl text-muted-foreground max-w-2xl mx-auto">
            With bend detection. Upload your track, get studio-quality stems and
            editable MIDI — ready for your DAW.
          </p>
          <div className="mt-8 flex justify-center gap-4">
            <Button size="lg" asChild>
              <a href="#upload">Try Free</a>
            </Button>
            <Button size="lg" variant="outline" asChild>
              <a href="/pricing">View Pricing</a>
            </Button>
          </div>
        </div>
      </section>

      {/* Upload Section */}
      <section id="upload" className="py-12 px-4">
        <div className="container mx-auto max-w-2xl">
          <Card>
            <CardHeader>
              <CardTitle>Upload Audio</CardTitle>
              <CardDescription>
                WAV, FLAC, AIFF, or MP3 up to 2GB
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {!isAuthenticated ? (
                <div className="text-center py-8">
                  <p className="text-muted-foreground mb-4">
                    Sign in to upload and process your audio
                  </p>
                  <Button asChild>
                    <a href="/account">Sign In or Create Account</a>
                  </Button>
                </div>
              ) : (
                <>
                  {/* Drop Zone */}
                  <div
                    className={cn(
                      "border-2 border-dashed rounded-lg p-8 text-center transition-colors cursor-pointer",
                      isDragging ? "border-primary bg-primary/5" : "border-muted-foreground/25",
                      file && "border-green-500 bg-green-500/5"
                    )}
                    onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                    onDragLeave={() => setIsDragging(false)}
                    onDrop={handleDrop}
                    onClick={() => document.getElementById("file-input")?.click()}
                  >
                    <input
                      id="file-input"
                      type="file"
                      accept=".wav,.flac,.aiff,.mp3,audio/*"
                      className="hidden"
                      onChange={(e) => {
                        const f = e.target.files?.[0];
                        if (f) handleFile(f);
                      }}
                    />
                    {file ? (
                      <div className="flex items-center justify-center gap-3">
                        <FileAudio className="h-8 w-8 text-green-500" />
                        <div className="text-left">
                          <p className="font-medium">{file.name}</p>
                          <p className="text-sm text-muted-foreground">
                            {(file.size / (1024 * 1024)).toFixed(1)} MB
                            {duration && ` • ${formatDuration(duration)}`}
                          </p>
                        </div>
                        <Check className="h-5 w-5 text-green-500" />
                      </div>
                    ) : (
                      <>
                        <Upload className="h-10 w-10 mx-auto text-muted-foreground mb-3" />
                        <p className="text-muted-foreground">
                          Drag & drop your audio file here, or click to browse
                        </p>
                      </>
                    )}
                  </div>

                  {/* Error */}
                  {error && (
                    <div className="flex items-center gap-2 text-sm text-destructive">
                      <AlertCircle className="h-4 w-4" />
                      {error}
                    </div>
                  )}

                  {/* Cost Estimate */}
                  {file && duration && (
                    <div className="flex items-center justify-between p-4 bg-muted rounded-lg">
                      <div>
                        <p className="font-medium">Estimated Cost</p>
                        <p className="text-sm text-muted-foreground">
                          {duration > 600 && `${formatDuration(duration)} audio = `}
                          {estimatedCredits} credit{estimatedCredits > 1 ? "s" : ""}
                        </p>
                      </div>
                      <div className="text-right">
                        <p className="text-2xl font-bold">{estimatedCredits}</p>
                        <p className="text-sm text-muted-foreground">
                          of {user.creditBalance} available
                        </p>
                      </div>
                    </div>
                  )}

                  {/* Upload Progress */}
                  {isUploading && (
                    <div className="space-y-2">
                      <div className="flex items-center gap-2">
                        <Loader2 className="h-4 w-4 animate-spin" />
                        <span className="text-sm">Uploading...</span>
                      </div>
                      <Progress value={uploadProgress} />
                    </div>
                  )}

                  {/* Submit Button */}
                  <Button
                    className="w-full"
                    size="lg"
                    disabled={!canProcess}
                    onClick={handleSubmit}
                  >
                    {isUploading ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Processing...
                      </>
                    ) : !file ? (
                      "Select a file to continue"
                    ) : user && user.creditBalance < estimatedCredits ? (
                      `Need ${estimatedCredits} credits (have ${user.creditBalance})`
                    ) : (
                      "Process Audio"
                    )}
                  </Button>

                  {/* Job ID display */}
                  {uploadedJobId && (
                    <div className="p-4 bg-green-500/10 rounded-lg">
                      <p className="font-medium text-green-700 dark:text-green-400">
                        Upload successful! Job ID: {uploadedJobId}
                      </p>
                      <p className="text-sm text-muted-foreground mt-1">
                        Check your email when processing completes, or view status in your account.
                      </p>
                    </div>
                  )}
                </>
              )}
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-16 px-4 bg-muted/50">
        <div className="container mx-auto max-w-4xl">
          <h2 className="text-3xl font-bold text-center mb-12">How It Works</h2>
          <div className="grid md:grid-cols-3 gap-8">
            <div className="text-center">
              <div className="h-12 w-12 mx-auto mb-4 rounded-full bg-primary/10 flex items-center justify-center">
                <Upload className="h-6 w-6 text-primary" />
              </div>
              <h3 className="font-semibold mb-2">1. Upload</h3>
              <p className="text-sm text-muted-foreground">
                Drag & drop your WAV, FLAC, AIFF, or MP3 file
              </p>
            </div>
            <div className="text-center">
              <div className="h-12 w-12 mx-auto mb-4 rounded-full bg-primary/10 flex items-center justify-center">
                <FileAudio className="h-6 w-6 text-primary" />
              </div>
              <h3 className="font-semibold mb-2">2. AI Processing</h3>
              <p className="text-sm text-muted-foreground">
                Stem separation + bend detection runs on our servers
              </p>
            </div>
            <div className="text-center">
              <div className="h-12 w-12 mx-auto mb-4 rounded-full bg-primary/10 flex items-center justify-center">
                <Check className="h-6 w-6 text-primary" />
              </div>
              <h3 className="font-semibold mb-2">3. Download</h3>
              <p className="text-sm text-muted-foreground">
                Get stems + MIDI, with bend/vibrato/slide data
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
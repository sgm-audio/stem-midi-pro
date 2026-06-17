import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Loader2, Plus, ArrowRight, Music, Clock, CheckCircle, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { useAuthStore } from "@/stores/authStore";
import { api, Transaction } from "@/lib/api";

export function AccountPage() {
  const { user, isAuthenticated, setUser, updateCreditBalance, logout } = useAuthStore();
  const navigate = useNavigate();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [authError, setAuthError] = useState<string | null>(null);
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [isLoadingTransactions, setIsLoadingTransactions] = useState(false);

  useEffect(() => {
    if (isAuthenticated && user) {
      loadCredits();
      loadTransactions();
    }
  }, [isAuthenticated, user?.id]);

  const loadCredits = async () => {
    if (!user) return;
    try {
      const data = await api.getCredits(user.id);
      updateCreditBalance(data.balance);
    } catch (err) {
      console.error("Failed to load credits:", err);
    }
  };

  const loadTransactions = async () => {
    if (!user) return;
    setIsLoadingTransactions(true);
    try {
      const data = await api.getTransactions(user.id);
      setTransactions(data.transactions);
    } catch (err) {
      console.error("Failed to load transactions:", err);
    } finally {
      setIsLoadingTransactions(false);
    }
  };

  const handleSignup = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setAuthError(null);

    try {
      // For demo: create local user with a generated ID
      // In production, this would call a proper auth API
      const userId = `user_${Date.now()}`;
      setUser({
        id: userId,
        email,
        creditBalance: 0,
      });
      navigate("/");
    } catch (err) {
      setAuthError(err instanceof Error ? err.message : "Signup failed");
    } finally {
      setIsLoading(false);
    }
  };

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setAuthError(null);

    try {
      // For demo: lookup existing user by email (would be real auth in production)
      const userId = `user_${email.replace(/[^a-z0-9]/gi, "_")}`;
      setUser({
        id: userId,
        email,
        creditBalance: 0,
      });
      navigate("/");
    } catch (err) {
      setAuthError(err instanceof Error ? err.message : "Login failed");
    } finally {
      setIsLoading(false);
    }
  };

  const handleLogout = () => {
    logout();
    navigate("/");
  };

  const handleBuyCredits = (packageId: string) => {
    if (!user) return;
    window.location.href = `/pricing?package=${packageId}`;
  };

  if (!isAuthenticated) {
    return (
      <div className="min-h-[80vh] flex items-center justify-center px-4">
        <Card className="w-full max-w-md">
          <CardHeader>
            <CardTitle>Welcome to Stem+MIDI Pro</CardTitle>
            <CardDescription>
              Sign in to upload audio and process it with AI
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <form onSubmit={handleSignup} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="email">Email</Label>
                <Input
                  id="email"
                  type="email"
                  placeholder="you@example.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="password">Password</Label>
                <Input
                  id="password"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                />
              </div>
              {authError && (
                <p className="text-sm text-destructive">{authError}</p>
              )}
              <Button type="submit" className="w-full" disabled={isLoading}>
                {isLoading ? (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                ) : (
                  "Create Account"
                )}
              </Button>
            </form>
            <div className="relative">
              <div className="absolute inset-0 flex items-center">
                <span className="w-full border-t" />
              </div>
              <div className="relative flex justify-center text-xs uppercase">
                <span className="bg-background px-2 text-muted-foreground">Or</span>
              </div>
            </div>
            <form onSubmit={handleLogin} className="space-y-4">
              <Button type="submit" variant="outline" className="w-full" disabled={isLoading}>
                Sign In
              </Button>
            </form>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="container mx-auto max-w-4xl px-4 py-12">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold">My Account</h1>
          <p className="text-muted-foreground">{user?.email}</p>
        </div>
        <Button variant="outline" onClick={handleLogout}>
          Sign Out
        </Button>
      </div>

      <div className="grid lg:grid-cols-2 gap-8">
        {/* Credits Section */}
        <Card>
          <CardHeader>
            <CardTitle>Credits</CardTitle>
            <CardDescription>
              Purchase credits to process audio files
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="text-center py-6">
              <p className="text-5xl font-bold">{user?.creditBalance || 0}</p>
              <p className="text-muted-foreground">credits available</p>
            </div>
            <Button className="w-full" onClick={() => handleBuyCredits("popular")}>
              Buy Credits <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
          </CardContent>
        </Card>

        {/* Job History Section */}
        <Card>
          <CardHeader>
            <CardTitle>Recent Jobs</CardTitle>
            <CardDescription>Your last 20 processing jobs</CardDescription>
          </CardHeader>
          <CardContent>
            {isLoadingTransactions ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              </div>
            ) : transactions.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                <Music className="h-10 w-10 mx-auto mb-3 opacity-50" />
                <p>No jobs yet</p>
                <p className="text-sm">Upload an audio file to get started</p>
              </div>
            ) : (
              <div className="space-y-3">
                {transactions.map((tx) => (
                  <div
                    key={tx.id}
                    className="flex items-center justify-between p-3 bg-muted rounded-lg"
                  >
                    <div className="flex items-center gap-3">
                      {tx.transaction_type === "spend" ? (
                        <div className="h-8 w-8 rounded-full bg-orange-100 dark:bg-orange-900/30 flex items-center justify-center">
                          <Music className="h-4 w-4 text-orange-600" />
                        </div>
                      ) : (
                        <div className="h-8 w-8 rounded-full bg-green-100 dark:bg-green-900/30 flex items-center justify-center">
                          <Plus className="h-4 w-4 text-green-600" />
                        </div>
                      )}
                      <div>
                        <p className="font-medium text-sm">
                          {tx.description || (tx.transaction_type === "spend" ? "Job Processing" : "Credit Purchase")}
                        </p>
                        <p className="text-xs text-muted-foreground">
                          {new Date(tx.created_at).toLocaleDateString()}
                        </p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className={tx.amount > 0 ? "text-green-600" : "text-foreground"}>
                        {tx.amount > 0 ? "+" : ""}{tx.amount}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        Balance: {tx.balance_after}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
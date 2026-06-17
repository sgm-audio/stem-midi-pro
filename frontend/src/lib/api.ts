const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

interface ApiError {
  detail: string;
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const error: ApiError = await response.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }
  return response.json();
}

export interface CreditPackage {
  id: string;
  name: string;
  price_cents: number;
  price_formatted: string;
  credits: number;
  badge: string | null;
  description: string;
  value_per_credit: string;
}

export interface CreateOrderRequest {
  package_id: string;
  user_id: string;
  user_email?: string;
}

export interface CreateOrderResponse {
  checkout_url: string;
  checkout_id: string;
  package_id: string;
  credits: number;
  amount_cents: number;
}

export interface Transaction {
  id: number;
  amount: number;
  balance_after: number;
  description: string | null;
  transaction_type: string;
  payment_id: string | null;
  job_id: string | null;
  created_at: string;
}

export interface UserCredits {
  balance: number;
  packages: CreditPackage[];
}

export interface JobStatus {
  id: string;
  status: "pending" | "processing" | "completed" | "failed";
  progress?: number;
  result_url?: string;
  quality_tier?: "studio" | "draft" | "complex";
  error?: string;
}

export const api = {
  async getPackages(): Promise<{ packages: CreditPackage[] }> {
    const res = await fetch(`${API_BASE}/api/v1/payments/packages`);
    return handleResponse(res);
  },

  async createOrder(data: CreateOrderRequest): Promise<CreateOrderResponse> {
    const res = await fetch(`${API_BASE}/api/v1/payments/create-order`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    return handleResponse(res);
  },

  async getCredits(userId: string): Promise<UserCredits> {
    const res = await fetch(`${API_BASE}/api/v1/users/me/credits`, {
      headers: { "X-User-ID": userId },
    });
    return handleResponse(res);
  },

  async getTransactions(userId: string, limit = 20, offset = 0): Promise<{
    transactions: Transaction[];
    total: number;
    limit: number;
    offset: number;
  }> {
    const res = await fetch(
      `${API_BASE}/api/v1/users/me/transactions?limit=${limit}&offset=${offset}`,
      { headers: { "X-User-ID": userId } }
    );
    return handleResponse(res);
  },

  async deductCredits(userId: string, jobId: string, creditsAmount: number): Promise<{
    success: boolean;
    balance: number;
    transaction_id: number;
  }> {
    const res = await fetch(`${API_BASE}/api/v1/users/me/credits/deduct`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-User-ID": userId,
      },
      body: JSON.stringify({ job_id: jobId, credits_amount: creditsAmount }),
    });
    return handleResponse(res);
  },

  async submitJob(userId: string, file: File): Promise<{ job_id: string }> {
    const formData = new FormData();
    formData.append("file", file);
    const res = await fetch(`${API_BASE}/api/v1/jobs/submit`, {
      method: "POST",
      headers: { "X-User-ID": userId },
      body: formData,
    });
    return handleResponse(res);
  },

  async getJobStatus(jobId: string): Promise<JobStatus> {
    const res = await fetch(`${API_BASE}/api/v1/jobs/${jobId}/status`);
    return handleResponse(res);
  },
};
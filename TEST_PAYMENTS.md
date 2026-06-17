# Stem+MIDI Pro - Payment API Curl Commands
# For manual testing of Square payment integration

BASE_URL="http://localhost:8000"

# ===========================================
# LIST CREDIT PACKAGES
# ===========================================
echo "=== List Credit Packages ==="
curl -s "$BASE_URL/api/v1/payments/packages" | python -m json.tool

# ===========================================
# CREATE PAYMENT ORDER ($5 Starter Pack)
# ===========================================
echo -e "\n=== Create Order (Starter Pack) ==="
curl -s -X POST "$BASE_URL/api/v1/payments/create-order" \
  -H "Content-Type: application/json" \
  -d '{"package_id": "starter", "user_id": "user_abc123", "user_email": "test@example.com"}' \
  | python -m json.tool

# ===========================================
# CREATE PAYMENT ORDER ($20 Popular Pack)
# ===========================================
echo -e "\n=== Create Order (Popular Pack) ==="
curl -s -X POST "$BASE_URL/api/v1/payments/create-order" \
  -H "Content-Type: application/json" \
  -d '{"package_id": "popular", "user_id": "user_abc123", "user_email": "test@example.com"}' \
  | python -m json.tool

# ===========================================
# CREATE PAYMENT ORDER ($50 Best Value Pack)
# ===========================================
echo -e "\n=== Create Order (Best Value Pack) ==="
curl -s -X POST "$BASE_URL/api/v1/payments/create-order" \
  -H "Content-Type: application/json" \
  -d '{"package_id": "best_value", "user_id": "user_abc123", "user_email": "test@example.com"}' \
  | python -m json.tool

# ===========================================
# GET USER CREDITS
# ===========================================
echo -e "\n=== Get User Credits ==="
curl -s "$BASE_URL/api/v1/users/me/credits" \
  -H "X-User-ID: user_abc123" \
  | python -m json.tool

# ===========================================
# GET USER TRANSACTIONS
# ===========================================
echo -e "\n=== Get User Transactions ==="
curl -s "$BASE_URL/api/v1/users/me/transactions" \
  -H "X-User-ID: user_abc123" \
  | python -m json.tool

# ===========================================
# DEDUCT CREDITS FOR JOB
# ===========================================
echo -e "\n=== Deduct Credits for Job ==="
curl -s -X POST "$BASE_URL/api/v1/users/me/credits/deduct" \
  -H "Content-Type: application/json" \
  -H "X-User-ID: user_abc123" \
  -d '{"job_id": "job_xyz789", "credits_amount": 1}' \
  | python -m json.tool

# ===========================================
# SQUARE WEBHOOK (simulated)
# ===========================================
# Note: Square calls this endpoint after payment completion
# You can test it manually with a mock payload:
echo -e "\n=== Square Webhook (example) ==="
curl -s -X POST "$BASE_URL/api/v1/payments/square/webhook" \
  -H "Content-Type: application/json" \
  -H "x-square-hmacsha256-signature: test_signature" \
  -d '{
    "type": "payment.completed",
    "event_id": "test_event_123",
    "data": {
      "object": {
        "payment": {
          "id": "mock_payment_id_123",
          "order_id": "mock_order_id_456",
          "amount_money": {"amount": 500, "currency": "USD"},
          "receipt_email": "test@example.com",
          "reference_id": "user_abc123"
        }
      }
    }
  }' \
  | python -m json.tool

# ===========================================
# CREDIT CALCULATION VERIFICATION
# ===========================================
# ≤10 min audio = 1 credit
# 10-20 min = 2 credits
# 20-30 min = 3 credits
# 30+ min = floor(duration_sec / 600) + 1
echo -e "\n=== Credit Calculation Examples ==="
echo "5 min (300s) = 1 credit"
echo "10 min (600s) = 1 credit"
echo "10min+1sec (601s) = 2 credits"
echo "20 min (1200s) = 2 credits"
echo "20min+1sec (1201s) = 3 credits"
echo "30 min (1800s) = 3 credits"
echo "40 min (2400s) = 4 credits"
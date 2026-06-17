"""
Tests for payment endpoints and credit service.
"""

from services.credit_service import (
    CreditService,
    CREDIT_PACKAGES,
    calculate_job_credits,
)


class TestCreditService:
    """Tests for credit service."""

    def test_get_or_create_user(self, db_session):
        """Test user creation."""
        service = CreditService(db_session)
        user = service.get_or_create_user("user_abc", "test@example.com")

        assert user.id is not None
        assert user.external_id == "user_abc"
        assert user.email == "test@example.com"
        assert user.credit_balance == 0

    def test_get_or_create_user_existing(self, db_session, test_user):
        """Test getting existing user."""
        service = CreditService(db_session)
        user = service.get_or_create_user("test_user_123")

        assert user.id == test_user.id
        assert user.credit_balance == 100

    def test_add_credits(self, db_session, test_user):
        """Test adding credits."""
        service = CreditService(db_session)
        tx = service.add_credits(
            user_id=test_user.id,
            amount=50,
            payment_id="payment_123",
            description="Test purchase",
        )

        assert tx.amount == 50
        assert tx.balance_after == 150
        assert tx.payment_id == "payment_123"
        assert tx.transaction_type == "purchase"

        db_session.refresh(test_user)
        assert test_user.credit_balance == 150

    def test_spend_credits(self, db_session, test_user):
        """Test spending credits."""
        service = CreditService(db_session)
        tx = service.spend_credits(
            user_id=test_user.id,
            amount=10,
            job_id="job_abc",
            description="Test job",
        )

        assert tx.amount == -10
        assert tx.balance_after == 90
        assert tx.job_id == "job_abc"
        assert tx.transaction_type == "spend"

        db_session.refresh(test_user)
        assert test_user.credit_balance == 90

    def test_spend_credits_insufficient(self, db_session, test_user):
        """Test insufficient balance."""
        service = CreditService(db_session)
        tx = service.spend_credits(
            user_id=test_user.id,
            amount=200,
            job_id="job_abc",
        )

        assert tx is None
        assert test_user.credit_balance == 100

    def test_idempotency_check(self, db_session, test_user):
        """Test payment idempotency check."""
        service = CreditService(db_session)

        service.add_credits(
            user_id=test_user.id,
            amount=50,
            payment_id="same_payment_id",
        )

        assert service.check_payment_already_processed("same_payment_id") is True
        assert service.check_payment_already_processed("different_payment_id") is False

    def test_get_transactions(self, db_session, test_user):
        """Test transaction history."""
        service = CreditService(db_session)

        service.add_credits(test_user.id, 50, "p1")
        service.add_credits(test_user.id, 25, "p2")
        service.spend_credits(test_user.id, 10, "j1")

        txs = service.get_transactions(test_user.id)
        assert len(txs) == 3
        assert txs[0].payment_id == "p2"
        assert txs[1].payment_id == "p1"
        assert txs[2].job_id == "j1"


class TestCreditPackages:
    """Tests for credit package calculations."""

    def test_package_values(self):
        """Test package definitions."""
        assert CREDIT_PACKAGES["starter"].credits == 10
        assert CREDIT_PACKAGES["starter"].price_cents == 500

        assert CREDIT_PACKAGES["popular"].credits == 50
        assert CREDIT_PACKAGES["popular"].price_cents == 2000

        assert CREDIT_PACKAGES["best_value"].credits == 150
        assert CREDIT_PACKAGES["best_value"].price_cents == 5000

    def test_calculate_job_credits(self):
        """Test credit calculation based on duration."""
        assert calculate_job_credits(300) == 1  # 5 min
        assert calculate_job_credits(600) == 1  # 10 min exactly
        assert calculate_job_credits(601) == 2  # 10 min + 1 sec
        assert calculate_job_credits(1200) == 2  # 20 min exactly
        assert calculate_job_credits(1201) == 3  # 20 min + 1 sec
        assert calculate_job_credits(1800) == 3  # 30 min exactly
        assert calculate_job_credits(2400) == 4  # 40 min


class TestPaymentEndpoints:
    """Tests for payment API endpoints."""

    def test_list_packages(self, client):
        """Test GET /api/v1/payments/packages."""
        response = client.get("/api/v1/payments/packages")
        assert response.status_code == 200

        data = response.json()
        assert "packages" in data
        assert len(data["packages"]) == 3

        starter = next(p for p in data["packages"] if p["id"] == "starter")
        assert starter["credits"] == 10
        assert starter["price_formatted"] == "$5.00"

    def test_get_credits(self, client, test_user):
        """Test GET /api/v1/users/me/credits."""
        response = client.get(
            "/api/v1/users/me/credits",
            headers={"X-User-ID": "test_user_123"},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["balance"] == 100
        assert "packages" in data

    def test_get_credits_no_header(self, client):
        """Test GET /api/v1/users/me/credits without header."""
        response = client.get("/api/v1/users/me/credits")
        assert response.status_code == 422

    def test_get_transactions(self, client, test_user):
        """Test GET /api/v1/users/me/transactions."""
        response = client.get(
            "/api/v1/users/me/transactions",
            headers={"X-User-ID": "test_user_123"},
        )
        assert response.status_code == 200

        data = response.json()
        assert "transactions" in data
        assert "total" in data

    def test_deduct_credits(self, client, test_user):
        """Test POST /api/v1/users/me/credits/deduct."""
        response = client.post(
            "/api/v1/users/me/credits/deduct",
            headers={"X-User-ID": "test_user_123"},
            json={"job_id": "test_job_1", "credits_amount": 5},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["balance"] == 95

    def test_deduct_credits_insufficient(self, client, test_user):
        """Test deducting more credits than available."""
        response = client.post(
            "/api/v1/users/me/credits/deduct",
            headers={"X-User-ID": "test_user_123"},
            json={"job_id": "test_job_2", "credits_amount": 500},
        )
        assert response.status_code == 400

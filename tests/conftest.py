"""
Pytest fixtures and configuration for Stem+MIDI Pro tests.
"""

import os
import sys

import pytest
from sqlalchemy.orm import sessionmaker

os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["SQUARE_ACCESS_TOKEN"] = "test_token"
os.environ["RESEND_API_KEY"] = "test_key"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.database import Base, get_db
from db.models import User
from api.v1.payments import router as payments_router
from api.v1.users import router as users_router

from fastapi import FastAPI

app = FastAPI(title="Stem+MIDI Pro Test")
app.include_router(payments_router, prefix="/api/v1")
app.include_router(users_router, prefix="/api/v1")


@pytest.fixture
def db_engine():
    """Create in-memory SQLite engine for testing."""
    from sqlalchemy import create_engine
    from sqlalchemy.pool import StaticPool

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    return engine


@pytest.fixture
def db_session(db_engine):
    """Create database session for testing."""
    TestingSessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=db_engine
    )
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def test_user(db_session):
    """Create a test user."""
    user = User(
        external_id="test_user_123",
        email="test@example.com",
        credit_balance=100,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def client(db_session, monkeypatch):
    """Create test client with database override."""
    from fastapi.testclient import TestClient

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


@pytest.fixture
def square_mock(monkeypatch):
    """Mock Square client for testing."""

    class MockPaymentLink:
        id = "mock_checkout_id"
        url = "https://squareup.com/checkout/mock"

    class MockCheckout:
        payment_link = MockPaymentLink()

        @property
        def errors(self):
            return None

    class MockClient:
        def create_payment_link(self, **kwargs):
            return MockCheckout()

        def get_order(self, order_id):
            class MockOrder:
                order = {
                    "id": order_id,
                    "line_items": [{"name": "Starter Pack", "quantity": "1"}],
                }

            return MockOrder()

    monkeypatch.setattr("api.v1.payments.square_service.client", MockClient())
    yield MockClient()

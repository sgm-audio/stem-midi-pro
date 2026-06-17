"""
Credit package definitions and credit management service.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict

from sqlalchemy.orm import Session

from db.models import User, CreditTransaction


class CreditPackage(Enum):
    STARTER = "starter"
    POPULAR = "popular"
    BEST_VALUE = "best_value"


@dataclass(frozen=True)
class CreditPackageInfo:
    id: str
    name: str
    price_cents: int
    credits: int
    badge: Optional[str] = None
    description: Optional[str] = None


CREDIT_PACKAGES: Dict[str, CreditPackageInfo] = {
    "starter": CreditPackageInfo(
        id="starter",
        name="Starter Pack",
        price_cents=500,
        credits=10,
        badge=None,
        description="10 credits - $5.00",
    ),
    "popular": CreditPackageInfo(
        id="popular",
        name="Popular Pack",
        price_cents=2000,
        credits=50,
        badge="Most Popular",
        description="50 credits - $20.00",
    ),
    "best_value": CreditPackageInfo(
        id="best_value",
        name="Best Value Pack",
        price_cents=5000,
        credits=150,
        badge="Best Deal",
        description="150 credits - $50.00 (3x value)",
    ),
}


def calculate_job_credits(duration_seconds: float) -> int:
    """
    Calculate credits required for a job based on audio duration.

    Rules:
    - Up to 10 minutes (600s): 1 credit
    - 10-20 minutes (600-1200s): 2 credits
    - 20-30 minutes (1200-1800s): 3 credits
    - Linear beyond that: floor(duration_seconds / 600) + 1
    """
    if duration_seconds <= 600:
        return 1
    elif duration_seconds <= 1200:
        return 2
    elif duration_seconds <= 1800:
        return 3
    else:
        return int(duration_seconds // 600) + 1


class CreditService:
    """Service for managing user credits and transactions."""

    def __init__(self, db: Session):
        self.db = db

    def get_or_create_user(self, external_id: str, email: Optional[str] = None) -> User:
        """Get existing user or create new one."""
        user = self.db.query(User).filter(User.external_id == external_id).first()
        if not user:
            user = User(
                external_id=external_id,
                email=email,
                credit_balance=0,
            )
            self.db.add(user)
            self.db.commit()
            self.db.refresh(user)
        return user

    def get_user_by_external_id(self, external_id: str) -> Optional[User]:
        """Get user by external ID."""
        return self.db.query(User).filter(User.external_id == external_id).first()

    def get_balance(self, user_id: int) -> int:
        """Get current credit balance for user."""
        user = self.db.query(User).filter(User.id == user_id).first()
        if not user:
            return 0
        return user.credit_balance

    def add_credits(
        self,
        user_id: int,
        amount: int,
        payment_id: str,
        description: Optional[str] = None,
    ) -> CreditTransaction:
        """
        Add credits to user account (for purchases).
        Returns the transaction record.
        """
        user = self.db.query(User).filter(User.id == user_id).first()
        if not user:
            raise ValueError(f"User {user_id} not found")

        new_balance = user.credit_balance + amount
        user.credit_balance = new_balance

        transaction = CreditTransaction(
            user_id=user_id,
            amount=amount,
            balance_after=new_balance,
            payment_id=payment_id,
            transaction_type="purchase",
            description=description or f"Purchased {amount} credits",
        )
        self.db.add(transaction)
        self.db.commit()
        self.db.refresh(transaction)
        return transaction

    def spend_credits(
        self,
        user_id: int,
        amount: int,
        job_id: str,
        description: Optional[str] = None,
    ) -> Optional[CreditTransaction]:
        """
        Spend credits from user account (for job processing).
        Returns the transaction record, or None if insufficient balance.
        """
        user = self.db.query(User).filter(User.id == user_id).first()
        if not user:
            raise ValueError(f"User {user_id} not found")

        if user.credit_balance < amount:
            return None

        new_balance = user.credit_balance - amount
        user.credit_balance = new_balance

        transaction = CreditTransaction(
            user_id=user_id,
            amount=-amount,
            balance_after=new_balance,
            job_id=job_id,
            transaction_type="spend",
            description=description or f"Processed job {job_id}",
        )
        self.db.add(transaction)
        self.db.commit()
        self.db.refresh(transaction)
        return transaction

    def get_transactions(
        self,
        user_id: int,
        limit: int = 50,
        offset: int = 0,
    ) -> List[CreditTransaction]:
        """Get transaction history for user."""
        return (
            self.db.query(CreditTransaction)
            .filter(CreditTransaction.user_id == user_id)
            .order_by(CreditTransaction.created_at.desc())
            .limit(limit)
            .offset(offset)
            .all()
        )

    def check_payment_already_processed(self, payment_id: str) -> bool:
        """Check if payment has already been credited (idempotency)."""
        existing = (
            self.db.query(CreditTransaction)
            .filter(CreditTransaction.payment_id == payment_id)
            .first()
        )
        return existing is not None

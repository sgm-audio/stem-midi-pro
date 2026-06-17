"""
SQLAlchemy models for Stem+MIDI Pro
"""

from datetime import datetime
from typing import List

from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, Index
from sqlalchemy.orm import relationship

from db.database import Base


class User(Base):
    """User model for Stem+MIDI Pro."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=True)
    external_id = Column(String(255), unique=True, index=True, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    credit_balance = Column(Integer, nullable=False, default=0)

    transactions: List["CreditTransaction"] = relationship(
        "CreditTransaction", back_populates="user"
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email} credits={self.credit_balance}>"


class CreditTransaction(Base):
    """Credit transaction ledger for PIPEDA-compliant audit trail."""

    __tablename__ = "credit_transactions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    amount = Column(Integer, nullable=False)
    balance_after = Column(Integer, nullable=False)

    payment_id = Column(String(255), unique=True, nullable=True, index=True)
    job_id = Column(String(255), nullable=True, index=True)

    description = Column(String(500), nullable=True)
    transaction_type = Column(String(50), nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    user: "User" = relationship("User", back_populates="transactions")

    __table_args__ = (
        Index("ix_credit_transactions_user_created", "user_id", "created_at"),
        Index("ix_credit_transactions_payment_id", "payment_id"),
    )

    def __repr__(self) -> str:
        return f"<CreditTransaction id={self.id} user_id={self.user_id} amount={self.amount} type={self.transaction_type}>"

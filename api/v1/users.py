"""
User API endpoints for Stem+MIDI Pro.
Handles user credits and transaction history.
"""

from typing import Optional, List, Any

from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.orm import Session
from pydantic import BaseModel

from db.database import get_db
from services.credit_service import CreditService, CREDIT_PACKAGES

router = APIRouter(prefix="/users", tags=["users"])


class UserCreditsResponse(BaseModel):
    balance: int
    packages: dict


class TransactionResponse(BaseModel):
    id: int
    amount: int
    balance_after: int
    description: Optional[str]
    transaction_type: str
    created_at: str

    class Config:
        from_attributes = True


class TransactionsResponse(BaseModel):
    transactions: List[TransactionResponse]
    total: int


def get_user_from_header(db: Session, x_user_id: str) -> Any:
    """Extract user from header - in production this would be from JWT/session."""
    if not x_user_id:
        raise HTTPException(status_code=401, detail="User ID required")

    credit_service = CreditService(db)
    user = credit_service.get_user_by_external_id(x_user_id)

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return user


@router.get("/me/credits")
async def get_user_credits(
    x_user_id: str = Header(..., alias="X-User-ID"),
    db: Session = Depends(get_db),
):
    """
    Get current user's credit balance and available packages.

    Headers:
    - X-User-ID: User's external ID (would be from auth in production)
    """
    user = get_user_from_header(db, x_user_id)

    packages = []
    for pkg_id, pkg in CREDIT_PACKAGES.items():
        packages.append(
            {
                "id": pkg.id,
                "name": pkg.name,
                "price_cents": pkg.price_cents,
                "price_formatted": f"${pkg.price_cents / 100:.2f}",
                "credits": pkg.credits,
                "badge": pkg.badge,
            }
        )

    return {
        "balance": user.credit_balance,
        "packages": packages,
    }


@router.get("/me/transactions")
async def get_user_transactions(
    x_user_id: str = Header(..., alias="X-User-ID"),
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    """
    Get current user's transaction history.

    Headers:
    - X-User-ID: User's external ID (would be from auth in production)

    Query params:
    - limit: Max transactions to return (default 50, max 100)
    - offset: Pagination offset
    """
    user = get_user_from_header(db, x_user_id)

    credit_service = CreditService(db)
    transactions = credit_service.get_transactions(
        user_id=user.id,
        limit=min(limit, 100),
        offset=offset,
    )

    return {
        "transactions": [
            {
                "id": tx.id,
                "amount": tx.amount,
                "balance_after": tx.balance_after,
                "description": tx.description,
                "transaction_type": tx.transaction_type,
                "payment_id": tx.payment_id,
                "job_id": tx.job_id,
                "created_at": tx.created_at.isoformat(),
            }
            for tx in transactions
        ],
        "total": len(transactions),
        "limit": limit,
        "offset": offset,
    }


@router.post("/me/credits/deduct")
async def deduct_credits_for_job(
    x_user_id: str = Header(..., alias="X-User-ID"),
    job_id: str = ...,
    credits_amount: int = ...,
    db: Session = Depends(get_db),
):
    """
    Deduct credits for a processing job.
    This is called internally when a job is submitted.

    Headers:
    - X-User-ID: User's external ID

    Body:
    - job_id: Unique job identifier
    - credits_amount: Number of credits to deduct
    """
    user = get_user_from_header(db, x_user_id)

    credit_service = CreditService(db)

    if user.credit_balance < credits_amount:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient credits. Balance: {user.credit_balance}, Required: {credits_amount}",
        )

    transaction = credit_service.spend_credits(
        user_id=user.id,
        amount=credits_amount,
        job_id=job_id,
        description=f"Job processing: {job_id}",
    )

    if not transaction:
        raise HTTPException(status_code=500, detail="Failed to deduct credits")

    return {
        "success": True,
        "balance": transaction.balance_after,
        "transaction_id": transaction.id,
    }

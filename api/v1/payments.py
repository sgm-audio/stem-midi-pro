"""
Payment API endpoints for Stem+MIDI Pro.
Handles Square checkout creation and webhook processing.
"""

import os
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Header, Request
from pydantic import BaseModel

from db.database import get_db
from services.credit_service import CreditService, CREDIT_PACKAGES
from services.square_service import square_service
from services.email_service import email_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/payments", tags=["payments"])


class CreateOrderRequest(BaseModel):
    package_id: str
    user_id: str
    user_email: Optional[str] = None


class CreateOrderResponse(BaseModel):
    checkout_url: str
    checkout_id: str
    package_id: str
    credits: int
    amount_cents: int


class WebhookPayload(BaseModel):
    """Expected structure of Square webhook payload."""

    merchant_id: Optional[str] = None
    type: Optional[str] = None
    event_id: Optional[str] = None
    created_at: Optional[str] = None
    data: Optional[dict] = None


@router.get("/packages")
async def list_packages():
    """List available credit packages."""
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
                "description": pkg.description,
                "value_per_credit": f"${pkg.price_cents / pkg.credits / 100:.2f}"
                if pkg.credits > 0
                else "N/A",
            }
        )
    return {"packages": packages}


@router.post("/create-order", response_model=CreateOrderResponse)
async def create_order(request: CreateOrderRequest):
    """
    Create a Square checkout session for purchasing credits.

    This returns a checkout URL that the client should redirect to.
    After payment, Square will webhook notify us at /payments/square/webhook.
    """
    if not square_service.is_configured():
        raise HTTPException(
            status_code=503,
            detail="Payment processing not configured",
        )

    package = CREDIT_PACKAGES.get(request.package_id)
    if not package:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown package: {request.package_id}. Available: {list(CREDIT_PACKAGES.keys())}",
        )

    success, data = square_service.create_checkout(
        package_id=request.package_id,
        user_id=request.user_id,
        user_email=request.user_email,
    )

    if not success:
        raise HTTPException(
            status_code=500,
            detail=data.get("error", "Failed to create checkout"),
        )

    return CreateOrderResponse(**data)


@router.post("/square/webhook")
async def square_webhook(
    request: Request,
    x_square_signature: Optional[str] = Header(
        None, alias="x-square-hmacsha256-signature"
    ),
):
    """
    Handle Square webhook events.

    This endpoint is called by Square after payment completion.
    We verify the signature and credit the user's account.

    Required env vars:
    - SQUARE_ACCESS_TOKEN: Square API access token
    - SQUARE_WEBHOOK_SIGNATURE_KEY: For signature verification
    """
    body = await request.body()

    webhook_signature_key = os.getenv("SQUARE_WEBHOOK_SIGNATURE_KEY", "")
    if webhook_signature_key:
        if not square_service.verify_webhook_signature(
            body, x_square_signature or "", webhook_signature_key
        ):
            logger.warning("Invalid Square webhook signature")
            raise HTTPException(status_code=401, detail="Invalid signature")

    import json

    payload = json.loads(body)

    event_type = payload.get("type", "")
    logger.info(f"Square webhook: {event_type}")

    if event_type == "payment.completed":
        await _handle_payment_completed(payload)
    elif event_type == "payment.failed":
        logger.info(
            f"Payment failed: {payload.get('data', {}).get('object', {}).get('payment_id', 'unknown')}"
        )
    else:
        logger.debug(f"Ignoring event type: {event_type}")

    return {"received": True}


async def _handle_payment_completed(payload: dict):
    """Process completed payment and credit user account."""
    try:
        obj = payload.get("data", {}).get("object", {})
        payment_info = obj.get("payment", {})

        payment_id = payment_info.get("id")
        amount_money = payment_info.get("amount_money", {})
        amount_cents = amount_money.get("amount", 0)
        currency = amount_money.get("currency", "USD")
        receiver_email = payment_info.get("receipt_email")

        if not payment_id:
            logger.error("No payment_id in webhook payload")
            return

        db = next(get_db())
        credit_service = CreditService(db)

        if credit_service.check_payment_already_processed(payment_id):
            logger.info(
                f"Payment {payment_id} already processed - skipping (idempotency)"
            )
            return

        order_id = payment_info.get("order_id")
        if not order_id:
            logger.error(f"No order_id in payment {payment_id}")
            return

        from services.square_service import square_service as sq

        if not sq.is_configured():
            logger.error("Square not configured for order lookup")
            return

        try:
            order_response = sq.client.orders.get_order(order_id)
            if order_response.order:
                line_items = order_response.order.get("line_items", [])
                if line_items:
                    item_name = line_items[0].get("name", "")
                    credits = 0
                    package_id = None

                    for pkg_id, pkg in CREDIT_PACKAGES.items():
                        if pkg.name in item_name or pkg_id in item_name.lower():
                            credits = pkg.credits
                            package_id = pkg_id
                            break

                    if credits > 0:
                        user_external_id = payment_info.get("reference_id", "")
                        if not user_external_id:
                            logger.error(
                                f"No reference_id (user_id) in payment {payment_id}"
                            )
                            return

                        user = credit_service.get_or_create_user(
                            external_id=user_external_id,
                            email=receiver_email,
                        )

                        credit_service.add_credits(
                            user_id=user.id,
                            amount=credits,
                            payment_id=payment_id,
                            description=f"Purchased {package_id} package",
                        )

                        logger.info(
                            f"Credited {credits} to user {user.id} for payment {payment_id}"
                        )

                        if receiver_email:
                            email_service.send_credit_purchase_receipt(
                                to_email=receiver_email,
                                credits_added=credits,
                                amount_paid=f"${amount_cents / 100:.2f} {currency}",
                                package_name=item_name,
                                payment_id=payment_id,
                            )
                    else:
                        logger.error(
                            f"Could not determine credits for order {order_id}"
                        )
            else:
                logger.error(f"Order not found: {order_id}")
        except Exception as e:
            logger.error(f"Error processing payment {payment_id}: {e}")
            raise

    except Exception as e:
        logger.error(f"Error handling payment completed: {e}")
        raise

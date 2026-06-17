"""
Square SDK integration for payment processing.
"""

import os
import logging
from typing import Optional, Dict, Any, Tuple

from square.client import Client
from square.exceptions import ApiError

from services.credit_service import CREDIT_PACKAGES, CreditPackageInfo

logger = logging.getLogger(__name__)


class SquareService:
    """Service for interacting with Square Payments API."""

    def __init__(self):
        self.access_token = os.getenv("SQUARE_ACCESS_TOKEN", "")
        self.environment = os.getenv("SQUARE_ENVIRONMENT", "sandbox")

        if not self.access_token:
            logger.warning("SQUARE_ACCESS_TOKEN not set - Square payments disabled")

        self._client: Optional[Client] = None

    @property
    def client(self) -> Optional[Client]:
        """Lazy initialization of Square client."""
        if self._client is None and self.access_token:
            self._client = Client(
                access_token=self.access_token,
                environment=self.environment,
            )
        return self._client

    def is_configured(self) -> bool:
        """Check if Square is properly configured."""
        return bool(self.access_token and self.client)

    def get_package_info(self, package_id: str) -> Optional[CreditPackageInfo]:
        """Get credit package info by ID."""
        return CREDIT_PACKAGES.get(package_id)

    def create_checkout(
        self,
        package_id: str,
        user_id: int,
        user_email: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Create a Square checkout for a credit package.

        Returns:
            Tuple of (success, data)
            - On success: (True, {"checkout_url": str, "checkout_id": str})
            - On failure: (False, {"error": str})
        """
        if not self.is_configured():
            return False, {"error": "Square not configured"}

        package = self.get_package_info(package_id)
        if not package:
            return False, {"error": f"Unknown package: {package_id}"}

        if not idempotency_key:
            import uuid

            idempotency_key = str(uuid.uuid4())

        try:
            item_list = [
                {
                    "name": package.name,
                    "description": package.description,
                    "quantity": "1",
                    "base_price_money": {
                        "amount": package.price_cents,
                        "currency": "USD",
                    },
                }
            ]

            response = self.client.checkout.create_payment_link(
                idempotency_key=idempotency_key,
                order=[
                    {
                        "items": item_list,
                    }
                ],
            )

            if response.errors:
                error_messages = [e.get("detail", str(e)) for e in response.errors]
                logger.error(f"Square checkout errors: {error_messages}")
                return False, {"error": "; ".join(error_messages)}

            checkout = response.payment_link
            checkout_url = checkout.url
            checkout_id = checkout.id

            return True, {
                "checkout_url": checkout_url,
                "checkout_id": checkout_id,
                "package_id": package_id,
                "credits": package.credits,
                "amount_cents": package.price_cents,
            }

        except ApiError as e:
            logger.error(f"Square API error: {e}")
            return False, {"error": f"Square API error: {str(e)}"}
        except Exception as e:
            logger.error(f"Unexpected error creating checkout: {e}")
            return False, {"error": str(e)}

    def verify_webhook_signature(
        self,
        body: bytes,
        signature_header: str,
        webhook_signature_key: Optional[str] = None,
    ) -> bool:
        """
        Verify that a webhook request came from Square.
        In production, set SQUARE_WEBHOOK_SIGNATURE_KEY.
        """
        if not webhook_signature_key:
            webhook_signature_key = os.getenv("SQUARE_WEBHOOK_SIGNATURE_KEY", "")

        if not webhook_signature_key:
            logger.warning(
                "SQUARE_WEBHOOK_SIGNATURE_KEY not set - skipping verification"
            )
            return True

        import hmac
        import hashlib

        expected_signature = hmac.new(
            webhook_signature_key.encode("utf-8"),
            body,
            hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(expected_signature, signature_header)


square_service = SquareService()

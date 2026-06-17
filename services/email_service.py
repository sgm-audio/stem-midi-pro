"""
Email service using Resend for transactional emails.
"""

import os
import logging

try:
    import resend

    RESEND_AVAILABLE = True
except ImportError:
    RESEND_AVAILABLE = False
    logging.warning("resend package not installed - email disabled")

logger = logging.getLogger(__name__)


class EmailService:
    """Service for sending emails via Resend."""

    def __init__(self):
        self.api_key = os.getenv("RESEND_API_KEY", "")
        self.from_email = os.getenv(
            "RESEND_FROM_EMAIL", "Stem+MIDI Pro <noreply@stem-midi.pro>"
        )
        self.enabled = RESEND_AVAILABLE and bool(self.api_key)

        if not self.enabled:
            if not RESEND_AVAILABLE:
                logger.warning("Resend package not installed - emails disabled")
            elif not self.api_key:
                logger.warning("RESEND_API_KEY not set - emails disabled")

    def send_credit_purchase_receipt(
        self,
        to_email: str,
        credits_added: int,
        amount_paid: str,
        package_name: str,
        payment_id: str,
    ) -> bool:
        """
        Send receipt email for credit purchase.

        Args:
            to_email: Recipient email address
            credits_added: Number of credits purchased
            amount_paid: Formatted amount (e.g., "$5.00")
            package_name: Name of the package purchased
            payment_id: Square payment ID for reference

        Returns:
            True if email sent successfully, False otherwise
        """
        if not self.enabled:
            logger.info(f"Email disabled - would send receipt to {to_email}")
            return True

        subject = f"Your Stem+MIDI Pro Receipt - {credits_added} Credits Added"

        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
                .credits {{ font-size: 48px; font-weight: bold; color: #667eea; text-align: center; margin: 20px 0; }}
                .details {{ background: white; padding: 20px; border-radius: 8px; margin: 20px 0; }}
                .details div {{ display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #eee; }}
                .details div:last-child {{ border-bottom: none; }}
                .footer {{ text-align: center; color: #888; font-size: 12px; margin-top: 30px; }}
                .badge {{ background: #667eea; color: white; padding: 4px 12px; border-radius: 20px; font-size: 12px; display: inline-block; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Stem+MIDI Pro</h1>
                <p>Your receipt</p>
            </div>
            <div class="content">
                <div class="credits">{credits_added} Credits</div>
                <div class="details">
                    <div><span>Package</span><span>{package_name}</span></div>
                    <div><span>Amount Paid</span><span>{amount_paid}</span></div>
                    <div><span>Credits Added</span><span>{credits_added}</span></div>
                    <div><span>Payment ID</span><span style="font-size: 12px;">{payment_id}</span></div>
                </div>
                <p style="text-align: center; color: #666;">
                    Your credits are now available in your account.<br>
                    Credits never expire - use them whenever you're ready!
                </p>
            </div>
            <div class="footer">
                <p>Stem+MIDI Pro - Studio-grade stem separation + MIDI transcription</p>
                <p>This email was sent because you made a purchase.</p>
            </div>
        </body>
        </html>
        """

        text_body = f"""
Stem+MIDI Pro Receipt
=====================

Package: {package_name}
Amount Paid: {amount_paid}
Credits Added: {credits_added}
Payment ID: {payment_id}

Your credits are now available in your account.
Credits never expire - use them whenever you're ready!

---
Stem+MIDI Pro - Studio-grade stem separation + MIDI transcription
        """

        return self._send(to_email, subject, html_body, text_body)

    def _send(
        self,
        to_email: str,
        subject: str,
        html_body: str,
        text_body: str,
    ) -> bool:
        """Internal method to send email via Resend API."""
        if not self.enabled:
            logger.info(f"Would send email to {to_email}: {subject}")
            return True

        try:
            resend.Emails.send(
                {
                    "from": self.from_email,
                    "to": to_email,
                    "subject": subject,
                    "html": html_body,
                    "text": text_body,
                }
            )
            logger.info(f"Receipt email sent to {to_email}")
            return True
        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {e}")
            return False


email_service = EmailService()

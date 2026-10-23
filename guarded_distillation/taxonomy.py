"""Label taxonomy for the banking support triage pipeline.

Labels are the 77 Banking77 intents plus ``out_of_scope``. Routing team and urgency are
T0 skills: fixed lookup tables from the intent, so they are exact given the intent.
"""
from __future__ import annotations

OOS = "out_of_scope"

# The "product launch" drift event: these intents are absent from traffic until the launch
# phase, then arrive in a burst. They sit close together in meaning (virtual cards and
# mobile wallets), so the stress they cause is local, not global.
LAUNCH_INTENTS: tuple[str, ...] = (
    "getting_virtual_card",
    "get_disposable_virtual_card",
    "virtual_card_not_working",
    "disposable_card_limits",
    "apple_pay_or_google_pay",
)

ROUTES: dict[str, tuple[str, ...]] = {
    "cards": (
        "activate_my_card", "card_arrival", "card_delivery_estimate", "card_linking",
        "card_not_working", "card_about_to_expire", "card_acceptance", "card_swallowed",
        "change_pin", "pin_blocked", "get_physical_card", "order_physical_card",
        "getting_spare_card", "contactless_not_working", "visa_or_mastercard",
        "supported_cards_and_currencies", "getting_virtual_card", "get_disposable_virtual_card",
        "virtual_card_not_working", "disposable_card_limits", "apple_pay_or_google_pay",
    ),
    "fraud_security": (
        "compromised_card", "lost_or_stolen_card", "lost_or_stolen_phone",
        "card_payment_not_recognised", "cash_withdrawal_not_recognised",
        "direct_debit_payment_not_recognised",
    ),
    "payments_disputes": (
        "card_payment_fee_charged", "card_payment_wrong_exchange_rate", "declined_card_payment",
        "pending_card_payment", "reverted_card_payment?", "request_refund", "Refund_not_showing_up",
        "extra_charge_on_statement", "transaction_charged_twice", "wrong_amount_of_cash_received",
        "wrong_exchange_rate_for_cash_withdrawal", "cash_withdrawal_charge",
        "declined_cash_withdrawal", "pending_cash_withdrawal", "atm_support",
    ),
    "transfers": (
        "cancel_transfer", "declined_transfer", "failed_transfer", "pending_transfer",
        "transfer_fee_charged", "transfer_into_account", "transfer_not_received_by_recipient",
        "transfer_timing", "receiving_money", "beneficiary_not_allowed",
        "balance_not_updated_after_bank_transfer",
    ),
    "top_ups": (
        "automatic_top_up", "pending_top_up", "top_up_by_bank_transfer_charge",
        "top_up_by_card_charge", "top_up_by_cash_or_cheque", "top_up_failed", "top_up_limits",
        "top_up_reverted", "topping_up_by_card", "verify_top_up",
        "balance_not_updated_after_cheque_or_cash_deposit",
    ),
    "fx": ("exchange_charge", "exchange_rate", "exchange_via_app", "fiat_currency_support"),
    "account_kyc": (
        "age_limit", "edit_personal_details", "passcode_forgotten", "terminate_account",
        "unable_to_verify_identity", "verify_my_identity", "verify_source_of_funds",
        "why_verify_identity", "country_support",
    ),
    "general_desk": (OOS,),
}

HIGH_URGENCY: frozenset[str] = frozenset({
    "compromised_card", "lost_or_stolen_card", "lost_or_stolen_phone",
    "card_payment_not_recognised", "cash_withdrawal_not_recognised",
    "direct_debit_payment_not_recognised", "card_swallowed", "transaction_charged_twice",
    "wrong_amount_of_cash_received",
})
LOW_URGENCY: frozenset[str] = frozenset({
    "exchange_rate", "age_limit", "country_support", "fiat_currency_support",
    "supported_cards_and_currencies", "visa_or_mastercard", "card_delivery_estimate",
    "top_up_limits", "disposable_card_limits", "transfer_timing", "why_verify_identity",
    "apple_pay_or_google_pay", "getting_virtual_card", "get_disposable_virtual_card",
    "exchange_via_app", "getting_spare_card", "order_physical_card", "get_physical_card",
    OOS,
})

ROUTE_OF: dict[str, str] = {lab: team for team, labs in ROUTES.items() for lab in labs}


def route(label: str) -> str:
    """T0 skill: routing team for an intent."""
    return ROUTE_OF.get(label, "general_desk")


def urgency(label: str) -> str:
    """T0 skill: urgency for an intent."""
    if label in HIGH_URGENCY:
        return "high"
    if label in LOW_URGENCY:
        return "low"
    return "normal"


def check_complete(intents: list[str]) -> None:
    """Every Banking77 intent must have a route. Raises if the tables drift from the data."""
    missing = sorted(set(intents) - set(ROUTE_OF))
    extra = sorted(set(ROUTE_OF) - set(intents) - {OOS})
    if missing or extra:
        raise ValueError(f"route table mismatch: missing={missing} extra={extra}")

# UK Open Banking — PSD2, Data Rights, and How This Project Relates

**Topics:** uk_open_banking, financial_literacy

Open Banking is a regulatory framework that allows consumers to securely share their financial data with authorised third-party providers (TPPs). In the UK it is governed by the Competition and Markets Authority (CMA) and the Financial Conduct Authority (FCA), implementing the EU's Payment Services Directive 2 (PSD2).

## What PSD2 Mandates

The Payment Services Directive 2 (PSD2) is the EU regulation that underpins Open Banking. The UK retained equivalent obligations post-Brexit.

PSD2 requires banks (Account Servicing Payment Service Providers, or ASPSPs) to:

1. **Expose standardised APIs** — banks must provide secure, machine-readable access to account data through open APIs that third parties can use with customer consent.
2. **Enable third-party access** — FCA-authorised third parties can request read access to a customer's account data.
3. **Support account information and payment initiation** — two categories of service: Account Information Service Providers (AISPs) and Payment Initiation Service Providers (PISPs).
4. **Apply Strong Customer Authentication (SCA)** — multi-factor authentication (e.g. biometric + device possession) required for online payments and account access.

## What Data Banks Must Share

Under Open Banking, banks must allow authorised providers to access the following data types (with customer consent):

### Account Information
- Account name, sort code, account number.
- Account type (current, savings, credit card).
- Currency.

### Balances
- Available balance.
- Current balance.
- Pending transactions.

### Transaction History
- Transaction date, amount, merchant name, merchant category.
- Payment references and descriptions.
- Typically 90 days of history (extendable with re-consent for some providers).

### Scheduled Payments and Standing Orders
- Future-dated payments.
- Direct debit mandates.

## Consumer Rights

As a consumer using Open Banking services, you have the following rights:

### Right to Consent
- You must explicitly consent before any third party can access your account data.
- Consent must specify the provider, the data accessed, and the duration.
- You cannot be penalised by your bank for giving or withholding consent to TPPs.

### Right to Revoke Access at Any Time
- You can revoke a third party's access instantly — either through the TPP's app or directly through your bank's settings.
- Revocation must take effect immediately.
- The TPP must stop accessing your data and must not store it beyond their stated purpose.

### Right to Data Portability
- Your transaction history is your data.
- You can request it in machine-readable format under GDPR's right to data portability.

### Right to Redress
- If a TPP misuses your data, you can complain to the FCA or the Financial Ombudsman Service (FOS).
- Banks that wrongly block Open Banking access can be reported to the CMA or FCA.

## Security Model

Open Banking does not share your bank login credentials with third parties. Instead:

1. You are redirected to your bank's own login page to authenticate.
2. The bank issues a short-lived access token to the TPP.
3. The TPP uses this token to read data via the bank's API — never seeing your password.
4. Tokens expire and must be renewed through re-consent.

This model (OAuth 2.0 / FAPI) is significantly more secure than screen scraping (where apps historically stored your username and password).

## How This Project Relates to Open Banking

This AI-powered personal finance assistant is designed around Open Banking principles:

- **Transaction data is fetched via an API** (the mock banking API at `:8001`) rather than requiring users to upload CSV files or enter data manually — mirroring how a real Open Banking integration would work.
- **The data schema mirrors Open Banking API responses:** transaction objects include `id`, `amount`, `merchant`, `category`, `timestamp`, and `description` — the same fields exposed by UK banks' Open Banking APIs (e.g. HSBC, Lloyds, Barclays, Monzo).
- **Per-user data isolation:** All queries are scoped to `user_id` — reflecting the consent-per-user model of real Open Banking, where each customer's data is siloed.
- **Financial health scoring and anomaly detection** are the exact use-cases that authorised Account Information Service Providers build — categorisation, spending insights, and fraud flagging.

In a production implementation, the mock API would be replaced with a real Open Banking aggregator (e.g. TrueLayer, Plaid UK, Moneyhub) to retrieve consented live bank data. The agent architecture would remain unchanged.

## Key UK Open Banking Institutions

| Organisation | Role |
|---|---|
| Open Banking Limited (OBL) | Operates the Open Banking Implementation Entity (OBIE); maintains the API standards |
| Financial Conduct Authority (FCA) | Authorises and supervises AISPs and PISPs |
| Competition and Markets Authority (CMA) | Mandated the original nine banks to implement Open Banking |
| Financial Ombudsman Service (FOS) | Handles consumer complaints about financial data misuse |

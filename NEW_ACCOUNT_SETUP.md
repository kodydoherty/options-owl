# OptionsOwl — Account Setup

To get your bot running, I need two sets of credentials from you. Follow the steps below and email the results to **kody@L2investing.com**.

---

## 1. Webull OpenAPI (for trading)

### Prerequisites
- A Webull account with a **CASH** account (no margin)
- **Options trading enabled** on the account (apply through the Webull mobile app → Settings if not already approved)

### Steps
1. Go to **https://developer.webull.com**
2. Log in with your Webull account credentials
3. Create a new **App** (name it whatever, e.g. "OptionsOwl")
4. Once approved, copy your **App Key** and **App Secret**

That's it — just two values. The bot auto-detects everything else at startup.

---

## 2. Polygon.io (for real-time market data)

### Steps
1. Go to **https://polygon.io** and create an account
2. Subscribe to the **Options plan ($199/mo)** — this covers both stock and option quotes. Without it the bot can't price entries or exits in real time.
3. Copy your **API Key** from the Polygon dashboard

---

## Send to Kody

Email the following three values to **kody@L2investing.com**:

```
Webull App Key:    <paste here>
Webull App Secret: <paste here>
Polygon API Key:   <paste here>
```

I'll handle the rest of the setup from there.

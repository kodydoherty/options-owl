"""Black-Scholes Greeks calculations in pure Python.

Feature flag: ENABLE_GREEKS
"""

import math
from datetime import date

import yfinance as yf


def norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Compute d1 in Black-Scholes formula."""
    return (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))


def _d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Compute d2 in Black-Scholes formula."""
    return _d1(S, K, T, r, sigma) - sigma * math.sqrt(T)


def _bs_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
    """Black-Scholes option price."""
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    if option_type.lower() == "call":
        return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)


def calc_delta(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str
) -> float:
    """Calculate option delta.

    Args:
        S: Current underlying price.
        K: Strike price.
        T: Time to expiration in years.
        r: Risk-free rate (annualized, e.g. 0.05 for 5%).
        sigma: Implied volatility (annualized, e.g. 0.30 for 30%).
        option_type: "call" or "put".

    Returns:
        Delta value.
    """
    if T <= 0:
        return 0.0
    d1 = _d1(S, K, T, r, sigma)
    if option_type.lower() == "call":
        return norm_cdf(d1)
    else:
        return norm_cdf(d1) - 1.0


def calc_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate option gamma (same for calls and puts).

    Args:
        S: Current underlying price.
        K: Strike price.
        T: Time to expiration in years.
        r: Risk-free rate.
        sigma: Implied volatility.

    Returns:
        Gamma value.
    """
    if T <= 0:
        return 0.0
    d1 = _d1(S, K, T, r, sigma)
    return norm_pdf(d1) / (S * sigma * math.sqrt(T))


def calc_theta(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str
) -> float:
    """Calculate option theta (per calendar day).

    Args:
        S: Current underlying price.
        K: Strike price.
        T: Time to expiration in years.
        r: Risk-free rate.
        sigma: Implied volatility.
        option_type: "call" or "put".

    Returns:
        Theta value (per day, typically negative).
    """
    if T <= 0:
        return 0.0
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    common = -(S * norm_pdf(d1) * sigma) / (2.0 * math.sqrt(T))
    if option_type.lower() == "call":
        theta_annual = common - r * K * math.exp(-r * T) * norm_cdf(d2)
    else:
        theta_annual = common + r * K * math.exp(-r * T) * norm_cdf(-d2)
    return theta_annual / 365.0


def calc_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate option vega (per 1% move in IV).

    Args:
        S: Current underlying price.
        K: Strike price.
        T: Time to expiration in years.
        r: Risk-free rate.
        sigma: Implied volatility.

    Returns:
        Vega value (per 1 percentage point change in IV).
    """
    if T <= 0:
        return 0.0
    d1 = _d1(S, K, T, r, sigma)
    return S * norm_pdf(d1) * math.sqrt(T) / 100.0


def calc_charm(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str
) -> float:
    """Calculate option charm (delta decay, dDelta/dt) per calendar day.

    Black-Scholes charm with zero dividend yield (q = 0), per Hull /
    standard references (e.g. Wikipedia "Greeks (finance)"):

        charm_call = q*e^{-qT}*N(d1)
                     - e^{-qT}*phi(d1) * (2(r-q)T - d2*sigma*sqrt(T)) / (2*T*sigma*sqrt(T))
        charm_put  = -q*e^{-qT}*N(-d1)
                     - e^{-qT}*phi(d1) * (2(r-q)T - d2*sigma*sqrt(T)) / (2*T*sigma*sqrt(T))

    With q = 0 (our convention everywhere in this module) both reduce to:

        charm = -phi(d1) * (2rT - d2*sigma*sqrt(T)) / (2*T*sigma*sqrt(T))

    so call and put charm are identical. charm = dDelta/dt where t is calendar
    time moving forward (i.e. -dDelta/dT). The annualized value is divided by
    365 to express per-calendar-day delta decay, matching calc_theta's
    per-day convention.

    Args:
        S: Current underlying price.
        K: Strike price.
        T: Time to expiration in years.
        r: Risk-free rate.
        sigma: Implied volatility.
        option_type: "call" or "put" (identical under q=0; kept for API
            symmetry with calc_delta/calc_theta).

    Returns:
        Charm value (delta change per calendar day).
    """
    if T <= 0:
        return 0.0
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    sqrt_t = math.sqrt(T)
    # q = 0: call charm == put charm
    charm_annual = -norm_pdf(d1) * (2.0 * r * T - d2 * sigma * sqrt_t) / (
        2.0 * T * sigma * sqrt_t
    )
    return charm_annual / 365.0


def calc_vanna(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate option vanna (dDelta/dVol) per 1% move in IV.

    Black-Scholes vanna (same for calls and puts — it is the cross partial
    d2V/dS dsigma, identical for both by put-call parity). Standard formula
    (q = 0), per Hull / Wikipedia "Greeks (finance)":

        vanna = -e^{-qT} * phi(d1) * d2 / sigma      (q = 0 here)

    Divided by 100 to express the delta change per 1 percentage point change
    in IV, matching calc_vega's per-1% convention.

    Args:
        S: Current underlying price.
        K: Strike price.
        T: Time to expiration in years.
        r: Risk-free rate.
        sigma: Implied volatility.

    Returns:
        Vanna value (delta change per 1 percentage point change in IV).
    """
    if T <= 0:
        return 0.0
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    return -norm_pdf(d1) * d2 / sigma / 100.0


def calc_iv_from_premium(
    premium: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float | None:
    """Estimate implied volatility from an option premium using bisection.

    Args:
        premium: Observed market premium.
        S: Current underlying price.
        K: Strike price.
        T: Time to expiration in years.
        r: Risk-free rate.
        option_type: "call" or "put".
        tol: Convergence tolerance.
        max_iter: Maximum bisection iterations.

    Returns:
        Implied volatility (annualized) or None if it cannot converge.
    """
    if T <= 0 or premium <= 0:
        return None

    lo, hi = 0.001, 5.0

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        price = _bs_price(S, K, T, r, mid, option_type)
        if abs(price - premium) < tol:
            return mid
        if price < premium:
            lo = mid
        else:
            hi = mid

    # Return best estimate even if not fully converged
    return (lo + hi) / 2.0


def get_greeks_for_position(
    ticker: str,
    strike: float,
    expiry_date: str,
    option_type: str,
    premium: float,
    risk_free_rate: float = 0.05,
) -> dict:
    """Fetch current price via yfinance and return all Greeks for a position.

    Args:
        ticker: Stock ticker symbol.
        strike: Option strike price.
        expiry_date: Expiration date string (YYYY-MM-DD).
        option_type: "call" or "put".
        premium: Current option premium.
        risk_free_rate: Risk-free rate (default 5%).

    Returns:
        Dictionary with delta, gamma, theta, vega, iv, and underlying price.
    """
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1d")
    if hist.empty:
        return {"error": f"Could not fetch price for {ticker}"}

    S = float(hist["Close"].iloc[-1])
    K = strike

    # Calculate time to expiration in years
    expiry = date.fromisoformat(expiry_date)
    today = date.today()
    days_to_expiry = (expiry - today).days
    if days_to_expiry <= 0:
        return {"error": "Option has already expired"}
    T = days_to_expiry / 365.0

    r = risk_free_rate

    # Estimate IV from market premium
    iv = calc_iv_from_premium(premium, S, K, T, r, option_type)
    if iv is None:
        return {"error": "Could not compute implied volatility"}

    return {
        "underlying_price": S,
        "strike": K,
        "expiry": expiry_date,
        "days_to_expiry": days_to_expiry,
        "option_type": option_type,
        "iv": round(iv, 4),
        "delta": round(calc_delta(S, K, T, r, iv, option_type), 4),
        "gamma": round(calc_gamma(S, K, T, r, iv), 4),
        "theta": round(calc_theta(S, K, T, r, iv, option_type), 4),
        "vega": round(calc_vega(S, K, T, r, iv), 4),
        "charm": round(calc_charm(S, K, T, r, iv, option_type), 6),
        "vanna": round(calc_vanna(S, K, T, r, iv), 6),
    }

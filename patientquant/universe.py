"""The investable universe.

Every instrument here is a US-listed exchange-traded fund or fund ETF that a
single investor can buy with a few hundred dollars.  There are no individual
stocks, no options, no futures, no leveraged or inverse products: the book's
universe is deliberately restricted to what an ordinary account can hold.

Each entry carries the group it belongs to, which is what the feature layer
uses to build cross-sectionally comparable signals (a momentum score for a
municipal bond ETF is not comparable to a momentum score for an S&P 500 fund
unless the horizons are handled deliberately).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Instrument:
    ticker: str
    name: str
    group: str
    asset_class: str
    benchmark: str


# --- Broad equity ---------------------------------------------------------
BROAD_EQUITY = [
    Instrument("SPY", "SPDR S&P 500 ETF Trust", "us_equity", "equity", "S&P 500"),
    Instrument("VTI", "Vanguard Total Stock Market ETF", "us_equity", "equity", "CRSP US Total Market"),
    Instrument("IVV", "iShares Core S&P 500 ETF", "us_equity", "equity", "S&P 500"),
    Instrument("QQQ", "Invesco QQQ Trust", "us_equity", "equity", "NASDAQ-100"),
    Instrument("IWM", "iShares Russell 2000 ETF", "us_equity", "equity", "Russell 2000"),
    Instrument("RSP", "Invesco S&P 500 Equal Weight ETF", "us_equity", "equity", "S&P 500 Equal Weight"),
    Instrument("MDY", "SPDR S&P MidCap 400 ETF Trust", "us_equity", "equity", "S&P MidCap 400"),
    Instrument("VUG", "Vanguard Growth ETF", "us_equity", "equity", "CRSP US Growth"),
    Instrument("VTV", "Vanguard Value ETF", "us_equity", "equity", "CRSP US Value"),
    Instrument("MGK", "VanEck Morningstar US Growth ETF", "us_equity", "equity", "Morningstar US Growth"),
    Instrument("MGY", "VanEck Morningstar US Value ETF", "us_equity", "equity", "Morningstar US Value"),
]

# --- International equity -------------------------------------------------
INTERNATIONAL = [
    Instrument("EFA", "iShares MSCI EAFE ETF", "intl_equity", "equity", "MSCI EAFE"),
    Instrument("EEM", "iShares MSCI Emerging Markets ETF", "intl_equity", "equity", "MSCI EM"),
    Instrument("VEA", "Vanguard FTSE Developed Markets ETF", "intl_equity", "equity", "FTSE Developed"),
    Instrument("VWO", "Vanguard FTSE Emerging Markets ETF", "intl_equity", "equity", "FTSE Emerging"),
    Instrument("ACWI", "iShares MSCI ACWI ETF", "intl_equity", "equity", "MSCI ACWI"),
    Instrument("SPDW", "SPDR MSCI ACWI IMI ETF", "intl_equity", "equity", "MSCI ACWI IMI"),
]

# --- Sector equity --------------------------------------------------------
SECTORS = [
    Instrument("XLK", "Technology Select Sector SPDR", "sector", "equity", "S&P Technology"),
    Instrument("XLF", "Financial Select Sector SPDR", "sector", "equity", "S&P Financials"),
    Instrument("XLE", "Energy Select Sector SPDR", "sector", "equity", "S&P Energy"),
    Instrument("XLV", "Health Care Select Sector SPDR", "sector", "equity", "S&P Health Care"),
    Instrument("XLI", "Industrial Select Sector SPDR", "sector", "equity", "S&P Industrials"),
    Instrument("XLY", "Consumer Discretionary Select Sector SPDR", "sector", "equity", "S&P Cons. Discretionary"),
    Instrument("XLP", "Consumer Staples Select Sector SPDR", "sector", "equity", "S&P Cons. Staples"),
    Instrument("XLU", "Utilities Select Sector SPDR", "sector", "equity", "S&P Utilities"),
    Instrument("XLB", "Materials Select Sector SPDR", "sector", "equity", "S&P Materials"),
    Instrument("XLRE", "Real Estate Select Sector SPDR", "sector", "equity", "S&P Real Estate"),
    Instrument("XLC", "Communication Services Select Sector SPDR", "sector", "equity", "S&P Comm. Services"),
]

# --- Factor tilts ---------------------------------------------------------
FACTOR = [
    Instrument("MTUM", "iShares MSCI USA Momentum Factor ETF", "factor", "equity", "MSCI USA Momentum"),
    Instrument("VLUE", "iShares MSCI USA Value Factor ETF", "factor", "equity", "MSCI USA Value"),
    Instrument("QUAL", "iShares MSCI USA Quality Factor ETF", "factor", "equity", "MSCI USA Quality"),
    Instrument("USMV", "iShares MSCI USA Min Vol Factor ETF", "factor", "equity", "MSCI USA Min Vol"),
    Instrument("SIZE", "iShares MSCI USA Size Factor ETF", "factor", "equity", "MSCI USA Size"),
    Instrument("SPHD", "Invesco S&P 500 High Dividend ETF", "factor", "equity", "S&P 500 High Dividend"),
    Instrument("USSG", "iShares MSCI USA ESG Select ETF", "factor", "equity", "MSCI USA ESG Select"),
    Instrument("EFV", "iShares MSCI EAFE Value Factor ETF", "factor", "equity", "MSCI EAFE Value"),
    Instrument("AVUV", "Avantis US Value ETF", "factor", "equity", "Avantis US Value"),
    Instrument("SFY", "First Trust S&P 500 Factor ETF", "factor", "equity", "First Trust S&P 500 Factor"),
    Instrument("DEM", "Dynamic Equity ETF", "factor", "equity", "Dynamic Equity"),
]

# --- Fixed income ---------------------------------------------------------
FIXED_INCOME = [
    Instrument("SHY", "iShares 1-3 Year Treasury Bond ETF", "treasury", "bond", "ICE US Treasury 1-3"),
    Instrument("IEF", "iShares 7-10 Year Treasury Bond ETF", "treasury", "bond", "ICE US Treasury 7-10"),
    Instrument("TLT", "iShares 20+ Year Treasury Bond ETF", "treasury", "bond", "ICE US Treasury 20+"),
    Instrument("BIL", "SPDR Bloomberg 1-3 Month T-Bill ETF", "treasury", "bond", "Bloomberg 1-3 Month T-Bill"),
    Instrument("SHV", "iShares Short Treasury Bond ETF", "treasury", "bond", "ICE US Treasury 1-3"),
    Instrument("BND", "Vanguard Total Bond Market ETF", "bond", "bond", "Bloomberg US Aggregate"),
    Instrument("LQD", "iShares iBoxx USD Investment Grade Corporate Bond ETF", "credit", "bond", "iBoxx USD IG"),
    Instrument("HYG", "iShares iBoxx USD High Yield Corporate Bond ETF", "credit", "bond", "iBoxx USD HY"),
    Instrument("JNK", "SPDR Bloomberg High Yield Bond ETF", "credit", "bond", "Bloomberg US High Yield"),
    Instrument("TIP", "iShares TIPS Bond ETF", "tips", "bond", "Bloomberg US TIPS"),
    Instrument("MUB", "iShares National Muni Bond ETF", "muni", "bond", "ICE National Muni"),
    Instrument("VCIT", "Vanguard Intermediate-Term Corporate Bond ETF", "credit", "bond", "Bloomberg US Corporate"),
    Instrument("VGLT", "Vanguard Long-Term Treasury ETF", "treasury", "bond", "Bloomberg US Long Treasury"),
    Instrument("BNDX", "Vanguard Total International Bond ETF", "bond", "bond", "Bloomberg Global Aggregate ex US"),
]

# --- Real assets and cash -------------------------------------------------
REAL_ASSETS = [
    Instrument("GLD", "SPDR Gold Shares", "commodity", "commodity", "Gold spot"),
    Instrument("IAU", "iShares Gold Trust", "commodity", "commodity", "Gold spot"),
    Instrument("VNQ", "Vanguard Real Estate ETF", "real_estate", "real_estate", "MSCI US REIT"),
    Instrument("VNQI", "Vanguard International Real Estate ETF", "real_estate", "real_estate", "FTSE EPRA NAREIT Dev"),
    Instrument("DBC", "Invesco DB Commodity Index Tracking Fund", "commodity", "commodity", "DBIQ Optimum Yield"),
    Instrument("PDBC", "Invesco Optimum Yield Diversified Commodity Strategy", "commodity", "commodity", "DBIQ Optimum Yield"),
    Instrument("DBMF", "Invesco Managed Futures ETF", "managed_futures", "commodity", "DBIQ Managed Futures"),
    Instrument("RINF", "Amplify PureAlpha Real Estate Income ETF", "real_estate", "real_estate", "Real Estate Income"),
]

UNIVERSE: list[Instrument] = (
    BROAD_EQUITY + INTERNATIONAL + SECTORS + FACTOR + FIXED_INCOME + REAL_ASSETS
)

TICKERS: list[str] = [i.ticker for i in UNIVERSE]

BY_TICKER: dict[str, Instrument] = {i.ticker: i for i in UNIVERSE}


def group_of(ticker: str) -> str:
    return BY_TICKER[ticker].group


def asset_class_of(ticker: str) -> str:
    return BY_TICKER[ticker].asset_class


def equity_tickers() -> list[str]:
    return [i.ticker for i in UNIVERSE if i.asset_class == "equity"]


def defensive_tickers() -> list[str]:
    """Instruments an investor can hold instead of cash.

    A short Treasury bill fund and a money-market fund both have a defined
    net asset value and near-zero duration.  They are the instruments the
    system uses when it decides it has no edge: not the broker's sweep
    balance, which pays interest but is not a portfolio decision.
    """
    return ["SHY", "BIL", "SHV"]

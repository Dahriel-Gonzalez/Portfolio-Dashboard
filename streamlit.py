import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf

# Page configuration
st.set_page_config(
    page_title="Portfolio Analysis Dashboard",
    layout="wide"
)

# ---------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# ---------------------------------------------------------------------------

if 'portfolio' not in st.session_state:
    st.session_state.portfolio = []

if 'auto_rebalance' not in st.session_state:
    st.session_state.auto_rebalance = True

if 'total_investment' not in st.session_state:
    st.session_state.total_investment = 10000.0

if 'input_mode' not in st.session_state:
    st.session_state.input_mode = "Dollar Amount"

if '_pending_change' not in st.session_state:
    st.session_state._pending_change = None

# ---------------------------------------------------------------------------
# DOLLAR <-> WEIGHT CONVERSION HELPERS
# ---------------------------------------------------------------------------

def dollars_to_weight(dollars):
    """Convert a dollar amount to a portfolio weight percentage."""
    total = st.session_state.total_investment
    if total <= 0:
        return 0.0
    return (dollars / total) * 100.0


def weight_to_dollars(weight):
    """Convert a portfolio weight percentage to a dollar amount."""
    total = st.session_state.total_investment
    return (weight / 100.0) * total


def get_item_dollars(item):
    """Get the dollar amount for a portfolio item."""
    return weight_to_dollars(item['weight'])


# ---------------------------------------------------------------------------
# CORE WEIGHT MANAGEMENT ENGINE
# ---------------------------------------------------------------------------

def get_locked_total():
    return sum(
        item['weight'] for item in st.session_state.portfolio if item['locked']
    )


def get_unlocked_indices(exclude_idx=None):
    return [
        i for i, item in enumerate(st.session_state.portfolio)
        if not item['locked'] and i != exclude_idx
    ]


def rebalance_weights(changed_idx, new_weight):
    portfolio = st.session_state.portfolio

    if len(portfolio) <= 1:
        portfolio[0]['weight'] = 100.0
        return

    new_weight = max(0.0, min(100.0, new_weight))

    locked_total = sum(
        item['weight'] for i, item in enumerate(portfolio)
        if item['locked'] and i != changed_idx
    )
    budget = 100.0 - locked_total - new_weight

    if budget < 0:
        new_weight = 100.0 - locked_total
        budget = 0.0

    other_unlocked = get_unlocked_indices(exclude_idx=changed_idx)
    other_total = sum(portfolio[i]['weight'] for i in other_unlocked)

    for i, item in enumerate(portfolio):
        if i == changed_idx:
            item['weight'] = round(new_weight, 2)
        elif item['locked']:
            pass
        elif i in other_unlocked:
            if other_total > 0:
                item['weight'] = round(
                    (item['weight'] / other_total) * budget, 2
                )
            else:
                item['weight'] = round(budget / len(other_unlocked), 2)

    current_total = sum(item['weight'] for item in portfolio)
    drift = round(100.0 - current_total, 2)
    if abs(drift) > 0.001:
        candidates = other_unlocked if other_unlocked else [changed_idx]
        target = max(candidates, key=lambda i: portfolio[i]['weight'])
        portfolio[target]['weight'] = round(portfolio[target]['weight'] + drift, 2)


def normalize_weights():
    portfolio = st.session_state.portfolio
    total = sum(item['weight'] for item in portfolio)
    if total > 0:
        for item in portfolio:
            item['weight'] = round((item['weight'] / total) * 100, 2)
    drift = round(100.0 - sum(item['weight'] for item in portfolio), 2)
    if abs(drift) > 0.001 and portfolio:
        portfolio[0]['weight'] = round(portfolio[0]['weight'] + drift, 2)


def equal_split_weights():
    portfolio = st.session_state.portfolio
    n = len(portfolio)
    if n == 0:
        return
    base = round(100.0 / n, 2)
    for item in portfolio:
        item['weight'] = base
    drift = round(100.0 - sum(item['weight'] for item in portfolio), 2)
    if abs(drift) > 0.001:
        portfolio[0]['weight'] = round(portfolio[0]['weight'] + drift, 2)


def equal_split_unlocked():
    portfolio = st.session_state.portfolio
    locked_total = get_locked_total()
    budget = max(0.0, 100.0 - locked_total)

    unlocked = [i for i, item in enumerate(portfolio) if not item['locked']]
    n_unlocked = len(unlocked)

    if n_unlocked == 0:
        return

    base = round(budget / n_unlocked, 2)
    for i in unlocked:
        portfolio[i]['weight'] = base

    drift = round(100.0 - sum(item['weight'] for item in portfolio), 2)
    if abs(drift) > 0.001:
        portfolio[unlocked[0]]['weight'] = round(portfolio[unlocked[0]]['weight'] + drift, 2)


def sync_slider_keys():
    for i, item in enumerate(st.session_state.portfolio):
        key = f"slider_{item['ticker']}_{i}"
        if st.session_state.input_mode == "Dollar Amount":
            st.session_state[key] = round(weight_to_dollars(item['weight']), 2)
        else:
            st.session_state[key] = float(item['weight'])


def on_slider_change(idx):
    item = st.session_state.portfolio[idx]
    key = f"slider_{item['ticker']}_{idx}"
    new_val = st.session_state[key]

    if st.session_state.input_mode == "Dollar Amount":
        new_weight = dollars_to_weight(new_val)
    else:
        new_weight = new_val

    st.session_state._pending_change = (idx, new_weight)


def on_lock_change(idx):
    item = st.session_state.portfolio[idx]
    key = f"lock_{item['ticker']}_{idx}"
    new_locked = st.session_state[key]
    item['locked'] = new_locked

    if new_locked and st.session_state.auto_rebalance:
        equal_split_unlocked()

    sync_slider_keys()


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------

st.sidebar.header("Settings")

analysis_mode = st.sidebar.radio(
    "Analysis Mode",
    ["Single Stock", "Portfolio Analysis"],
    index=0
)

if analysis_mode == "Single Stock":
    ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")
    days = st.sidebar.slider("Days of Historical Data", 30, 365, 180)
    show_ma = st.sidebar.checkbox("Show Moving Average", value=True)

else:
    # ---------------------------------------------------------------------------
    # PORTFOLIO SETTINGS
    # ---------------------------------------------------------------------------
    st.sidebar.subheader("Portfolio Builder")

    # --- Total investment amount ---
    st.session_state.total_investment = st.sidebar.number_input(
        "Total Portfolio Value ($)",
        min_value=100.0,
        max_value=100_000_000.0,
        value=st.session_state.total_investment,
        step=1000.0,
        format="%.2f",
        help="The total dollar amount you want to invest across all stocks."
    )

    # --- Input mode toggle ---
    st.session_state.input_mode = st.sidebar.radio(
        "Input Mode",
        ["Dollar Amount", "Percentage"],
        index=0 if st.session_state.input_mode == "Dollar Amount" else 1,
        help="Choose whether to set allocations in dollars or percentages.",
        horizontal=True
    )

    # --- Add stock ---
    new_ticker = st.sidebar.text_input("Add Stock Ticker", key="new_ticker")

    if st.session_state.portfolio and st.session_state.auto_rebalance:
        default_pct = round(100.0 / (len(st.session_state.portfolio) + 1), 1)
    else:
        default_pct = 10.0

    if st.session_state.input_mode == "Dollar Amount":
        default_dollars = round(weight_to_dollars(default_pct), 2)
        new_amount = st.sidebar.number_input(
            "Initial Amount ($)",
            min_value=0.0,
            max_value=st.session_state.total_investment,
            value=default_dollars,
            step=100.0,
            format="%.2f",
            key="new_amount_dollars"
        )
        new_weight = dollars_to_weight(new_amount)
        st.sidebar.caption(f"{new_weight:.1f}% of portfolio")
    else:
        new_weight = st.sidebar.number_input(
            "Initial Weight (%)",
            min_value=0.0, max_value=100.0,
            value=default_pct, step=5.0,
            key="new_weight_pct"
        )
        st.sidebar.caption(f"${weight_to_dollars(new_weight):,.2f}")

    col_add, col_clear = st.sidebar.columns(2)

    if col_add.button("Add"):
        if new_ticker:
            existing = [item['ticker'] for item in st.session_state.portfolio]
            if new_ticker.upper() not in existing:
                if st.session_state.auto_rebalance and st.session_state.portfolio:
                    locked_total = get_locked_total()
                    available = 100.0 - locked_total
                    clamped_weight = min(new_weight, available)

                    st.session_state.portfolio.append({
                        'ticker': new_ticker.upper(),
                        'weight': clamped_weight,
                        'locked': False
                    })

                    new_idx = len(st.session_state.portfolio) - 1
                    rebalance_weights(new_idx, clamped_weight)

                    if clamped_weight < new_weight:
                        st.sidebar.caption(
                            f"Clamped {new_ticker.upper()} to "
                            f"${weight_to_dollars(clamped_weight):,.2f} "
                            f"({clamped_weight:.1f}%) — "
                            f"locked stocks use {locked_total:.1f}%"
                        )
                else:
                    st.session_state.portfolio.append({
                        'ticker': new_ticker.upper(),
                        'weight': new_weight,
                        'locked': False
                    })

                sync_slider_keys()
                st.rerun()
            else:
                st.sidebar.caption(f"{new_ticker.upper()} already in portfolio")

    if col_clear.button("Clear"):
        st.session_state.portfolio = []
        st.session_state._pending_change = None
        st.rerun()

    # ---------------------------------------------------------------------------
    # WEIGHT EDITOR
    # ---------------------------------------------------------------------------
    if st.session_state.portfolio:
        st.sidebar.divider()
        st.sidebar.subheader("Adjust Allocations")

        st.session_state.auto_rebalance = st.sidebar.toggle(
            "Auto-rebalance",
            value=st.session_state.auto_rebalance,
            help="Changing one unlocked stock automatically adjusts "
                 "the other UNLOCKED stocks so total stays at 100%. "
                 "Locked stocks are never touched."
        )

        st.sidebar.markdown("**Quick Actions:**")

        qa1, qa2 = st.sidebar.columns(2)
        if qa1.button("Equal (unlocked)", help="Split remaining budget equally among unlocked"):
            equal_split_unlocked()
            sync_slider_keys()
            st.rerun()
        if qa2.button("Unlock All", help="Remove all locks"):
            for item in st.session_state.portfolio:
                item['locked'] = False
            sync_slider_keys()
            st.rerun()

        qa3, qa4 = st.sidebar.columns(2)
        if qa3.button("Equal (all)", help="Split 100% equally, ignoring locks"):
            equal_split_weights()
            sync_slider_keys()
            st.rerun()
        if qa4.button("Normalize", help="Rescale all to sum to 100%"):
            normalize_weights()
            sync_slider_keys()
            st.rerun()

        st.sidebar.divider()

        # --- Apply pending change from last run ---
        if st.session_state._pending_change is not None:
            changed_idx, changed_value = st.session_state._pending_change
            st.session_state._pending_change = None

            if st.session_state.auto_rebalance:
                rebalance_weights(changed_idx, changed_value)
            else:
                st.session_state.portfolio[changed_idx]['weight'] = round(changed_value, 2)

            sync_slider_keys()

        # --- Per-stock controls ---
        total_inv = st.session_state.total_investment
        is_dollar_mode = st.session_state.input_mode == "Dollar Amount"

        for i, item in enumerate(st.session_state.portfolio):
            item_dollars = weight_to_dollars(item['weight'])
            lock_label = "[locked]" if item['locked'] else ""

            if is_dollar_mode:
                st.sidebar.markdown(
                    f"**{item['ticker']}** {lock_label} — "
                    f"`${item_dollars:,.2f}` "
                    f"(`{item['weight']:.1f}%`)"
                )
            else:
                st.sidebar.markdown(
                    f"**{item['ticker']}** {lock_label} — "
                    f"`{item['weight']:.1f}%` "
                    f"(`${item_dollars:,.2f}`)"
                )

            ctrl_lock, ctrl_slider, ctrl_rm = st.sidebar.columns([1, 5, 1])

            ctrl_lock.toggle(
                "Lock",
                value=item['locked'],
                key=f"lock_{item['ticker']}_{i}",
                label_visibility="collapsed",
                help=f"Lock {item['ticker']} at {item['weight']:.1f}% (${item_dollars:,.2f})",
                on_change=on_lock_change,
                args=(i,)
            )

            slider_key = f"slider_{item['ticker']}_{i}"

            if is_dollar_mode:
                if slider_key not in st.session_state:
                    st.session_state[slider_key] = round(item_dollars, 2)

                dollar_step = max(10.0, round(total_inv / 200, -1))

                ctrl_slider.slider(
                    label=f"${item['ticker']}",
                    min_value=0.0,
                    max_value=round(total_inv, 2),
                    step=dollar_step,
                    key=slider_key,
                    format="$%,.0f",
                    label_visibility="collapsed",
                    disabled=item['locked'],
                    on_change=on_slider_change,
                    args=(i,)
                )
            else:
                if slider_key not in st.session_state:
                    st.session_state[slider_key] = float(item['weight'])

                ctrl_slider.slider(
                    label=f"w_{item['ticker']}",
                    min_value=0.0,
                    max_value=100.0,
                    step=0.5,
                    key=slider_key,
                    label_visibility="collapsed",
                    disabled=item['locked'],
                    on_change=on_slider_change,
                    args=(i,)
                )

            if ctrl_rm.button("X", key=f"rm_{i}"):
                st.session_state.portfolio.pop(i)
                if st.session_state.auto_rebalance and st.session_state.portfolio:
                    equal_split_unlocked()
                sync_slider_keys()
                st.rerun()

        # ---------------------------------------------------------------------------
        # STATUS BAR
        # ---------------------------------------------------------------------------
        st.sidebar.divider()

        total_weight = sum(item['weight'] for item in st.session_state.portfolio)
        total_dollars_allocated = weight_to_dollars(total_weight)
        n_locked = sum(1 for item in st.session_state.portfolio if item['locked'])
        locked_total = get_locked_total()

        bar_pct = min(total_weight / 100.0, 1.0)

        if is_dollar_mode:
            st.sidebar.progress(
                bar_pct,
                text=f"Allocated: ${total_dollars_allocated:,.2f} / "
                     f"${total_inv:,.2f} ({total_weight:.1f}%)"
            )
        else:
            st.sidebar.progress(bar_pct, text=f"Total: {total_weight:.1f}%")

        if n_locked > 0:
            locked_tickers = ", ".join(
                item['ticker'] for item in st.session_state.portfolio if item['locked']
            )
            locked_dollars = weight_to_dollars(locked_total)
            unlocked_dollars = total_inv - locked_dollars
            st.sidebar.caption(
                f"{n_locked} locked ({locked_tickers}) — "
                f"${locked_dollars:,.2f} ({locked_total:.1f}%) fixed. "
                f"Unlocked stocks share ${unlocked_dollars:,.2f} "
                f"({100 - locked_total:.1f}%)."
            )

        if abs(total_weight - 100.0) < 0.1:
            st.sidebar.caption("Weights sum to 100%")
        elif total_weight > 100.0:
            over = total_weight - 100.0
            st.sidebar.caption(
                f"Over-allocated by ${weight_to_dollars(over):,.2f} ({over:.1f}%)"
            )
        else:
            under = 100.0 - total_weight
            st.sidebar.caption(
                f"Under-allocated by ${weight_to_dollars(under):,.2f} ({under:.1f}%)"
            )

    # ---------------------------------------------------------------------------
    # REMAINING SIDEBAR SETTINGS
    # ---------------------------------------------------------------------------
    days = st.sidebar.slider("Days of Historical Data", 30, 730, 365)

    st.sidebar.subheader("Compare Against")
    benchmarks = st.sidebar.multiselect(
        "Select Benchmarks",
        ["S&P 500", "NASDAQ", "Dow Jones", "Russell 2000"],
        default=["S&P 500"]
    )

    show_ma = st.sidebar.checkbox("Show Moving Average", value=True)


# ---------------------------------------------------------------------------
# DATA FETCHING & CALCULATIONS
# ---------------------------------------------------------------------------

@st.cache_data
def fetch_stock_data(ticker, num_days):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=num_days)
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        if df.empty:
            st.write(f"No data found for ticker '{ticker}'. Please check the symbol.")
            return None
        df = df.reset_index()
        df = df.rename(columns={'Close': 'Price'})
        return df[['Date', 'Price', 'Volume', 'High', 'Low', 'Open']]
    except Exception as e:
        st.write(f"Error fetching data: {str(e)}")
        return None


@st.cache_data
def fetch_benchmark_data(benchmark_name, num_days):
    benchmark_tickers = {
        "S&P 500": "^GSPC",
        "NASDAQ": "^IXIC",
        "Dow Jones": "^DJI",
        "Russell 2000": "^RUT"
    }
    ticker = benchmark_tickers.get(benchmark_name)
    if ticker:
        return fetch_stock_data(ticker, num_days)
    return None


def calculate_portfolio_returns(portfolio_stocks, portfolio_weights):
    if not portfolio_stocks:
        return None

    dates = portfolio_stocks[0]['Date']
    portfolio_returns = pd.Series(0.0, index=range(len(dates)))

    for stock_data, weight in zip(portfolio_stocks, portfolio_weights):
        returns = stock_data['Price'].pct_change().fillna(0)
        portfolio_returns += returns * (weight / 100.0)

    portfolio_price = 100 * (1 + portfolio_returns).cumprod()

    return pd.DataFrame({
        'Date': dates,
        'Price': portfolio_price,
        'Returns': portfolio_returns
    })


def calculate_max_drawdown(df):
    if len(df) < 2:
        return None

    prices = df['Price'].copy()
    cumulative = (1 + prices.pct_change()).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max

    max_dd = drawdown.min()
    max_dd_idx = drawdown.idxmin()
    peak_idx = running_max[:max_dd_idx].idxmax()

    return {
        'max_drawdown': max_dd,
        'drawdown_series': drawdown,
        'peak_date': df.loc[peak_idx, 'Date'],
        'trough_date': df.loc[max_dd_idx, 'Date'],
        'peak_price': df.loc[peak_idx, 'Price'],
        'trough_price': df.loc[max_dd_idx, 'Price']
    }


# ---------------------------------------------------------------------------
# MAIN DATA LOADING
# ---------------------------------------------------------------------------

if analysis_mode == "Single Stock":
    with st.spinner('Loading data...'):
        df = fetch_stock_data(ticker, days)

    if df is None:
        st.stop()

    dd_info = calculate_max_drawdown(df)
    if dd_info is None:
        st.write("Not enough data to calculate drawdown metrics.")
        st.stop()

else:
    if not st.session_state.portfolio:
        st.write("Add stocks to your portfolio using the sidebar to get started.")
        st.stop()

    total_weight = sum(item['weight'] for item in st.session_state.portfolio)
    if abs(total_weight - 100.0) > 0.1:
        st.write(
            "Weights don't sum to 100%. "
            "Enable auto-rebalance or click Normalize."
        )
        st.stop()

    with st.spinner('Loading portfolio data...'):
        portfolio_stocks = []
        portfolio_weights = []

        for item in st.session_state.portfolio:
            stock_data = fetch_stock_data(item['ticker'], days)
            if stock_data is not None:
                portfolio_stocks.append(stock_data)
                portfolio_weights.append(item['weight'])
            else:
                st.write(f"Failed to fetch data for {item['ticker']}")
                st.stop()

        df = calculate_portfolio_returns(portfolio_stocks, portfolio_weights)
        if df is None:
            st.write("Failed to calculate portfolio returns.")
            st.stop()

        benchmark_data = {}
        for benchmark in benchmarks:
            bench_df = fetch_benchmark_data(benchmark, days)
            if bench_df is not None:
                bench_df['Price'] = 100 * (bench_df['Price'] / bench_df['Price'].iloc[0])
                benchmark_data[benchmark] = bench_df

    dd_info = calculate_max_drawdown(df)
    if dd_info is None:
        st.write("Not enough data to calculate drawdown metrics.")
        st.stop()


# ---------------------------------------------------------------------------
# METRICS ROW
# ---------------------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

if analysis_mode == "Single Stock":
    with col1:
        st.metric(
            label="Current Price",
            value=f"${df['Price'].iloc[-1]:.2f}",
            delta=f"{((df['Price'].iloc[-1] / df['Price'].iloc[-2] - 1) * 100):.2f}%"
        )
    with col2:
        st.metric(label=f"{days}-Day High", value=f"${df['Price'].max():.2f}")
    with col3:
        st.metric(label=f"{days}-Day Low", value=f"${df['Price'].min():.2f}")
    with col4:
        st.metric(
            label="Max Drawdown",
            value=f"{dd_info['max_drawdown']*100:.2f}%",
            delta=f"{dd_info['max_drawdown']*100:.2f}%",
            delta_color="inverse"
        )
else:
    total_return = ((df['Price'].iloc[-1] / df['Price'].iloc[0]) - 1) * 100
    total_inv = st.session_state.total_investment
    current_value = total_inv * (df['Price'].iloc[-1] / df['Price'].iloc[0])
    dollar_gain = current_value - total_inv

    with col1:
        st.metric(
            label="Portfolio Value",
            value=f"${current_value:,.2f}",
            delta=f"{((df['Price'].iloc[-1] / df['Price'].iloc[-2] - 1) * 100):.2f}% today"
        )
    with col2:
        st.metric(
            label="Total Return",
            value=f"{total_return:.2f}%",
            delta=f"${dollar_gain:+,.2f}"
        )
    with col3:
        annualized_vol = df['Returns'].std() * np.sqrt(252) * 100
        st.metric(label="Annualized Volatility", value=f"{annualized_vol:.2f}%")
    with col4:
        st.metric(
            label="Max Drawdown",
            value=f"{dd_info['max_drawdown']*100:.2f}%",
            delta=f"${total_inv * dd_info['max_drawdown']:,.2f}",
            delta_color="inverse"
        )

    # ---------------------------------------------------------------------------
    # PIE CHART
    # ---------------------------------------------------------------------------
    st.subheader("Portfolio Allocation")
    portfolio_df = pd.DataFrame(st.session_state.portfolio)

    portfolio_df['dollars'] = portfolio_df['weight'].apply(weight_to_dollars)
    portfolio_df['label'] = portfolio_df.apply(
        lambda row: f"{row['ticker']}{' (locked)' if row['locked'] else ''}",
        axis=1
    )

    fig_pie = px.pie(
        portfolio_df,
        values='weight',
        names='label',
        title='Portfolio Weight Distribution',
        hole=0.4,
        custom_data=['dollars']
    )

    fig_pie.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate=(
            '<b>%{label}</b><br>'
            'Weight: %{value:.1f}%<br>'
            'Amount: $%{customdata[0]:,.2f}'
            '<extra></extra>'
        )
    )

    fig_pie.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=-0.2,
            xanchor="center", x=0.5
        )
    )

    st.plotly_chart(fig_pie, use_container_width=True, key="portfolio_pie_chart")


# ---------------------------------------------------------------------------
# MAIN PRICE CHART
# ---------------------------------------------------------------------------

if show_ma and analysis_mode == "Single Stock":
    df['MA_20'] = df['Price'].rolling(window=20).mean()

if analysis_mode == "Single Stock":
    st.subheader(f"{ticker} Price Chart")
    fig = px.line(df, x='Date', y='Price', title=f'{ticker} Stock Price Over Time')

    if show_ma:
        fig.add_scatter(
            x=df['Date'], y=df['MA_20'], mode='lines', name='20-Day MA',
            line=dict(dash='dash', color='orange')
        )

    fig.update_layout(xaxis_title="Date", yaxis_title="Price ($)", hovermode='x unified')

else:
    st.subheader("Portfolio Performance vs Benchmarks")
    fig = go.Figure()

    scale_factor = total_inv / 100.0

    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Price'] * scale_factor, mode='lines',
        name='Your Portfolio',
        line=dict(color='#1f77b4', width=3)
    ))

    if show_ma:
        df['MA_20'] = df['Price'].rolling(window=20).mean()
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['MA_20'] * scale_factor, mode='lines',
            name='Portfolio 20-Day MA',
            line=dict(color='orange', width=2, dash='dash')
        ))

    colors = ['green', 'red', 'purple', 'brown']
    for i, (benchmark_name, bench_df) in enumerate(benchmark_data.items()):
        fig.add_trace(go.Scatter(
            x=bench_df['Date'], y=bench_df['Price'] * scale_factor, mode='lines',
            name=benchmark_name,
            line=dict(color=colors[i % len(colors)], width=2, dash='dot')
        ))

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        yaxis_tickprefix="$",
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

st.plotly_chart(fig, use_container_width=True, key="main_price_chart")


# ---------------------------------------------------------------------------
# EXPANDABLE RAW DATA
# ---------------------------------------------------------------------------

if analysis_mode == "Single Stock":
    with st.expander("View Raw Data"):
        st.dataframe(df.tail(20), use_container_width=True)
else:
    with st.expander("View Portfolio Composition & Performance"):
        c1, c2 = st.columns(2)

        with c1:
            st.write("**Portfolio Holdings**")
            display_df = pd.DataFrame(st.session_state.portfolio)
            display_df['Status'] = display_df['locked'].apply(
                lambda x: "Locked" if x else "Unlocked"
            )
            display_df['Weight'] = display_df['weight'].apply(lambda x: f"{x:.1f}%")
            display_df['Amount'] = display_df['weight'].apply(
                lambda x: f"${weight_to_dollars(x):,.2f}"
            )
            st.dataframe(
                display_df[['ticker', 'Weight', 'Amount', 'Status']].rename(
                    columns={'ticker': 'Ticker'}
                ),
                use_container_width=True,
                hide_index=True
            )

        with c2:
            st.write("**Performance Comparison**")
            perf_data = []

            portfolio_return = ((df['Price'].iloc[-1] / df['Price'].iloc[0]) - 1) * 100
            portfolio_dollar_return = total_inv * (portfolio_return / 100.0)
            perf_data.append({
                'Asset': 'Your Portfolio',
                'Total Return': f"{portfolio_return:.2f}%",
                'Dollar P&L': f"${portfolio_dollar_return:+,.2f}",
                'Max Drawdown': f"{dd_info['max_drawdown']*100:.2f}%"
            })

            for benchmark_name, bench_df in benchmark_data.items():
                bench_return = ((bench_df['Price'].iloc[-1] / bench_df['Price'].iloc[0]) - 1) * 100
                bench_dd_info = calculate_max_drawdown(bench_df)
                bench_dollar = total_inv * (bench_return / 100.0)
                perf_data.append({
                    'Asset': benchmark_name,
                    'Total Return': f"{bench_return:.2f}%",
                    'Dollar P&L': f"${bench_dollar:+,.2f}",
                    'Max Drawdown': f"{bench_dd_info['max_drawdown']*100:.2f}%" if bench_dd_info else "N/A"
                })

            st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------

if analysis_mode == "Single Stock":
    tab1, tab2, tab3, tab4 = st.tabs([
        "Statistics", "Returns Distribution", "Max Drawdown", "About"
    ])
else:
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Statistics", "Returns Distribution", "Max Drawdown",
        "Correlation Matrix", "About"
    ])

# --- Tab 1 ---
with tab1:
    st.subheader("Statistical Summary")

    if analysis_mode == "Single Stock":
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Price Statistics**")
            st.write(df['Price'].describe())
        with c2:
            st.write("**Returns Distribution**")
            returns = df['Price'].pct_change().dropna()
            fig_hist = px.histogram(returns, x=returns.values, nbins=50, title="Daily Returns Distribution")
            st.plotly_chart(fig_hist, use_container_width=True, key="ss_ret_hist")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Portfolio Metrics**")
            returns = df['Returns'].dropna()
            total_ret = ((df['Price'].iloc[-1] / df['Price'].iloc[0]) - 1) * 100
            ann_ret = (((df['Price'].iloc[-1] / df['Price'].iloc[0]) ** (252 / len(df))) - 1) * 100
            ann_vol = returns.std() * np.sqrt(252) * 100
            sharpe = returns.mean() / returns.std() * np.sqrt(252)

            metrics_data = {
                'Metric': [
                    'Initial Investment', 'Current Value', 'Dollar P&L',
                    'Total Return', 'Annualized Return', 'Annualized Volatility',
                    'Sharpe Ratio (Rf=0)', 'Best Day', 'Worst Day'
                ],
                'Value': [
                    f"${total_inv:,.2f}",
                    f"${total_inv * (1 + total_ret / 100):,.2f}",
                    f"${total_inv * total_ret / 100:+,.2f}",
                    f"{total_ret:.2f}%",
                    f"{ann_ret:.2f}%",
                    f"{ann_vol:.2f}%",
                    f"{sharpe:.2f}",
                    f"{returns.max() * 100:.2f}% (${total_inv * returns.max():+,.2f})",
                    f"{returns.min() * 100:.2f}% (${total_inv * returns.min():+,.2f})"
                ]
            }
            st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)
        with c2:
            st.write("**Returns Distribution**")
            fig_hist = px.histogram(
                returns * 100, x=returns.values * 100, nbins=50,
                title="Daily Returns Distribution (%)", labels={'x': 'Daily Return (%)'}
            )
            st.plotly_chart(fig_hist, use_container_width=True, key="port_ret_hist")

# --- Tab 2 ---
with tab2:
    if analysis_mode == "Single Stock":
        st.subheader("Trading Volume")
        fig_vol = px.bar(df.tail(60), x='Date', y='Volume', title='Trading Volume (Last 60 Days)')
        st.plotly_chart(fig_vol, use_container_width=True, key="ss_volume")
    else:
        st.subheader("Rolling Returns Analysis")
        rolling_30 = df['Price'].pct_change(30).dropna() * 100
        rolling_90 = df['Price'].pct_change(90).dropna() * 100

        c1, c2 = st.columns(2)
        with c1:
            fig_r30 = px.line(x=df['Date'][30:], y=rolling_30, title='30-Day Rolling Returns (%)')
            fig_r30.update_layout(showlegend=False, xaxis_title='Date', yaxis_title='Return (%)')
            st.plotly_chart(fig_r30, use_container_width=True, key="roll30")
        with c2:
            fig_r90 = px.line(x=df['Date'][90:], y=rolling_90, title='90-Day Rolling Returns (%)')
            fig_r90.update_layout(showlegend=False, xaxis_title='Date', yaxis_title='Return (%)')
            st.plotly_chart(fig_r90, use_container_width=True, key="roll90")

# --- Tab 3 ---
with tab3:
    st.subheader("Maximum Drawdown Analysis")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Maximum Drawdown", f"{dd_info['max_drawdown']*100:.2f}%")
        if analysis_mode == "Portfolio Analysis":
            st.write(f"**Dollar Impact:** ${total_inv * dd_info['max_drawdown']:,.2f}")
        st.write(f"**Peak Date:** {pd.to_datetime(dd_info['peak_date']).strftime('%Y-%m-%d')}")
        prefix = "$" if analysis_mode == "Single Stock" else ""
        label = "Peak Price" if analysis_mode == "Single Stock" else "Peak Value"
        if analysis_mode == "Portfolio Analysis":
            st.write(f"**{label}:** ${dd_info['peak_price'] * total_inv / 100:,.2f}")
        else:
            st.write(f"**{label}:** {prefix}{dd_info['peak_price']:.2f}")
    with c2:
        peak_dt = pd.to_datetime(dd_info['peak_date'])
        trough_dt = pd.to_datetime(dd_info['trough_date'])
        st.metric("Drawdown Period", f"{(trough_dt - peak_dt).days} days")
        st.write(f"**Trough Date:** {trough_dt.strftime('%Y-%m-%d')}")
        label = "Trough Price" if analysis_mode == "Single Stock" else "Trough Value"
        if analysis_mode == "Portfolio Analysis":
            st.write(f"**{label}:** ${dd_info['trough_price'] * total_inv / 100:,.2f}")
        else:
            st.write(f"**{label}:** {prefix}{dd_info['trough_price']:.2f}")

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=df['Date'], y=dd_info['drawdown_series'] * 100,
        fill='tozeroy', name='Drawdown %',
        line=dict(color='red'), fillcolor='rgba(255, 0, 0, 0.3)'
    ))
    fig_dd.add_trace(go.Scatter(
        x=[pd.to_datetime(dd_info['trough_date'])],
        y=[dd_info['max_drawdown'] * 100],
        mode='markers', name='Max Drawdown',
        marker=dict(color='darkred', size=12, symbol='x')
    ))

    title = f'{ticker} Drawdown Over Time' if analysis_mode == "Single Stock" else 'Portfolio Drawdown Over Time'
    fig_dd.update_layout(title=title, xaxis_title='Date', yaxis_title='Drawdown (%)', hovermode='x unified')
    st.plotly_chart(fig_dd, use_container_width=True, key="dd_chart")

    if analysis_mode == "Portfolio Analysis" and benchmark_data:
        st.subheader("Drawdown Comparison with Benchmarks")
        comp = [{
            'Asset': 'Your Portfolio',
            'Max Drawdown': f"{dd_info['max_drawdown']*100:.2f}%",
            'Dollar Impact': f"${total_inv * dd_info['max_drawdown']:,.2f}",
            'Period (days)': (trough_dt - peak_dt).days
        }]
        for bname, bdf in benchmark_data.items():
            bdd = calculate_max_drawdown(bdf)
            if bdd:
                bp = pd.to_datetime(bdd['peak_date'])
                bt = pd.to_datetime(bdd['trough_date'])
                comp.append({
                    'Asset': bname,
                    'Max Drawdown': f"{bdd['max_drawdown']*100:.2f}%",
                    'Dollar Impact': f"${total_inv * bdd['max_drawdown']:,.2f}",
                    'Period (days)': (bt - bp).days
                })
        st.dataframe(pd.DataFrame(comp), use_container_width=True, hide_index=True)

    st.markdown("""
    **What is Maximum Drawdown?**
    Maximum drawdown (MDD) measures the largest peak-to-trough decline in value.
    It's a key risk metric showing the worst loss an investor would have experienced
    if they bought at the peak and sold at the lowest point during the period.
    """)

# --- Tab 4: Correlation (Portfolio only) ---
if analysis_mode == "Portfolio Analysis":
    with tab4:
        st.subheader("Correlation Matrix")

        returns_df = pd.DataFrame()
        for i, stock_data in enumerate(portfolio_stocks):
            tname = st.session_state.portfolio[i]['ticker']
            returns_df[tname] = stock_data['Price'].pct_change()

        returns_df = returns_df.dropna()
        corr_matrix = returns_df.corr()

        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns, y=corr_matrix.columns,
            colorscale='RdBu', zmid=0,
            text=corr_matrix.values, texttemplate='%{text:.2f}',
            textfont={"size": 12},
            colorbar=dict(title="Correlation")
        ))
        fig_corr.update_layout(title='Correlation Matrix', width=600, height=600)
        st.plotly_chart(fig_corr, use_container_width=True, key="corr_matrix")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            **Understanding Correlation:**
            - **+1.0**: Perfect positive correlation
            - **0.0**: No correlation
            - **-1.0**: Perfect negative correlation

            **For Diversification:**
            - Aim for correlations < 0.7
            - Negative correlations provide hedge protection
            """)
        with c2:
            cnd = corr_matrix.values.copy()
            np.fill_diagonal(cnd, np.nan)
            mx = np.unravel_index(np.nanargmax(cnd), cnd.shape)
            mn = np.unravel_index(np.nanargmin(cnd), cnd.shape)
            avg = np.nanmean(cnd)

            st.markdown("**Portfolio Insights:**")
            st.write(f"**Highest:** {corr_matrix.columns[mx[0]]} & {corr_matrix.columns[mx[1]]}: {cnd[mx]:.2f}")
            st.write(f"**Lowest:** {corr_matrix.columns[mn[0]]} & {corr_matrix.columns[mn[1]]}: {cnd[mn]:.2f}")
            st.write(f"**Average:** {avg:.2f}")

            if avg > 0.7:
                st.caption("High average correlation — portfolio may not be well diversified.")
            elif avg < 0.3:
                st.caption("Low average correlation — good diversification.")
            else:
                st.caption("Moderate average correlation — reasonable diversification.")

# --- About Tab ---
about_tab = tab4 if analysis_mode == "Single Stock" else tab5

with about_tab:
    st.subheader("About This Dashboard")
    st.markdown("""
    **Real market data** from Yahoo Finance with to visualize how a portfolio would have performed if invested anywhere from 30 days to 2 years ago.

    ---

    ### Dollar Amount Mode

    You can set your total portfolio value (e.g., $10,000) and allocate stocks
    using dollar amounts instead of percentages. Switch between modes using the
    **Input Mode** toggle in the sidebar.

    | Mode | Slider Shows | Example |
    |---|---|---|
    | **Dollar Amount** | $0 – $10,000 | Drag AAPL to $4,000 |
    | **Percentage** | 0% – 100% | Drag AAPL to 40% |

    Both modes maintain the same underlying weights — switching modes simply
    changes how you interact with the sliders.

    ---

    ### How the Lock System Works

    | Action | What Happens |
    |---|---|
    | **Lock a stock** | Its weight is frozen. No operation will change it. |
    | **Move an unlocked slider** | Only the *other unlocked* stocks adjust to keep total at 100%. |
    | **Equal (unlocked)** | Locked stocks stay. Remaining budget splits evenly among unlocked. |
    | **Equal (all)** | Ignores all locks. Splits 100% evenly. |
    | **Normalize** | Rescales everything (locked and unlocked) to 100%. |
    | **Unlock All** | Removes every lock. |
    | **Add a stock** | If auto-rebalance is on, unlocked stocks shrink to make room. |
    | **Remove a stock** | If auto-rebalance is on, unlocked stocks grow to fill the gap. |
    """)

# Footer
st.divider()
st.caption("Portfolio Analysis Dashboard | Powered by yfinance & Streamlit")
import yfinance as yf
import mysql.connector
from mysql.connector import Error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import warnings
warnings.filterwarnings('ignore')

def setup_database(db_config):
    """Initialize database and create tables"""
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        
        # acts like grabage collector to free up memory 

        cursor.execute("DROP TABLE IF EXISTS technical_indicators")
        cursor.execute("DROP TABLE IF EXISTS price_history")
        cursor.execute("DROP TABLE IF EXISTS stocks")
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stocks (
                id INT AUTO_INCREMENT PRIMARY KEY,
                symbol VARCHAR(40) NOT NULL,
                name VARCHAR(255),
                sector VARCHAR(50),
                market_cap BIGINT,
                pe_ratio DECIMAL(10, 2),
                dividend_yield DECIMAL(5, 2),
                beta DECIMAL(5, 2),
                eps DECIMAL(10, 2),
                UNIQUE KEY unique_symbol (symbol)
            )
        """)
        
        # Create price history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS price_history (
                id INT AUTO_INCREMENT PRIMARY KEY,
                symbol VARCHAR(40) NOT NULL,
                date DATE NOT NULL,
                open DECIMAL(10, 2),
                high DECIMAL(10, 2),
                low DECIMAL(10, 2),
                close DECIMAL(10, 2),
                volume BIGINT,
                adj_close DECIMAL(10, 2),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY unique_symbol_date (symbol, date),
                FOREIGN KEY (symbol) REFERENCES stocks(symbol)
            )
        """)
        
        # Create technical indicators table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS technical_indicators (
                id INT AUTO_INCREMENT PRIMARY KEY,
                symbol VARCHAR(40) NOT NULL,
                date DATE NOT NULL,
                sma_5 DECIMAL(10, 2),
                sma_10 DECIMAL(10, 2),
                sma_20 DECIMAL(10, 2),
                sma_50 DECIMAL(10, 2),
                ema_12 DECIMAL(10, 2),
                ema_26 DECIMAL(10, 2),
                rsi DECIMAL(5, 2),
                macd DECIMAL(10, 4),
                macd_signal DECIMAL(10, 4),
                macd_histogram DECIMAL(10, 4),
                bb_upper DECIMAL(10, 2),
                bb_lower DECIMAL(10, 2),
                bb_middle DECIMAL(10, 2),
                atr DECIMAL(10, 2),
                stoch_k DECIMAL(5, 2),
                stoch_d DECIMAL(5, 2),
                williams_r DECIMAL(5, 2),
                momentum DECIMAL(10, 4),
                roc DECIMAL(5, 2),
                cci DECIMAL(10, 2),
                obv BIGINT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY unique_symbol_date (symbol, date),
                FOREIGN KEY (symbol) REFERENCES stocks(symbol)
            )
        """)

        connection.commit()
        cursor.close()
        print("Database setup completed successfully")
        return connection
    except Error as e:
        print(f"Error setting up database: {e}")
        return None

def get_stock_sym(company_name):
    """Convert company name to stock symbol"""
    search_url = "https://query2.finance.yahoo.com/v1/finance/search"
    params = {
        "q": company_name,
        "quotes_count": 1,  
        "news_count": 0,
        "enable_fuzzy_query": False,
        "enable_related_link": False,
    }
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    try:
        response = requests.get(search_url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        if 'quotes' in data and data['quotes']:
            # Get the first matching symbol
            symbol = data['quotes'][0].get('symbol', 'Symbol not found')
            name = data['quotes'][0].get('longname', data['quotes'][0].get('shortname', 'Unknown'))
            return f'company name: {name} | stock symbol: {symbol}', symbol
        else:
            return "Error!! No matching stock symbol found for this company name.", None

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Yahoo Finance: {e}")
        print("Note: Yahoo Finance services may be limited in some regions.")
        return f"Error fetching data: {e}", None
    except KeyError:
        return "Error parsing data from Yahoo Finance.", None
    except Exception as e:
        return f"An unexpected error occurred: {e}", None

def fetch_stock_data(symbol, period="1y"):
    """Fetch stock data """
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        info = stock.info
        return data, info
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None, None

def store_stock_info(connection, symbol, info):
    """Store comprehensive stock information"""
    cursor = connection.cursor()
    try:
        cursor.execute("""
            INSERT INTO stocks (symbol, name, sector, market_cap, pe_ratio, dividend_yield, beta, eps) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s) 
            ON DUPLICATE KEY UPDATE 
            name=VALUES(name), sector=VALUES(sector), market_cap=VALUES(market_cap),
            pe_ratio=VALUES(pe_ratio), dividend_yield=VALUES(dividend_yield),
            beta=VALUES(beta), eps=VALUES(eps)
        """, (
            symbol,
            info.get('longName', symbol),
            info.get('sector', 'N/A'),
            info.get('marketCap'),
            info.get('trailingPE'),
            info.get('dividendYield'),
            info.get('beta'),
            info.get('trailingEps')
        ))
        connection.commit()
    except Error as e:
        print(f"Error storing stock info: {e}")
    finally:
        cursor.close()

def calculate_technical_indicators(df):
    """technical indicators"""
    # SMV (Simple Moving Averages)
    df['sma_5'] = df['Close'].rolling(window=5).mean()
    df['sma_10'] = df['Close'].rolling(window=10).mean()
    df['sma_20'] = df['Close'].rolling(window=20).mean()
    df['sma_50'] = df['Close'].rolling(window=50).mean()
    
    # EMA (Exponential Moving Averages)
    df['ema_12'] = df['Close'].ewm(span=12).mean()
    df['ema_26'] = df['Close'].ewm(span=26).mean()
    
    # MACD (Moving Average Convergence Divergence)
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['bb_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    
    # ATR (Average True Range)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['atr'] = true_range.rolling(window=14).mean()
    
    # Stochastic Oscillator
    df['stoch_k'] = (df['Close'] - df['Low'].rolling(window=14).min()) / (df['High'].rolling(window=14).max() - df['Low'].rolling(window=14).min()) * 100
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    
    # Williams %R
    df['williams_r'] = (df['High'].rolling(window=14).max() - df['Close']) / (df['High'].rolling(window=14).max() - df['Low'].rolling(window=14).min()) * -100
    
    # Momentum
    df['momentum'] = df['Close'] - df['Close'].shift(10)
    
    # Rate of Change
    df['roc'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
    

    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    df['cci'] = (typical_price - typical_price.rolling(window=20).mean()) / (0.015 * typical_price.rolling(window=20).std())
    
    df['obv'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    return df

def store_price_history(connection, symbol, data):
    """Store historical price data"""
    cursor = connection.cursor()
    try:
        for date, row in data.iterrows():
            cursor.execute("""
                INSERT INTO price_history 
                (symbol, date, open, high, low, close, volume, adj_close)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                open=VALUES(open), high=VALUES(high), low=VALUES(low),
                close=VALUES(close), volume=VALUES(volume), adj_close=VALUES(adj_close)
            """, (
                symbol, date.strftime('%Y-%m-%d'),
                row['Open'], row['High'], row['Low'],
                row['Close'], row['Volume'], row['Close']
            ))
        connection.commit()
    except Error as e:
        print(f"Error storing price history: {e}")
    finally:
        cursor.close()

def store_technical_indicators(connection, symbol, df):
    """Store calculated technical indicators"""
    cursor = connection.cursor()
    try:
        for date, row in df.iterrows():
            if pd.notna(row[['sma_5', 'sma_10', 'sma_20', 'sma_50', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'macd_histogram', 'bb_upper', 'bb_lower', 'bb_middle', 'atr', 'stoch_k', 'stoch_d', 'williams_r', 'momentum', 'roc', 'cci', 'obv']]).all():
                cursor.execute("""
                    INSERT INTO technical_indicators 
                    (symbol, date, sma_5, sma_10, sma_20, sma_50, ema_12, ema_26, rsi, macd, macd_signal, 
                     macd_histogram, bb_upper, bb_lower, bb_middle, atr, stoch_k, stoch_d, williams_r, 
                     momentum, roc, cci, obv)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                    sma_5=VALUES(sma_5), sma_10=VALUES(sma_10), sma_20=VALUES(sma_20), sma_50=VALUES(sma_50),
                    ema_12=VALUES(ema_12), ema_26=VALUES(ema_26), rsi=VALUES(rsi), macd=VALUES(macd),
                    macd_signal=VALUES(macd_signal), macd_histogram=VALUES(macd_histogram),
                    bb_upper=VALUES(bb_upper), bb_lower=VALUES(bb_lower), bb_middle=VALUES(bb_middle),
                    atr=VALUES(atr), stoch_k=VALUES(stoch_k), stoch_d=VALUES(stoch_d),
                    williams_r=VALUES(williams_r), momentum=VALUES(momentum), roc=VALUES(roc),
                    cci=VALUES(cci), obv=VALUES(obv)
                """, (
                    symbol, date.strftime('%Y-%m-%d'), row['sma_5'], row['sma_10'], row['sma_20'],
                    row['sma_50'], row['ema_12'], row['ema_26'], row['rsi'], row['macd'],
                    row['macd_signal'], row['macd_histogram'], row['bb_upper'], row['bb_lower'],
                    row['bb_middle'], row['atr'], row['stoch_k'], row['stoch_d'], row['williams_r'],
                    row['momentum'], row['roc'], row['cci'], row['obv']
                ))
        connection.commit()
    except Error as e:
        print(f"Error storing technical indicators: {e}")
    finally:
        cursor.close()

def update_stock_data(connection, symbol):
    """Update stock data with latest information"""
    data, info = fetch_stock_data(symbol, period="1y")
    if data is not None and not data.empty:
        store_stock_info(connection, symbol, info)
        store_price_history(connection, symbol, data)
        df_with_indicators = calculate_technical_indicators(data.copy())
        store_technical_indicators(connection, symbol, df_with_indicators) # Calculate and store technical indicators
        
        print(f"Data for {symbol} updated successfully")

def get_comprehensive_analysis(connection, symbol):
    """Get comprehensive stock analysis with all indicators"""
    query = """
        SELECT 
            p.date, p.close, p.volume, p.high, p.low,
            t.sma_5, t.sma_10, t.sma_20, t.sma_50, t.ema_12, t.ema_26,
            t.rsi, t.macd, t.macd_signal, t.macd_histogram, t.bb_upper, t.bb_lower, t.bb_middle,
            t.atr, t.stoch_k, t.stoch_d, t.williams_r, t.momentum, t.roc, t.cci, t.obv
        FROM price_history p
        LEFT JOIN technical_indicators t ON p.symbol = t.symbol AND p.date = t.date
        WHERE p.symbol = %s
        ORDER BY p.date DESC
        LIMIT 252  -- 1 year of trading days
    """
    df = pd.read_sql(query, connection, params=[symbol])
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    
    # Calculate additional metrics
    df['daily_return'] = df['close'].pct_change()
    df['volatility'] = df['daily_return'].rolling(window=21).std() * np.sqrt(252)
    df['price_change_pct'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1) * 100
    df['volume_change_pct'] = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1) * 100
    df['high_low_pct'] = (df['high'] - df['low']) / df['close'] * 100
    
    # Generate signals
    df['sma_signal'] = np.where(df['sma_20'] > df['sma_50'], 1, 0)
    df['rsi_signal'] = np.where(df['rsi'] < 30, 1, np.where(df['rsi'] > 70, -1, 0))
    df['macd_signal'] = np.where(df['macd'] > df['macd_signal'], 1, 0)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    return df

def generate_detailed_report(connection, symbol):
    """Generate detailed analysis report"""
    df = get_comprehensive_analysis(connection, symbol)
    if df.empty:
        print(f"No data found for {symbol}")
        return
    
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    print(f"\n{'='*50}")
    print(f"DETAILED {symbol} ANALYSIS REPORT")
    print(f"{'='*50}")
    
    # Basic Information
    stock_info_query = "SELECT * FROM stocks WHERE symbol = %s"
    stock_info = pd.read_sql(stock_info_query, connection, params=[symbol]).iloc[0]
    
    print(f"Company: {stock_info['name']}")
    print(f"Sector: {stock_info['sector']}")
    print(f"Market Cap: ${stock_info['market_cap']:,}" if not pd.isna(stock_info['market_cap']) else "Market Cap: N/A")
    print(f"P/E Ratio: {stock_info['pe_ratio']:.2f}" if not pd.isna(stock_info['pe_ratio']) else "P/E Ratio: N/A")
    print(f"Dividend Yield: {stock_info['dividend_yield']:.2%}" if not pd.isna(stock_info['dividend_yield']) else "Dividend Yield: N/A")
    print(f"Beta: {stock_info['beta']:.2f}" if not pd.isna(stock_info['beta']) else "Beta: N/A")
    
    print(f"\nCurrent Metrics:")
    print(f"Current Price: ${latest['close']:.2f}")
    print(f"Daily Change: {latest['daily_return']:.2%}")
    print(f"Price Change %: {latest['price_change_pct']:.2f}%")
    print(f"Volume: {latest['volume']:,}")
    print(f"Volume Change %: {latest['volume_change_pct']:.2f}%")
    print(f"High-Low %: {latest['high_low_pct']:.2f}%")
    print(f"RSI: {latest['rsi']:.3f}")
    print(f"Volatility: {latest['volatility']:.2%}")
    
    # Technical Signals
    signals = {
        'SMA (20/50)': 'Bullish' if latest['sma_signal'] == 1 else 'Bearish',
        'RSI': 'Oversold' if latest['rsi_signal'] == 1 else 'Overbought' if latest['rsi_signal'] == -1 else 'Neutral',
        'MACD': 'Bullish' if latest['macd_signal'] == 1 else 'Bearish',
        'Bollinger Position': 'Overbought' if latest['bb_position'] > 0.8 else 'Oversold' if latest['bb_position'] < 0.2 else 'Neutral'
    }
    
    print(f"\nTechnical Signals:")
    for indicator, signal in signals.items():
        print(f"  {indicator}: {signal}")
    
    # Additional Analysis
    print(f"\nAdditional Analysis:")
    print(f"52-Week High: ${df['high'].max():.2f}")
    print(f"52-Week Low: ${df['low'].min():.2f}")
    print(f"Current Position: {((latest['close'] - df['low'].min()) / (df['high'].max() - df['low'].min())) * 100:.2f}% of 52-week range")
    print(f"Average Volume: {df['volume'].mean():,.0f}")
    print(f"Average True Range: {latest['atr']:.2f}")
    print(f"Stochastic K: {latest['stoch_k']:.2f}")
    print(f"Stochastic D: {latest['stoch_d']:.2f}")
    print(f"Williams %R: {latest['williams_r']:.2f}")
    print(f"Momentum: {latest['momentum']:.2f}")
    print(f"Rate of Change: {latest['roc']:.2f}%")
    print(f"Commodity Channel Index: {latest['cci']:.2f}")
    print(f"On Balance Volume: {latest['obv']:,.0f}")

def plot_comprehensive_analysis(connection, symbol):
    """Plot comprehensive technical analysis"""
    df = get_comprehensive_analysis(connection, symbol)
    if df.empty:
        return
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))

    # Price and moving averages
    ax1 = axes[0]
    ax1.plot(df.index, df['close'], label='Close Price', linewidth=1.5)
    ax1.plot(df.index, df['sma_20'], label='SMA 20', alpha=0.7)
    ax1.plot(df.index, df['sma_50'], label='SMA 50', alpha=0.7)
    ax1.fill_between(df.index, df['bb_lower'], df['bb_upper'], alpha=0.1, label='Bollinger Bands')
    ax1.set_title(f'{symbol} - Price and Moving Averages')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # RSI and Stochastic
    ax2 = axes[1]
    ax2.plot(df.index, df['rsi'], label='RSI', color='purple', linewidth=1)
    ax2.plot(df.index, df['stoch_k'], label='Stoch K', color='blue', alpha=0.7)
    ax2.plot(df.index, df['stoch_d'], label='Stoch D', color='orange', alpha=0.7)
    ax2.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought')
    ax2.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold')
    ax2.set_title('RSI and Stochastic Oscillator')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # MACD
    ax3 = axes[2]
    ax3.plot(df.index, df['macd'], label='MACD', color='blue')
    ax3.plot(df.index, df['macd_signal'], label='Signal', color='red')
    if 'macd_histogram' in df.columns:
        ax3.bar(df.index, df['macd_histogram'], label='Histogram', alpha=0.3, color='gray')
    ax3.set_title('MACD')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Volume and OBV
    ax4 = axes[3]
    ax4.bar(df.index, df['volume'], alpha=0.6, label='Volume', color='lightblue')
    ax4_twin = ax4.twinx()
    ax4_twin.plot(df.index, df['obv'], label='OBV', color='purple', linewidth=2)
    ax4.set_title('Volume and On Balance Volume (OBV)')
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    fig.tight_layout(rect=[0.024, 0.526, 0.483, 0.974]) 

    plt.get_current_fig_manager().window.state('zoomed')
    plt.show()

def main():
    print("=" * 50)
    print(" "* 10,"WELCOME TO STOCK ANALYZER")
    print(" "* 5,"Comprehensive Stock Analysis Tool")
    print("=" * 50)
    
    # Database configuration
    db_config = {
        'host': 'localhost',
        'database': 'stocks_info',
        'user': 'root',
        'password': 'password'
    }
    
    # Initialize database
    connection = setup_database(db_config)
    
    if not connection:
        print("Failed to connect to database. Exiting...")
        return
    
    while True:
        print("\n" + "-" * 50)
        company_name = input("Enter the company name you want to analyze (or 'quit' to exit): ").strip()
        
        if company_name.lower() == 'quit':
            print("Thank you for using Stock Analyzer. Goodbye!")
            break
        
        if not company_name:
            print("Please enter a valid company name.")
            continue
        
        result, symbol = get_stock_sym(company_name)
        
        if not symbol:
            print(result)
            continue
        
        print(f"\n{result}")
        
        # Update stock data
        update_stock_data(connection, symbol)
        
        while True:
            print(f"\nWhat would you like to do with {symbol}?")
            print("1. View Detailed Analysis Report")
            print("2. View Technical Analysis Chart")
            print("3. Search for another company")
            print("4. Exit")
            
            choice = input("Enter your choice (1-4): ").strip()
            
            if choice == '1':
                generate_detailed_report(connection, symbol)
            elif choice == '2':
                plot_comprehensive_analysis(connection, symbol)
            elif choice == '3':
                break  
            elif choice == '4':
                print("Thank you for using Stock Analyzer. Goodbye!")
                connection.close()
                return
            else:
                print("Invalid choice. Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()

def calculate_rsi(series, period=14):
    '''
    Calcula el RSI (Relative Strength Index) para una serie de precios.
    El RSI es un indicador de momentum que mide la velocidad y el cambio de los movimientos de precios.
    Este indicador nos indica cuando un activo ha subido demasiado, para no comprarlo.
    '''
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series):
    '''
    Calcula el MACD (Moving Average Convergence Divergence) para una serie de precios.
    El MACD es un indicador de momentum que muestra la relación entre dos medias móviles de los
    precios. Nos indica cuándo un activo comienza a subir, para comprarlo.
    '''
    exp1 = series.ewm(span=12, adjust=False).mean()
    exp2 = series.ewm(span=26, adjust=False).mean()
    return exp1 - exp2

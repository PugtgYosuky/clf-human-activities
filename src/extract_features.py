import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import fft
from scipy.signal import periodogram


def spectral_entropy(data, freq):
    """
    Calculate the spectral entropy of the signal
    from https://towardsdatascience.com/anomaly-detection-in-univariate-stochastic-time-series-with-spectral-entropy-834b63ec9343
    @param data: signal
    @param freq: sample rate
    @return: the calculated spectral entropy of the signal
    """
    _, spec_density = periodogram(data, freq, nfft=None)
    spec_density_norm = spec_density / np.sum(spec_density)
    entropy = np.nansum(spec_density_norm * np.log2(spec_density_norm))
    res = -(entropy / np.log2(spec_density_norm.size))
    return res


def extract_statistical_features(data, fs):
    """
    Receives the dataframe in a given windoand extracts the statistical features
    @param data: data to analyse
    @param fs: sample rate (Hz)
    @return: the statistical features extracted
    """
    df = pd.DataFrame()
    for column in data.columns:
        df[f'{column}_mean'] = [data[column].mean()]
        # mean
        # median
        df[f'{column}_median'] = [data[column].median()]
        # Std
        df[f'{column}_std'] = [data[column].std()]
        # variance
        df[f'{column}_variance'] = [data[column].var()]
        # Root mean square
        df[f'{column}_rms'] = [np.sqrt((data[column] ** 2).mean())]
        # average derivatives
        df[f'{column}_avg_derivatives'] = [np.mean(np.gradient(data[column], edge_order=1))]
        # Skewness
        df[f'{column}_skewness'] = [data[column].skew()]
        # kurtosis
        df[f'{column}_kurtosis'] = [data[column].kurtosis()]
        # Interquartile range
        df[f'{column}_interquantile'] = [stats.iqr(data[column])]
        # zero crossing rate
        # https://www.folkstalk.com/tech/zero-crossing-rate-python-with-code-examples/
        df[f'{column}_non_crossing_rate'] = [len(np.nonzero(np.diff(data[column] > 0))[0])]
        # mean crossing rate
        aux = data[column] - data[column].mean()  # to centre in zero
        df[f'{column}_mean_crossing_rate'] = [len(np.nonzero(np.diff(aux > 0))[0])]

        # spectral entropy
        df[f'{column}_spectral_entropy'] = [spectral_entropy(data[column], fs)]
    # pairwise correlation
    for i in range(len(data.columns)):
        for j in range(i + 1, len(data.columns)):
            df[f'corr_{data.columns[i]}_{data.columns[j]}'] = [
                np.corrcoef(data[data.columns[i]], data[data.columns[j]])[0, 1]]

    return df


# *** PHYSICAL FEATURES ***

# Physical features
def mi(data):
    return np.sqrt((data ** 2).sum(axis=1))


def ai(mi_data):
    return mi_data.mean()


def vi(mi_data):
    return mi_data.var()


def sma(data):
    return np.sum(data.abs().sum()) / len(data)


def eva(data):
    cov_data = data.cov()
    eigenvalues, _ = np.linalg.eig(cov_data)
    return eigenvalues[0], eigenvalues[1]


def cagh(data):
    """Accelerometer data"""
    norm = np.sqrt((data ** 2).sum(axis=1))
    coef = np.corrcoef(data['accelerometer_x'], norm)
    return coef[0, 1]


def avh(data, time):
    """Accelerometer data"""
    velo_y = data['accelerometer_y'].mean() * time
    velo_z = data['accelerometer_z'].mean() * time
    return np.sqrt(velo_y ** 2 + velo_z ** 2)


def avg(data):
    """Accelerometer data"""
    return np.trapz(data['accelerometer_x'])


def aratg(data):
    """Gyroscope data """
    aux = data['gyroscope_x'].sum() / len(data)
    return aux


def dominant_frequency(data):
    return np.argmax(fft(data.to_numpy()) ** 2)


def energy(data):
    return np.sum(np.abs(fft(data.to_numpy())) ** 2) / len(data)


def extract_physical_all(data):
    """
    Extracts the physical features (dominant frequency and energy) that are calculated for the tree vectors: accelerometer, gyroscpe and magnetometer
    @param data: signal
    @return: the extracted features of the signal
    """
    df = pd.DataFrame()
    for column in data.columns:
        # dominant frequency
        df[f'{column}_df'] = [dominant_frequency(data[column])]
        df[f'{column}_energy'] = [energy(data)]

    return df


accelerometer_columns = ['accelerometer_x', 'accelerometer_y', 'accelerometer_z']
gyroscope_columns = ['gyroscope_x', 'gyroscope_y', 'gyroscope_z']


def extract_physical_features(data, window_periods):
    """
    Extracts the physical features of the dataframe
    @param data: signal to use as input
    @param window_periods: window periods used
    @return: the extracted physical features
    """
    df = extract_physical_all(data)
    # MI - movement intensity - independent od the orientation not used
    mi_values = mi(data[accelerometer_columns])
    # eigenvalues of dominant directions
    eva_values = eva(data[accelerometer_columns])
    # AI - MI mean
    df['ai'] = [ai(mi_values)]
    # VI - MI variance
    df['vi'] = [vi(mi_values)]
    # SMA - normalized signal magnitude area
    # df['sma'] = [sma(data)]
    # EVA - eigenvalues of dominant directions
    df['eva_1'] = [eva_values[0]]
    df['eva_2'] = [eva_values[1]]
    # CAGH - correlation between acceleration along gravity and heading directions
    df['cagh'] = [cagh(data[accelerometer_columns])]
    # AVH - averaged velocity along heading direction
    df['avh'] = [avh(data[accelerometer_columns], window_periods)]
    # AVG - averaged velocity along gravity direction
    df['avg'] = [avg(data[accelerometer_columns])]
    # ARATG - average rotation angles related to gravity direction
    df['aratg'] = [aratg(data[gyroscope_columns])]
    # AAE averaged acceleration energy
    df['aae'] = np.mean([df['accelerometer_x_energy'], df['accelerometer_y_energy'], df['accelerometer_z_energy']])
    # ARE - averaged rotation energy
    df['are'] = np.mean([df['gyroscope_x_energy'], df['gyroscope_y_energy'], df['gyroscope_z_energy']])

    return df


def extract_features(data, window_size, fs, step, window_periods):
    """
    Extracts all the statistical and physical features of the data, by a given window
    @param data: data where to extract the features
    @param window_size: window size to use
    @param fs: sample rate
    @param step: step to use for each window
    @param window_periods: window periods
    @return: the extracted features
    """
    physical_dataframes = []
    statistical_dataframes = []
    activities = data.pop('activity')
    activities_series = []
    count = 0
    for i in range(window_size, len(data), step):
        physical_dataframes.append(extract_physical_features(data.iloc[i - window_size:i, :], window_periods))
        statistical_dataframes.append(extract_statistical_features(data.iloc[i - window_size:i, :], fs))
        activities_series.append(activities[i - window_size:i].value_counts().index[0])
        count += 1
    physical = pd.concat(physical_dataframes)
    statistical = pd.concat(statistical_dataframes)

    df =  pd.concat([statistical, physical], axis=1)
    df['activity'] = activities_series
    return df
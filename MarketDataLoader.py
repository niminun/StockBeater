import numpy as np
import pandas as pd
import pandas_datareader as fetch
import datetime


DUMP_SUFFIX = '.pickle'

MIN_YEAR = 1996
NUM_RECORDS_PER_SAMPLE = 30
OPTIONS = ["Adj Close"]


class MarketDataLoader(object):

    def __init__(self, symbols, option="Adj Close", train_set_ratio=0.8, records_per_sample=NUM_RECORDS_PER_SAMPLE,
                 fetch_data=False, first_year=MIN_YEAR):
        self.symbols = symbols
        self.option = option
        self.train_set_ratio = train_set_ratio
        self.records_per_sample = records_per_sample
        self.train_indices = None
        self.test_indices = None
        self.first_year = first_year
        self.data = None
        self.data_for_prediction = None
        self.used_symbols = None

        if fetch_data:
            self._fetch_all_data()
        else:
            self._load_all_data()

        self._generate_train_test()

################################# API ####################################

    def get_batch(self, batch_size, is_train):
        return self._generate_batch(batch_size, is_train)

    def get_data_for_prediction(self, date_to_predict, fetch_new=True):
        dump_file_path = "./" + self._generate_file_name(self.option) + "_pred.npy"
        if fetch_new:
            # taking some extra days prior to make sure we have at least num_records records.
            start_date = date_to_predict - datetime.timedelta(days=self.records_per_sample * 2)
            end_date = date_to_predict - datetime.timedelta(days=1)
            data = self._fetch_data(start_date, end_date)
            relevant_data = data.loc[:, self.used_symbols]
            observed = relevant_data.values[-self.records_per_sample:, ].T
            nan_idx = np.isnan(observed)
            if True in nan_idx:
                observed[np.isnan(observed)] = 1.0
            self.data_for_prediction = observed
            # dumping
            np.save(dump_file_path, self.data_for_prediction)
        else:
            self.data_for_prediction = np.load(dump_file_path)
        return [self.data_for_prediction]


############################# private methods #############################

    def _fetch_all_data(self):
        start_date = datetime.date(self.first_year, 1, 1)
        end_date = datetime.date.today()

        # fetching the data
        self.data = self._fetch_data(start_date, end_date)

        # dumping to pickle
        dump_file_path = "./" + self._generate_file_name(self.option) + DUMP_SUFFIX
        self.data.to_pickle(dump_file_path)
        # saving the data as matrix without nans as a class member
        self._save_clean_data()

    def _fetch_data(self, start_date, end_date):
        num_symbols = len(self.symbols)
        data = None
        i = 0
        while i < num_symbols:
            symbol = self.symbols[i]
            print("loading {0}\{1}: {2}".format(i + 1, num_symbols, symbol))
            try:
                stock_data = fetch.DataReader(symbol, 'yahoo', start_date, end_date)
                relevant_data = stock_data[self.option].to_frame()
                relevant_data.rename(columns={self.option: symbol}, inplace=True)
                if data is None:
                    data = relevant_data
                else:
                    data = data.merge(relevant_data, how='outer', left_index=True, right_index=True, sort=True)
                i += 1
            except Exception:
                print("failed, trying again.")
                pass
                # if data is None:
                #     return
                # date_list = data.index.values
                # relevant_data = pd.DataFrame(np.zeros((len(date_list), 1)),
                #                              index=date_list, columns=[symbol])

        return data

    def _load_all_data(self):
        self.data = pd.read_pickle("./" + self._generate_file_name(self.option) + DUMP_SUFFIX)
        # saving the data as matrix without nans as a class member
        self._save_clean_data()

    @staticmethod
    def _generate_file_name(option, year=''):
        return option.replace(' ', '_') + '_' + year

    def _generate_train_test(self):
        # splitting to train test
        num_records = self.data.shape[1]
        num_samples = num_records - self.records_per_sample
        rand_ind = np.random.permutation(num_samples)
        self.train_indices = rand_ind[: round(self.train_set_ratio * num_samples)]
        self.test_indices = rand_ind[round(self.train_set_ratio * num_samples):]

    def _generate_batch(self, batch_sz, is_train):

        if is_train:
            indices = self.train_indices
        else:
            indices = self.test_indices

        # creating observed data
        first_idx = np.random.choice(indices, batch_sz)
        first_idx_rep = np.repeat([first_idx], self.records_per_sample, axis=0)
        idx_add = np.repeat([np.arange(0, self.records_per_sample)], batch_sz, axis=0).T
        observed_idx = first_idx_rep + idx_add  # indices for the full batch.
        observed = self.data[:, observed_idx]
        observed = np.transpose(observed, (2, 0, 1))
        # creating the day after data
        day_after_idx = first_idx + self.records_per_sample
        day_after = self.data[:, day_after_idx]
        day_after = day_after.T
        return observed, day_after

    def _save_clean_data(self):
        self.used_symbols = self.data.columns.values
        self.data = self.data.values.T
        self.data[np.isnan(self.data)] = 1.0 #TODO- find a more clever way (find first non-nan).
        self.data_for_prediction = self.data[:, -self.records_per_sample:]

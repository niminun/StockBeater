import StockBeater
import sys
import datetime
import numpy as np

NUM_BEST_SYMBOLS = 10

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("USAGE: TradeManager.py <configuration file path>")
        exit()

    params_file = sys.argv[1]
    with open(params_file) as f:
        exec(f.read(), globals())
        params = BeaterParams()  # The name of the configuration class

    beater = StockBeater.StockBeater(params)
    if params.do_learn:
        beater.learn()
    else:
        pred = beater.predict(datetime.date.today()).flatten()
        print(pred)
        temp = np.argpartition(-pred, NUM_BEST_SYMBOLS)
        best_idx = temp[:NUM_BEST_SYMBOLS]
        best_symbols = beater.data_manager.used_symbols[best_idx]
        best_vals = pred[best_idx]

        print("Best possibilities are:")
        for i in range(NUM_BEST_SYMBOLS):
            print("{0}: {1}".format(best_symbols[i], best_vals[i]))



def plot_results():
    normalized_data, scaler = read_data()
    whale_folder_path = '/Users/thangnguyen/hust_project/cloud_autoscaling/data/lstm/whale/traffic/results/'
    whale_file_name = ['sli-2_batch-8_numunits-4_act-tanh_opt_adam_num_par-50',
                       'sli-3_batch-8_numunits-4_act-tanh_opt_adam_num_par-50',
                       'sli-4_batch-8_numunits-4_act-tanh_opt_adam_num_par-50',
                       'sli-5_batch-8_numunits-4_act-tanh_opt_adam_num_par-50']

    bp_folder_path = '/Users/thangnguyen/hust_project/cloud_autoscaling/data/lstm/bp/traffic/results/'
    bp_file_name = ['sli-5_batch-8_numunits-4_act-tanh_opt_adam',
                    'sli-2_batch-8_numunits-4_act-tanh_opt_adam',
                    'sli-3_batch-8_numunits-4_act-tanh_opt_adam',
                    'sli-4_batch-8_numunits-4_act-tanh_opt_adam']
    folder_path = bp_folder_path
    file_name = bp_file_name
    for _file_name in file_name:
        file_path = '{}{}.csv'.format(folder_path, _file_name)
        df = pd.read_csv(file_path)
        pred_data = df.values
        print(pred_data.shape)
        real_data = scaler.inverse_transform(normalized_data)
        real_data = real_data[-pred_data.shape[0]:]
        ax = plt.subplot()
        ax.plot(real_data, label="Actual")
        ax.plot(pred_data, label="predictions")
        plt.xlabel("TimeStamp")
        plt.ylabel("Traffic")
        plt.legend()
        # plt.show()
        plt.savefig('{}{}.png'.format(folder_path, _file_name))
        plt.close()

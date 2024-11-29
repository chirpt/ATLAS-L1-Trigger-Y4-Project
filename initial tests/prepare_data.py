def prepare_data(test_size=0.2, accept_data_filename="l1calo_hist_EGZ_extended.root", 
                 reject_data_filename="l1calo_hist_ZMUMU_extended.root", 
                 save_path="prepared_data.npz"):
    accept_data_filename = os.path.join(os.path.pardir, "data", accept_data_filename)
    reject_data_filename = os.path.join(os.path.pardir, "data", reject_data_filename)
    save_path = os.path.join(os.path.pardir, "data", save_path)

    if not os.path.exists(save_path):
        print("Preparing data...")
        DFs = import_data_files([accept_data_filename, reject_data_filename])

        accepted_numpy = ak.to_numpy(DFs[0]['SuperCell_ET'])
        rejected_numpy = ak.to_numpy(DFs[1]['SuperCell_ET'])

        accepted_labels = np.ones(accepted_numpy.shape[0])
        rejected_labels = np.zeros(rejected_numpy.shape[0])

        data = np.concatenate((accepted_numpy, rejected_numpy), axis=0)
        labels = np.concatenate((accepted_labels, rejected_labels), axis=0)

        
        print(f"Saving prepared data to {save_path}")
        np.savez(save_path, data=data, labels=labels)
    
    else:
            print(f"Loading prepared data from {save_path}")
        
    data = np.load(save_path)
        
    X_train, X_test, y_train, y_test = train_test_split(data['data'], data['labels'], test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test

def prepare_data_2(test_size=0.2, accept_data_filename="l1calo_hist_EGZ_extended.root", reject_data_filename="l1calo_hist_ZMUMU_extended.root"):
    accept_data_filename= os.path.join(os.path.pardir, "data", "l1calo_hist_EGZ_extended.root")
    reject_data_filename= os.path.join(os.path.pardir, "data", "l1calo_hist_ZMUMU_extended.root")
    DFs = import_data_files([accept_data_filename, reject_data_filename])
    accepted_df = pd.DataFrame({'SuperCell_ET': DFs[0]['SuperCell_ET'], 'offline_ele_pt': DFs[0]['offline_ele_pt'],'Label': 1})
    rejected_df = pd.DataFrame({'SuperCell_ET': DFs[1]['SuperCell_ET'], 'offline_ele_pt': DFs[1]['offline_ele_pt'],'Label': 0})
    input_df = pd.concat([accepted_df,rejected_df]).reset_index(drop=True)

    X_train_all, X_test_all, y_train_pd, y_test_pd = train_test_split(input_df, input_df["Label"], test_size=0.2, random_state=42)
    X_train = ak.to_numpy(X_train_all['SuperCell_ET'])
    X_test = ak.to_numpy(X_test_all['SuperCell_ET'])
    y_train = y_train_pd.to_numpy()
    y_test = y_test_pd.to_numpy()

    return X_train, X_test, y_train, y_test, X_train_all, X_test_all

def prepare_data_3(test_size=0.2, accept_data_filename="l1calo_hist_EGZ_extended.root", reject_data_filename="l1calo_hist_ZMUMU_extended.root", data_subdir="ZMUMU_EGZ_extended_np_pd"):
    save_path = os.path.join(os.path.pardir, "data", data_subdir)
    if os.path.exists(os.path.join(save_path,"np_data.npz")) and os.path.exists(os.path.join(save_path,"input_df.parquet")):
        print(f"found preprepared data in {save_path}")
        np_data = np.load(os.path.join(save_path,"np_data.npz"))
        input_np, labels_np = np_data["input_np"], np_data["labels_np"]
        input_df = pd.read_parquet(os.path.join(save_path,"input_df.parquet"))

    else:
        print(f"preprepared data in {save_path} is missing, preparing and saving here")
        accept_data_path= os.path.join(os.path.pardir, "data", accept_data_filename)
        reject_data_path= os.path.join(os.path.pardir, "data", reject_data_filename)
        DFs = import_data_files([accept_data_path, reject_data_path])

        accepted_numpy = ak.to_numpy(DFs[0]['SuperCell_ET'])
        rejected_numpy = ak.to_numpy(DFs[1]['SuperCell_ET'])
        accepted_labels = np.ones(accepted_numpy.shape[0])
        rejected_labels = np.zeros(rejected_numpy.shape[0])

        accepted_df = pd.DataFrame({'offline_ele_pt': DFs[0]['offline_ele_pt'],'Label': 1})
        rejected_df = pd.DataFrame({'offline_ele_pt': DFs[1]['offline_ele_pt'],'Label': 0})

        input_np = np.concatenate((accepted_numpy, rejected_numpy), axis=0)
        input_df = pd.concat([accepted_df,rejected_df]).reset_index(drop=True)
        labels_np = np.concatenate((accepted_labels, rejected_labels), axis=0)


        if not os.path.exists(save_path):
            os.mkdir(save_path)
        np.savez(os.path.join(save_path,"np_data.npz"), input_np=input_np,labels_np=labels_np)
        input_df.to_parquet(os.path.join(save_path,"input_df.parquet"), index=False)

    X_train, X_test, pd_passthrough_train, pd_passthrough_test, y_train, y_test = train_test_split(input_np, input_df, labels_np, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test, pd_passthrough_train, pd_passthrough_test
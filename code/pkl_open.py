import pickle

pkl_file = open('../save/Fed-MOODS-Random_mnist_cnn_method[fedavg]_straggler[0.5]_iid[0]_LE[10]_B[128]_GE['
                '10]_adaptive[5].pkl', 'rb')
# mydict1 = pickle.load(pkl_file)
# pkl_file.close()

try:
    pdm_temp = pickle.load(pkl_file)
except UnicodeDecodeError:
    pdm_temp = pickle.load(pkl_file, fix_imports=True, encoding="latin1")
print(pdm_temp)

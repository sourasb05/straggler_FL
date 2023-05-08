import pickle
import sys

pkl_file = open(sys.argv[1], 'rb')
# mydict1 = pickle.load(pkl_file)
# pkl_file.close()

try:
    pdm_temp = pickle.load(pkl_file)
except UnicodeDecodeError:
    pdm_temp = pickle.load(pkl_file, fix_imports=True, encoding="latin1")
print(pdm_temp)

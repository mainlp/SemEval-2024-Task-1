import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


targets = ['afr', 'amh', 'hau', 'ind', 'kin', 'spa', 'ary', 'arb', 'hin']
transfers = ["spa", "kin", "hau", "amh"]


# language similarity based on language vectors

# same as learned vectors in lang2vec repo
vecs = np.load('lang_vecs/lang_vecs.npy', allow_pickle=True, encoding='bytes') 
#print("hau and kin: ", cosine_similarity(vecs_cell_states.item()[b'afr'][0].reshape(1,-1), vecs_cell_states.item()[b'kin'][0].reshape(1,-1))[0])

# NMT Encoder Mean Cell States
vecs_cell_states = np.load('lang_vecs/lang_cell_states.npy', allow_pickle=True, encoding='bytes') 


lang_vecs = {}
for transfer in transfers:
  for target in targets:
    transfer = transfer.encode('utf-8')
    target = target.encode('utf-8')
    print(f"{transfer} and {target}", cosine_similarity(vecs_cell_states.item()[target][0].reshape(1,-1), vecs_cell_states.item()[transfer][0].reshape(1,-1))[0])